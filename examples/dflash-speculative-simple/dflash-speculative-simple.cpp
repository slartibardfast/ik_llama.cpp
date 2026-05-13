// dflash-speculative-simple.cpp
//
// Minimal single-slot driver for DFlash speculative decoding.
// Drives common_speculative_init with type=DFLASH and exercises
// the full cycle:
//   target prefill (extract hook fires) -> sample first token
//     -> common_speculative_draft (DFlash kernel pipeline)
//     -> target verify decode at BLOCK_SIZE+1 positions
//     -> longest-prefix accept against per-position target argmax
//     -> emit accepted prefix + bonus token, advance state, repeat
//
// T5 closure binding (PHASE_DFLASH.md):
//   coherent output AND mean accept rate >= 1.0 over n_predict tokens
//
// Spec: specs/dflash/dflash.allium
//       specs/dflash/DESIGN.md
//
// Usage:
//   llama-dflash-speculative-simple \
//      -m TARGET.gguf -md DRAFTER.gguf --spec-type dflash \
//      -p "Write a python quicksort" -n 128

#include "common.h"
#include "speculative.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

static int64_t now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }
    if (params.speculative.type != COMMON_SPECULATIVE_TYPE_DFLASH) {
        fprintf(stderr, "error: --spec-type dflash is required for this example\n");
        return 1;
    }
    // common_speculative_init reads the drafter path from
    // params.speculative.mparams_dft.path; ensure it's set from the
    // -md flag.
    if (params.speculative.mparams_dft.path.empty()) {
        if (!params.speculative.model.empty()) {
            params.speculative.mparams_dft.path = params.speculative.model;
        } else {
            fprintf(stderr, "error: provide drafter GGUF via -md / --model-draft\n");
            return 1;
        }
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = llama_init_from_gpt_params(params);
    llama_model   * model = llama_init.model;
    llama_context * ctx   = llama_init.context;
    if (!model || !ctx) {
        fprintf(stderr, "error: failed to init target\n");
        return 1;
    }

    // Tokenize prompt.
    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, params.prompt, true);
    if (prompt_tokens.empty()) {
        fprintf(stderr, "error: empty prompt after tokenization\n");
        return 1;
    }
    const int n_prompt = (int) prompt_tokens.size();

    // Initialise the speculative state. This calls llama_dflash_drafter_load
    // and llama_set_dflash, which also installs the cb_eval extract hook
    // on the target context.
    common_speculative * spec = common_speculative_init(params.speculative, ctx, /*seq_id*/ 0);
    if (!spec) {
        fprintf(stderr, "error: failed to init DFlash speculative state\n");
        return 1;
    }
    common_speculative_begin(spec, prompt_tokens);

    // Sampler for emitting tokens (target-side).
    common_sampler * smpl = common_sampler_init(model, params.sparams);

    // Prefill prompt: decode in a single batch.
    {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; ++i) {
            common_batch_add(batch, prompt_tokens[i], i, { 0 }, /*logits*/ i == n_prompt - 1);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "error: prefill decode failed\n");
            llama_batch_free(batch);
            return 1;
        }
        llama_batch_free(batch);
    }

    // Sample first token from prefill logits.
    llama_token id_last = common_sampler_sample(smpl, ctx, n_prompt - 1);
    common_sampler_accept(smpl, ctx, id_last, true);

    // Emit prompt + first token to stdout.
    printf("%s%s", params.prompt.c_str(), common_token_to_piece(ctx, id_last).c_str());
    fflush(stdout);

    // Speculative loop.
    const int n_predict = params.n_predict > 0 ? params.n_predict : 128;
    std::vector<llama_token> emitted; emitted.reserve(n_predict);
    emitted.push_back(id_last);

    int64_t t0 = now_us();
    int n_drafts = 0;
    int n_draft_tokens = 0;
    int n_accept_tokens = 0;

    std::vector<llama_token> prompt_tgt = prompt_tokens;

    while ((int) emitted.size() < n_predict) {
        // Build the candidate context tokens for the drafter.
        prompt_tgt.assign(prompt_tokens.begin(), prompt_tokens.end());
        prompt_tgt.insert(prompt_tgt.end(), emitted.begin(), emitted.end() - 1);
        // After this, anchor = id_last sits at position prompt_tgt.size().

        const llama_tokens draft = common_speculative_draft(
            spec, params.speculative, prompt_tgt, id_last);
        n_drafts += 1;
        n_draft_tokens += (int) draft.size();

        if (draft.empty()) {
            // Drafter failed — fall back to a single non-spec decode.
            llama_batch batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, id_last, (llama_pos) prompt_tgt.size(), { 0 }, true);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "error: fallback decode failed\n");
                llama_batch_free(batch);
                break;
            }
            const llama_token nxt = common_sampler_sample(smpl, ctx, 0);
            common_sampler_accept(smpl, ctx, nxt, true);
            printf("%s", common_token_to_piece(ctx, nxt).c_str());
            fflush(stdout);
            emitted.push_back(nxt);
            id_last = nxt;
            llama_batch_free(batch);
            if (llama_token_is_eog(model, nxt)) break;
            continue;
        }

        // Build verify batch: [id_last, c1, ..., cBS] at positions [P, P+BS]
        const int verify_bs = (int) draft.size() + 1;
        llama_batch batch = llama_batch_init(verify_bs, 0, 1);
        const llama_pos P = (llama_pos) prompt_tgt.size();
        common_batch_add(batch, id_last, P, { 0 }, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch, draft[i], P + 1 + (llama_pos) i, { 0 }, true);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "error: verify decode failed\n");
            llama_batch_free(batch);
            break;
        }

        // Per-position target argmax over the verify batch.
        // sampled_at[k] = token target picks at verify position k.
        std::vector<llama_token> sampled_at(verify_bs);
        for (int k = 0; k < verify_bs; ++k) {
            // common_sampler_sample expects logits at index k of the batch.
            sampled_at[k] = common_sampler_sample(smpl, ctx, k);
        }
        llama_batch_free(batch);

        // Accept-prefix: longest run where draft[k] == sampled_at[k].
        // sampled_at[0] = the token target picks AFTER id_last (anchor row),
        // so it always becomes emitted regardless of drafter. Drafter
        // produced draft[0..BS-1]; check draft[k] == sampled_at[k+1] for
        // k in [0, draft.size()).
        int n_accepted = 0;
        for (size_t k = 0; k < draft.size(); ++k) {
            if (draft[k] == sampled_at[k + 1]) {
                n_accepted += 1;
            } else {
                break;
            }
        }
        common_speculative_accept(spec, (uint16_t) n_accepted);
        n_accept_tokens += n_accepted;

        // Emit: sampled_at[0] (anchor's next, deterministic from id_last)
        //   ... then accepted draft tokens [0..n_accepted)
        //   ... plus the bonus = sampled_at[n_accepted+1] (the first
        //       diverging or extra-row token)
        // Actually the standard accept-emit pattern emits the accepted
        // run + 1 bonus. Here sampled_at[0] is the token at anchor's row,
        // which IS what we want to emit first (the deterministic
        // "anchor's continuation"). Then bonus = sampled_at[n_accepted+1]
        // (target's argmax at the first unaccepted position, or the
        // post-block bonus row).
        // For acceptance counting per the literature we count
        // n_accepted from the drafter. Emit count = 1 + n_accepted.

        for (int k = 0; k < n_accepted; ++k) {
            common_sampler_accept(smpl, ctx, draft[k], true);
            printf("%s", common_token_to_piece(ctx, draft[k]).c_str());
            emitted.push_back(draft[k]);
        }
        // Bonus.
        const llama_token bonus = sampled_at[n_accepted + 1];
        common_sampler_accept(smpl, ctx, bonus, true);
        printf("%s", common_token_to_piece(ctx, bonus).c_str());
        fflush(stdout);
        emitted.push_back(bonus);

        // Remove rejected positions from target KV cache. We decoded
        // [P, P+BS] but only positions [P, P+n_accepted+1] are kept.
        if (n_accepted < (int) draft.size()) {
            llama_kv_cache_seq_rm(ctx, 0, P + 1 + n_accepted + 1, -1);
        }

        id_last = bonus;
        if (llama_token_is_eog(model, bonus)) break;
    }

    int64_t t1 = now_us();
    const double dt = (t1 - t0) * 1e-6;
    const int n_emitted = (int) emitted.size() - 1;  // -1 because emitted[0] = first sampled
    const double tps = n_emitted / dt;
    const double accept_rate = n_drafts > 0
        ? (double) n_accept_tokens / (double) n_drafts
        : 0.0;

    printf("\n\n");
    fprintf(stderr, "\n=== dflash-speculative-simple summary ===\n");
    fprintf(stderr, "  emitted tokens : %d\n", n_emitted);
    fprintf(stderr, "  draft cycles   : %d\n", n_drafts);
    fprintf(stderr, "  draft tokens   : %d\n", n_draft_tokens);
    fprintf(stderr, "  accepted total : %d\n", n_accept_tokens);
    fprintf(stderr, "  mean accept    : %.3f tokens/draft\n", accept_rate);
    fprintf(stderr, "  wall (s)       : %.3f\n", dt);
    fprintf(stderr, "  tok/s          : %.2f\n", tps);

    common_sampler_free(smpl);
    common_speculative_free(spec);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
