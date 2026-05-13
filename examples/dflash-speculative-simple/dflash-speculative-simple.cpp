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

        // T6.A: snapshot DeltaNet state BEFORE the verify decode mutates
        // it. We'll restore from this checkpoint after the partial-accept
        // decision so the re-decode of the committed prefix sees the
        // correct recurrent state at every position.
        const llama_pos P = (llama_pos) prompt_tgt.size();
        if (llama_dflash_state_snapshot(ctx, /*slot*/0) != LLAMA_DFLASH_OK) {
            fprintf(stderr, "error: state_snapshot failed\n");
            break;
        }

        // Build verify batch: [id_last, c1, ..., cBS] at positions [P, P+BS]
        const int verify_bs = (int) draft.size() + 1;
        llama_batch batch = llama_batch_init(verify_bs, 0, 1);
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
        std::vector<llama_token> sampled_at(verify_bs);
        for (int k = 0; k < verify_bs; ++k) {
            sampled_at[k] = common_sampler_sample(smpl, ctx, k);
        }
        llama_batch_free(batch);

        // Accept-prefix: longest run where draft[k] == sampled_at[k].
        int n_accepted = 0;
        for (size_t k = 0; k < draft.size(); ++k) {
            if (draft[k] == sampled_at[k]) {
                n_accepted += 1;
            } else {
                break;
            }
        }
        common_speculative_accept(spec, (uint16_t) n_accepted);
        n_accept_tokens += n_accepted;

        // ── T6.B: bonus re-decode for late-stream coherence ──
        // The verify batch decoded positions [P, P+BS] with inputs
        // [id_last, c1, ..., cBS]. The slot at the bonus position
        // (P + n_accepted + 1) has KV + hidden from input c_{n_accepted+1},
        // which was REJECTED. To get the correct state for the next cycle:
        //   1. seq_rm to position P  (remove ALL of this cycle's KV)
        //   2. restore DeltaNet state to the pre-verify checkpoint
        //   3. trim the cb_eval extract buffer to the pre-verify length
        //   4. re-decode the committed sequence [id_last, accepted_drafts..., bonus]
        const llama_token bonus = sampled_at[n_accepted];

        llama_kv_cache_seq_rm(ctx, 0, P, -1);
        if (llama_dflash_state_restore(ctx, /*slot*/0) != LLAMA_DFLASH_OK) {
            fprintf(stderr, "error: state_restore failed\n");
            break;
        }
        // T6.C: trim the extract buffer to the pre-verify length.
        llama_dflash_trim_extract(ctx, (int32_t) P, -1);

        // Build re-decode batch: [id_last, c1, ..., c_{n_accepted}, bonus]
        // at positions [P, P+1, ..., P+n_accepted+1] (n_accepted+2 tokens).
        const int re_n = n_accepted + 2;
        llama_batch re_batch = llama_batch_init(re_n, 0, 1);
        common_batch_add(re_batch, id_last, P, { 0 }, true);
        for (int k = 0; k < n_accepted; ++k) {
            common_batch_add(re_batch, draft[k], P + 1 + (llama_pos) k, { 0 }, true);
        }
        common_batch_add(re_batch, bonus, P + 1 + (llama_pos) n_accepted, { 0 }, true);
        if (llama_decode(ctx, re_batch) != 0) {
            fprintf(stderr, "error: bonus re-decode failed\n");
            llama_batch_free(re_batch);
            break;
        }
        llama_batch_free(re_batch);

        // Emit the n_accepted draft tokens + bonus.
        for (int k = 0; k < n_accepted; ++k) {
            common_sampler_accept(smpl, ctx, draft[k], true);
            printf("%s", common_token_to_piece(ctx, draft[k]).c_str());
            emitted.push_back(draft[k]);
        }
        common_sampler_accept(smpl, ctx, bonus, true);
        printf("%s", common_token_to_piece(ctx, bonus).c_str());
        fflush(stdout);
        emitted.push_back(bonus);

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
