// Sweeps batch-invariance across N (A, B) probe pairs.
//
// For each pair, measures whether:
//     argmax(main_logits(batch=[A]))[0]  ==  argmax(main_logits(batch=[A, B]))[0]
//
// Speculation at T=0 silently corrupts output whenever this fails: the verify
// decode at batch=2 commits a position-0 KV state that differs from what a
// non-speculating batch=1 decode would have written. Over many decodes these
// per-position drifts compound and the greedy trajectory diverges from the
// no-speculation reference.
//
// PASS iff stability rate == 100% on the probed pairs.
// Reports per-pair argmax (batch=1 vs batch=2) and aggregate drift rate.
//
// Usage:
//     test-35b-batch-invariance-sweep -m <model.gguf> [--n-pairs 5] [-c 128 -b 16 -ub 16]

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static llama_token argmax(const float * logits, int n_vocab) {
    llama_token best = 0;
    float best_l = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_l) {
            best_l = logits[i];
            best = i;
        }
    }
    return best;
}

// Decode `batch_tokens` at positions [start_pos, start_pos + len) and return
// argmax at position 0.
static llama_token decode_and_argmax_pos0(
        llama_context * ctx,
        llama_batch   & batch,
        const std::vector<llama_token> & tokens,
        int             start_pos,
        int             n_vocab) {
    common_batch_clear(batch);
    for (size_t i = 0; i < tokens.size(); i++) {
        common_batch_add(batch, tokens[i], start_pos + (int) i, {0}, true);
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return -1;
    }
    const float * logits = llama_get_logits_ith(ctx, 0);
    return argmax(logits, n_vocab);
}

int main(int argc, char ** argv) {
    gpt_params params;
    params.prompt  = "The capital of France is";

    // Custom --n-pairs argument — strip before handing to gpt_params_parse.
    int n_pairs = 5;
    std::vector<char *> forward;
    forward.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n-pairs") == 0 && i + 1 < argc) {
            n_pairs = std::atoi(argv[++i]);
        } else {
            forward.push_back(argv[i]);
        }
    }

    if (!gpt_params_parse((int) forward.size(), forward.data(), params)) {
        fprintf(stderr, "Failed to parse params\n");
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_init_result init = llama_init_from_gpt_params(params);
    llama_model   * model = init.model;
    llama_context * ctx   = init.context;
    if (!model || !ctx) {
        fprintf(stderr, "Failed to load model/context\n");
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token> prompt = ::common_tokenize(ctx, params.prompt, true);
    const int n_prompt = (int) prompt.size();
    fprintf(stderr, "Prompt: '%s' (%d tokens), n_pairs=%d, vocab=%d\n\n",
            params.prompt.c_str(), n_prompt, n_pairs, n_vocab);

    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        common_batch_add(batch, prompt[i], i, {0}, i == n_prompt - 1);
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Prompt decode failed\n");
        return 1;
    }

    // Greedy-sample A from last prompt position — this is the first "real"
    // decode position in a chat-like flow.
    const llama_token A = argmax(
            llama_get_logits_ith(ctx, batch.n_tokens - 1), n_vocab);
    fprintf(stderr, "Sampled A = %d ('%s')\n", A,
            common_token_to_piece(ctx, A).c_str());

    // Snapshot post-prompt state. Needed because llama_kv_cache_seq_rm does
    // not roll back recurrent SSM state on hybrid models — between probes we
    // must restore this snapshot, not just clear KV positions.
    const size_t snap_size = llama_state_seq_get_size(ctx, 0, 0);
    std::vector<uint8_t> snapshot(snap_size);
    if (llama_state_seq_get_data(ctx, snapshot.data(), snap_size, 0, 0) == 0) {
        fprintf(stderr, "Failed to snapshot post-prompt state\n");
        return 1;
    }
    fprintf(stderr, "Post-prompt snapshot: %zu bytes\n", snap_size);

    auto restore_snapshot = [&]() {
        llama_kv_cache_clear(ctx);
        llama_state_seq_set_data(ctx, snapshot.data(), snap_size, 0, 0);
    };

    // Sweep: for each probe offset k in [1..n_pairs], pick B = (A + k * 3079) %
    // vocab. 3079 is a prime, chosen to spread B across different expert-
    // routing bins without landing on trivially-close tokens.
    int mismatches = 0;
    printf("\n%-4s %-8s %-30s %-8s %-30s %-10s\n",
            "k", "B", "B-piece", "argA", "argA-piece (batch=1)", "argAB (b=2)");
    for (int k = 1; k <= n_pairs; k++) {
        const llama_token B = (llama_token) (((long long) A + (long long) k * 3079) % n_vocab);
        if (llama_vocab_is_eog(llama_model_get_vocab(model), B)) {
            // EOG — skip by picking a different token
            continue;
        }

        // Probe batch=1: decode [A] at position n_prompt.
        restore_snapshot();
        llama_token arg1 = decode_and_argmax_pos0(
                ctx, batch, {A}, n_prompt, n_vocab);

        // Probe batch=2: decode [A, B] at positions n_prompt, n_prompt+1.
        restore_snapshot();
        llama_token arg2 = decode_and_argmax_pos0(
                ctx, batch, {A, B}, n_prompt, n_vocab);

        const bool match = (arg1 == arg2);
        if (!match) mismatches++;

        printf("%-4d %-8d %-30s %-8d %-30s %s\n",
                k, B,
                common_token_to_piece(ctx, B).c_str(),
                arg1,
                common_token_to_piece(ctx, arg1).c_str(),
                match ? "SAME" : "DIFF");
    }

    printf("\nmismatches = %d / %d\n", mismatches, n_pairs);
    const bool pass = (mismatches == 0);
    printf("%s\n", pass ? "PASS" : "FAIL: argmax batch-variant");

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return pass ? 0 : 1;
}
