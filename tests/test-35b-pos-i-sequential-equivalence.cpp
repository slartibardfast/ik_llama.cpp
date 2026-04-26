// Position-i sequential equivalence probe.
//
// Asserts: for a batch of N tokens [T0, T1, .., T_{N-1}] decoded at positions
// [P, P+1, .., P+N-1], the argmax of logits_ith(i) must equal the argmax of
// a single-token batch=1 decode of T_i at position P+i after a sequential
// chain of batch=1 decodes of T0..T_{i-1}.
//
// The `test-35b-batch-invariance-sweep` only checks position 0. This one
// checks EVERY position of the batch — which is what the speculator reads
// on a full-accept cycle (pos-1 sample becomes the next slot.sampled).
//
// FAIL at pos > 0 means: batch=N produces different pos-i logits than
// sequential batch=1 through the same tokens. This is the "batch-end state
// propagates wrong" drift source that affects full-accept speculation on
// hybrid/recurrent models.
//
// Usage:
//   test-35b-pos-i-sequential-equivalence -m <model.gguf> [--n-pairs 8]
//                                         [--prompt "..."] [--batch-size N]

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static llama_token argmax(const float * logits, int n_vocab) {
    llama_token best = 0;
    float best_l = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_l) { best_l = logits[i]; best = i; }
    }
    return best;
}

int main(int argc, char ** argv) {
    gpt_params params;
    params.prompt = "The capital of France is";

    int n_pairs    = 8;
    int batch_size = 4;
    std::vector<char *> forward;
    forward.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n-pairs") == 0 && i + 1 < argc) {
            n_pairs = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            batch_size = std::atoi(argv[++i]);
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
    llama_model   * model  = init.model;
    llama_context * ctx    = init.context;
    if (!model || !ctx) { fprintf(stderr, "Load failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token> prompt = ::common_tokenize(ctx, params.prompt, true);
    const int n_prompt = (int) prompt.size();
    fprintf(stderr, "Prompt: '%s' (%d tokens), n_pairs=%d, batch_size=%d\n",
            params.prompt.c_str(), n_prompt, n_pairs, batch_size);

    // Prompt eval.
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        common_batch_add(batch, prompt[i], i, {0}, i == n_prompt - 1);
    }
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Prompt decode failed\n"); return 1; }

    // Snapshot post-prompt state.
    const size_t snap_size = llama_state_seq_get_size(ctx, 0, 0);
    std::vector<uint8_t> snapshot(snap_size);
    if (llama_state_seq_get_data(ctx, snapshot.data(), snap_size, 0, 0) == 0) {
        fprintf(stderr, "Snapshot failed\n"); return 1;
    }

    auto restore = [&]() {
        llama_kv_cache_clear(ctx);
        llama_state_seq_set_data(ctx, snapshot.data(), snap_size, 0, 0);
    };

    std::mt19937 rng(0xBADC0DE);

    int mismatches = 0;
    printf("\n%-5s %-10s %-24s %-24s %s\n",
           "pair", "pos", "batch-N argmax", "sequential argmax", "match");

    for (int p = 0; p < n_pairs; p++) {
        // Generate a random K-token sequence, avoiding EOG.
        std::vector<llama_token> toks;
        toks.reserve(batch_size);
        for (int k = 0; k < batch_size; k++) {
            llama_token t;
            do {
                t = (llama_token)(rng() % n_vocab);
            } while (llama_vocab_is_eog(llama_model_get_vocab(model), t));
            toks.push_back(t);
        }

        // --- Path A: one batch-N decode. Collect argmax at each position. ---
        restore();
        common_batch_clear(batch);
        for (int i = 0; i < batch_size; i++) {
            common_batch_add(batch, toks[i], n_prompt + i, {0}, true);
        }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "pair %d batch-N decode failed\n", p); return 1; }
        std::vector<llama_token> argA(batch_size);
        for (int i = 0; i < batch_size; i++) {
            argA[i] = argmax(llama_get_logits_ith(ctx, i), n_vocab);
        }

        // --- Path B: sequential batch=1 decodes. ---
        restore();
        std::vector<llama_token> argB(batch_size);
        for (int i = 0; i < batch_size; i++) {
            common_batch_clear(batch);
            common_batch_add(batch, toks[i], n_prompt + i, {0}, true);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "pair %d seq step %d decode failed\n", p, i);
                return 1;
            }
            argB[i] = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
        }

        // --- Compare at every position. ---
        for (int i = 0; i < batch_size; i++) {
            const bool same = (argA[i] == argB[i]);
            if (!same) mismatches++;
            printf("%-5d %-10d %-24d %-24d %s\n",
                   p, i, argA[i], argB[i], same ? "SAME" : "DIFF");
        }
    }

    printf("\nmismatches = %d / %d (pairs=%d, batch_size=%d)\n",
           mismatches, n_pairs * batch_size, n_pairs, batch_size);
    printf("%s\n", mismatches == 0 ? "PASS" : "FAIL");

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return mismatches == 0 ? 0 : 1;
}
