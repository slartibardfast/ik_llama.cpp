// Rollback sweep correctness.
//
// Extends trajectory-drift's token_idx=0 test to sweep ALL valid token_idx
// values in a batch of N. Asserts: rolling back to token_idx=k after a
// batch-N decode of [T0..T_{N-1}] gives a state byte-equivalent to running
// batch=1 decodes of T0..Tk sequentially.
//
// Method: after the rollback, issue a single-token decode of some probe
// token P and record argmax(logits[0]). Compare to the same probe issued
// after the sequential batch=1 chain. Iterating k from 0..N-1 covers the
// whole intermediate rollback range.
//
// Any mismatch at specific k isolates which intermediate the rollback
// reads/writes wrong (e.g. SSM state slot k, or conv state sliding window
// reconstruction at column k+1..d_conv-1).
//
// Usage:
//   test-35b-rollback-sweep-correctness -m <model.gguf> [--batch-size 4] [-mtp]

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

    int batch_size = 4;
    std::vector<char *> forward;
    forward.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
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
    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> prompt = ::common_tokenize(ctx, params.prompt, true);
    const int n_prompt = (int) prompt.size();
    fprintf(stderr, "Prompt: '%s' (%d tokens), batch_size=%d\n",
            params.prompt.c_str(), n_prompt, batch_size);

    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        common_batch_add(batch, prompt[i], i, {0}, i == n_prompt - 1);
    }
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Prompt decode failed\n"); return 1; }

    const size_t snap_size = llama_state_seq_get_size(ctx, 0, 0);
    std::vector<uint8_t> snapshot(snap_size);
    if (llama_state_seq_get_data(ctx, snapshot.data(), snap_size, 0, 0) == 0) {
        fprintf(stderr, "Snapshot failed\n"); return 1;
    }
    auto restore = [&]() {
        llama_kv_cache_clear(ctx);
        llama_state_seq_set_data(ctx, snapshot.data(), snap_size, 0, 0);
    };

    // Deterministic token selection.
    std::mt19937 rng(0xD7A17E57);
    std::vector<llama_token> toks;
    toks.reserve(batch_size);
    for (int k = 0; k < batch_size; k++) {
        llama_token t;
        do {
            t = (llama_token)(rng() % n_vocab);
        } while (llama_vocab_is_eog(vocab, t));
        toks.push_back(t);
    }
    // Probe token: a different fixed id.
    llama_token probe = (llama_token)(n_vocab / 3);
    if (llama_vocab_is_eog(vocab, probe)) probe++;

    // --- Reference argmax via sequential batch=1 ---
    // For each k in 0..batch_size-1, record argmax(logits[0]) after
    // sequential decode of T0..Tk, then batch=1 [probe] at pos (n_prompt+k+1).
    std::vector<llama_token> ref(batch_size);
    for (int k = 0; k < batch_size; k++) {
        restore();
        for (int i = 0; i <= k; i++) {
            common_batch_clear(batch);
            common_batch_add(batch, toks[i], n_prompt + i, {0}, true);
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "ref step %d decode failed\n", i); return 1; }
        }
        // Probe.
        common_batch_clear(batch);
        common_batch_add(batch, probe, n_prompt + k + 1, {0}, true);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "ref probe at k=%d decode failed\n", k); return 1; }
        ref[k] = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
    }

    // --- Batch-N decode once, then for each k rollback to token_idx=k and probe ---
    int mismatches = 0;
    printf("\n%-5s %-18s %-18s %s\n", "k", "seq (batch=1)", "batch=N + rollback", "match");

    for (int k = 0; k < batch_size; k++) {
        restore();
        // One batch-N decode.
        common_batch_clear(batch);
        for (int i = 0; i < batch_size; i++) {
            common_batch_add(batch, toks[i], n_prompt + i, {0}, true);
        }
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "batch-N decode failed for k=%d\n", k); return 1; }

        // Rollback to token_idx=k. target_pos = n_prompt + k (last cache pos after rollback).
        const bool ok = llama_rollback_delta_net_state(
                ctx, /*token_idx=*/k, /*seq_id=*/0,
                /*target_pos=*/n_prompt + k);
        if (!ok) {
            fprintf(stderr, "rollback returned false at k=%d\n", k);
            printf("%-5d %-18s %-18s %s\n", k, "-", "rollback-FAIL", "DIFF");
            mismatches++;
            continue;
        }
        // Trim cache past n_prompt+k+1 (positions 0..n_prompt+k kept; token at
        // pos n_prompt+k+1 onward is the "rejected" tail).
        llama_kv_cache_seq_rm(ctx, 0, n_prompt + k + 1, -1);

        // Probe at position n_prompt+k+1 (single token batch=1).
        common_batch_clear(batch);
        common_batch_add(batch, probe, n_prompt + k + 1, {0}, true);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "probe at k=%d decode failed\n", k); return 1; }
        const llama_token got = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
        const bool same = (got == ref[k]);
        if (!same) mismatches++;
        printf("%-5d %-18d %-18d %s\n", k, ref[k], got, same ? "SAME" : "DIFF");
    }

    printf("\nmismatches = %d / %d\n", mismatches, batch_size);
    printf("%s\n", mismatches == 0 ? "PASS" : "FAIL");

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return mismatches == 0 ? 0 : 1;
}
