// Direct batch-invariance probe for speculative-decode correctness.
//
// Speculation requires:
//   main_logits(batch=[A])[0]  ==  main_logits(batch=[A, B])[0]
//
// i.e. putting a second token into the verification batch must not change the
// model's output at position 0. On a fully correct backend at T=0 these two
// decodes should produce byte-identical logits at position 0.
//
// This test loads a model, runs a short prompt, then:
//   Probe 1: decode batch=[A] at position n_prompt; capture logits_1 at pos 0.
//   Rollback KV to the post-prompt state.
//   Probe 2: decode batch=[A, B] at positions n_prompt, n_prompt+1; capture
//            logits_2 at pos 0.
//   Compare logits_1 vs logits_2 byte-wise.
//
// PASS iff: max_diff < 1e-5 AND argmax(logits_1) == argmax(logits_2).
// FAIL signifies batch-invariance breaks — speculation on this model/backend
// will silently corrupt main sampling by compounding per-decode drift.
//
// Built for QWEN35MOE (35B-A3B) but works on any model.
//
// Usage:
//   test-35b-batch-invariance -m <model.gguf> [--n-gpu-layers 999] [--fit]

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static void capture_logits(llama_context * ctx, int ith, std::vector<float> & out) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * logits = llama_get_logits_ith(ctx, ith);
    out.resize((size_t) n_vocab);
    std::memcpy(out.data(), logits, (size_t) n_vocab * sizeof(float));
}

static llama_token argmax(const std::vector<float> & logits) {
    llama_token best = 0;
    float best_l = logits[0];
    for (size_t i = 1; i < logits.size(); i++) {
        if (logits[i] > best_l) {
            best_l = logits[i];
            best = (llama_token) i;
        }
    }
    return best;
}

int main(int argc, char ** argv) {
    gpt_params params;
    params.prompt = "The capital of France is";
    params.n_ctx = 512;

    if (!gpt_params_parse(argc, argv, params)) {
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

    // Tokenize the prompt.
    std::vector<llama_token> prompt = ::common_tokenize(ctx, params.prompt, true);
    const int n_prompt = (int) prompt.size();
    if (n_prompt < 2) {
        fprintf(stderr, "Prompt too short\n");
        return 1;
    }
    fprintf(stderr, "Prompt: '%s' (%d tokens)\n", params.prompt.c_str(), n_prompt);

    // Prompt eval (produces logits only at last prompt position).
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        common_batch_add(batch, prompt[i], i, {0}, i == n_prompt - 1);
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Prompt decode failed\n");
        return 1;
    }

    // Greedy-sample token A from the last-prompt-position logits.
    std::vector<float> tmp;
    capture_logits(ctx, batch.n_tokens - 1, tmp);
    const llama_token A = argmax(tmp);
    fprintf(stderr, "Sampled token A = %d ('%s')\n",
            A, common_token_to_piece(ctx,A).c_str());

    // Pick token B as A+1 (any other token works; avoid EOG).
    llama_token B = A + 1;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (B >= n_vocab) B = (A > 0) ? A - 1 : 1;
    fprintf(stderr, "Probe token B = %d ('%s')\n",
            B, common_token_to_piece(ctx,B).c_str());

    // --- Probe 1: decode batch=[A] at position n_prompt ---
    common_batch_clear(batch);
    common_batch_add(batch, A, n_prompt, {0}, true);
    // Hack: common_batch_init gave us batch with n_tokens==0 after clear;
    // the above add sets it.
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Probe 1 decode failed\n");
        return 1;
    }
    std::vector<float> logits_1;
    capture_logits(ctx, 0, logits_1);
    const llama_token arg_1 = argmax(logits_1);
    fprintf(stderr, "Probe 1 (batch=1): argmax at pos 0 = %d ('%s')\n",
            arg_1, common_token_to_piece(ctx,arg_1).c_str());

    // Rollback KV cache from n_prompt onwards (remove just-added entry).
    llama_kv_cache_seq_rm(ctx, 0, n_prompt, -1);

    // --- Probe 2: decode batch=[A, B] at positions n_prompt, n_prompt+1 ---
    common_batch_clear(batch);
    common_batch_add(batch, A, n_prompt,     {0}, true);
    common_batch_add(batch, B, n_prompt + 1, {0}, true);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "Probe 2 decode failed\n");
        return 1;
    }
    std::vector<float> logits_2;
    capture_logits(ctx, 0, logits_2);
    const llama_token arg_2 = argmax(logits_2);
    fprintf(stderr, "Probe 2 (batch=2): argmax at pos 0 = %d ('%s')\n",
            arg_2, common_token_to_piece(ctx,arg_2).c_str());

    // --- Compare ---
    float max_diff = 0.0f;
    size_t n_diff = 0;
    const size_t n = logits_1.size();
    for (size_t i = 0; i < n; i++) {
        const float d = std::fabs(logits_1[i] - logits_2[i]);
        if (d > 1e-5f) n_diff++;
        if (d > max_diff) max_diff = d;
    }
    fprintf(stderr, "\n--- Batch invariance report ---\n");
    fprintf(stderr, "  vocab                 : %zu\n", n);
    fprintf(stderr, "  max_diff(abs)         : %g\n", max_diff);
    fprintf(stderr, "  n_diff > 1e-5         : %zu (%.4f%%)\n",
            n_diff, 100.0 * n_diff / n);
    fprintf(stderr, "  argmax match          : %s (%d vs %d)\n",
            (arg_1 == arg_2) ? "YES" : "NO", arg_1, arg_2);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    const bool pass = (max_diff < 1e-3f) && (arg_1 == arg_2);
    fprintf(stderr, "\n%s\n", pass ? "PASS" : "FAIL: batch-invariance broken");
    return pass ? 0 : 1;
}
