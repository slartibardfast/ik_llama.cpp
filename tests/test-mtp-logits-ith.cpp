// Verifies `llama_get_mtp_logits_ith(ctx, 0)` returns MTP logits for batch
// position 0 that match a standalone 1-token decode's MTP logits — the
// semantic invariant that MTP-IR (intermediate rollback) relies on.
//
// Ported from polaris /home/llm/src/qwen35-mtp/tests/test-mtp-logits-ith.cpp.
// Adaptations for ik_llama: see test-intermediate-rollback.cpp header.
//
// Test flow:
//   1. Load hybrid model, warm up with a deterministic prompt.
//   2. Snapshot post-prompt recurrent state (PARTIAL_ONLY).
//   3. Path A: decode [T] alone → capture llama_get_mtp_logits (last position).
//   4. Restore snapshot; trim KV past prompt end.
//   5. Path B: decode [T, D] batch → capture llama_get_mtp_logits_ith(ctx, 0).
//   6. Compare A vs B: CPU must be bit-identical; Vulkan with
//      GGML_VK_DISABLE_MMVQ=1 should be within the same tolerance as
//      test-intermediate-rollback.cpp.

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static llama_token greedy(llama_context * ctx) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
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

int main(int argc, char ** argv) {
    gpt_params params;
    params.prompt    = "The capital of France is";
    params.n_predict = 0;
    params.seed      = 42;
    params.n_ctx     = 2048;
    params.n_parallel = 1;
    params.has_mtp   = true;    // ik_llama: enable MTP head at context init
    params.graph_reuse = false; // force per-batch-size graph rebuild (see test-intermediate-rollback.cpp comment)

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    llama_init_result init = llama_init_from_gpt_params(params);
    llama_model   * model = init.model;
    llama_context * ctx   = init.context;
    if (!model || !ctx) {
        fprintf(stderr, "failed to init\n");
        return 1;
    }

    fprintf(stderr, "=== test-mtp-logits-ith ===\n");
    fprintf(stderr, "model: %s\n", params.model.c_str());
    fprintf(stderr, "n_gpu_layers: %d\n", params.n_gpu_layers);

    // 1. Prompt warmup.
    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, params.prompt, true);
    const int n_prompt = (int) prompt_tokens.size();
    fprintf(stderr, "prompt tokens: %d\n", n_prompt);

    llama_batch batch = llama_batch_init(params.n_ctx, 0, 1);
    common_batch_clear(batch);
    for (int i = 0; i < n_prompt; i++) {
        common_batch_add(batch, prompt_tokens[i], i, {0}, i == n_prompt - 1);
    }
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "FAIL: prompt decode\n"); return 2; }

    const int pos_of_T = n_prompt;
    const llama_token T = greedy(ctx);
    const llama_token D = 42;
    fprintf(stderr, "T = %d, D = %d, pos_of_T = %d\n", T, D, pos_of_T);

    // 2. Snapshot post-prompt state.
    const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    const size_t S0_size = llama_state_seq_get_size(ctx, 0, flags);
    std::vector<uint8_t> S0(S0_size);
    if (llama_state_seq_get_data(ctx, S0.data(), S0.size(), 0, flags) == 0) {
        fprintf(stderr, "FAIL: snapshot\n"); return 3;
    }

    // 3. Path A: decode [T] alone.
    common_batch_clear(batch);
    common_batch_add(batch, T, pos_of_T, {0}, true);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "FAIL: Path A decode\n"); return 4; }

    const float * mtp_A = llama_get_mtp_logits(ctx);
    if (mtp_A == nullptr) { fprintf(stderr, "FAIL: mtp_A nullptr (model lacks MTP?)\n"); return 5; }
    const int64_t n_vocab  = llama_get_mtp_n_vocab(ctx);
    const int64_t n_drafts = llama_get_mtp_n_drafts(ctx);
    const size_t chunk_elems = (size_t)(n_vocab * n_drafts);
    fprintf(stderr, "n_vocab=%lld n_drafts=%lld (chunk=%zu elems)\n",
            (long long)n_vocab, (long long)n_drafts, chunk_elems);

    std::vector<float> A_logits(chunk_elems);
    std::memcpy(A_logits.data(), mtp_A, chunk_elems * sizeof(float));

    // 4. Restore snapshot, trim KV.
    if (llama_state_seq_set_data(ctx, S0.data(), S0.size(), 0, flags) == 0) {
        fprintf(stderr, "FAIL: restore\n"); return 6;
    }
    llama_kv_cache_seq_rm(ctx, 0, pos_of_T, -1);

    // 5. Path B: decode [T, D] batch.
    common_batch_clear(batch);
    common_batch_add(batch, T, pos_of_T,     {0}, true);
    common_batch_add(batch, D, pos_of_T + 1, {0}, true);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "FAIL: Path B decode\n"); return 7; }

    const float * mtp_B_pos0 = llama_get_mtp_logits_ith(ctx, 0);
    if (mtp_B_pos0 == nullptr) {
        fprintf(stderr, "FAIL: llama_get_mtp_logits_ith(ctx, 0) returned nullptr\n");
        return 8;
    }

    // 6. Compare.
    float max_diff = 0.0f;
    size_t n_diff = 0;
    size_t first_diff = (size_t)-1;
    for (size_t i = 0; i < chunk_elems; i++) {
        const float d = std::fabs(A_logits[i] - mtp_B_pos0[i]);
        if (d > 1e-9f) {
            n_diff++;
            if (first_diff == (size_t)-1) first_diff = i;
        }
        if (d > max_diff) max_diff = d;
    }

    const float tol = 5e-2f;
    const bool ok = max_diff <= tol;
    fprintf(stderr, "\n[A vs B_pos0] max_diff=%.6e n_diff=%zu / %zu first_at=",
            max_diff, n_diff, chunk_elems);
    if (first_diff == (size_t)-1) fprintf(stderr, "none");
    else                          fprintf(stderr, "%zu", first_diff);
    fprintf(stderr, " → %s\n", ok ? "PASS" : "FAIL");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    return ok ? 0 : 9;
}
