// test-mtp-fused-chain-residual.cpp
//
// Drives:
//   - mtp_fused_draft.allium ChainResidualParity
//   - Phase 36 direction-#3 (per-step semantic equivalence in fused chain)
//
// Step k of the fused chain must consume the SAME residual the
// per-step path consumes from llama_get_embeddings_ith(ctx, 0) after
// step k-1's DRAFT_GEN decode. Both paths run greedy argmax over the
// same lm_head; given identical seed_hidden, identical seed_token,
// and identical pre-KV state, they MUST produce identical N-token
// sequences.
//
// The 2026-05-07 measurement (post seed-source fix) showed fused d=3
// at 52% accept vs per-step's 58% — the gap localised to step 1+ of
// the chain (step 0 matched per-step within float-cast precision).
// This test pins the parity contract that direction #3 must enforce.
//
// Test fixture:
//   - LLAMACPP_TEST_MODELFILE / argv[1]: a Qwen 3.5-class GGUF with
//     nextn_predict_layers (qwen35 or qwen35moe). The Phase 36
//     production GGUF
//     /opt/models/recast-out/qwen3.6-27b-V-F1.T1.qq-tool1lossless-vocab-fix.gguf
//     is the canonical fixture.
//   - Prompt: a fixed 5-token English string (deterministic at temp=0).
//   - Chain depth: 3.

#include "llama.h"
#include "../get-model.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

static std::vector<llama_token> tokenize_prompt(
        const llama_model * model,
        const char *        prompt) {
    int n_max = (int) std::strlen(prompt) + 8;
    std::vector<llama_token> tokens(n_max);
    int n = llama_tokenize(model, prompt, (int) std::strlen(prompt),
                           tokens.data(), n_max,
                           /*add_special=*/true,
                           /*parse_special=*/false);
    if (n < 0) {
        n_max = -n;
        tokens.resize(n_max);
        n = llama_tokenize(model, prompt, (int) std::strlen(prompt),
                           tokens.data(), n_max, true, false);
    }
    tokens.resize(n);
    return tokens;
}

static int run_chain_perstep(
        llama_context *           ctx,
        const std::vector<float> & seed_hidden,
        llama_token                seed_token,
        int                        n_past,
        int                        n_steps,
        std::vector<llama_token> & out_tokens) {
    out_tokens.clear();
    out_tokens.reserve(n_steps);

    llama_set_draft_input_hidden_state(ctx, seed_hidden.data());

    llama_token current_input = seed_token;
    int         current_pos   = n_past;

    llama_batch batch = llama_batch_init(1, 0, 1);

    for (int i = 0; i < n_steps; ++i) {
        llama_set_mtp_op_type(ctx, LLAMA_MTP_OP_DRAFT_GEN);
        batch.n_tokens     = 1;
        batch.token[0]     = current_input;
        batch.pos[0]       = current_pos;
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = true;
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[chain-residual] per-step decode failed at step=%d\n", i);
            llama_batch_free(batch);
            llama_set_mtp_op_type(ctx, LLAMA_MTP_OP_NONE);
            return -1;
        }
        int32_t arg_id  = -1;
        float   arg_prob = 0.0f;
        if (!llama_get_draft_argmax(ctx, 0, &arg_id, &arg_prob)) {
            fprintf(stderr, "[chain-residual] per-step argmax cache empty at step=%d\n", i);
            llama_batch_free(batch);
            llama_set_mtp_op_type(ctx, LLAMA_MTP_OP_NONE);
            return -1;
        }
        out_tokens.push_back(arg_id);

        // Pull next chain step's hidden state (result_norm) from this DRAFT_GEN's embd.
        const float * emb = llama_get_embeddings_ith(ctx, 0);
        if (emb && i + 1 < n_steps) {
            llama_set_draft_input_hidden_state(ctx, emb);
        }

        current_input = arg_id;
        current_pos++;
    }

    llama_batch_free(batch);
    llama_set_mtp_op_type(ctx, LLAMA_MTP_OP_NONE);
    return 0;
}

int main(int argc, char *argv[]) {
    char * model_path = get_model_or_exit(argc, argv);

    fprintf(stderr, "[chain-residual] model=%s\n", model_path);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.use_mmap = false;
    mparams.n_gpu_layers = 999;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "[chain-residual] failed to load model\n");
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 4096;
    cparams.n_batch         = 2048;
    cparams.n_ubatch        = 512;
    cparams.flash_attn      = true;
    cparams.mtp             = true;
    cparams.mtp_op_type     = LLAMA_MTP_OP_NONE;
    cparams.type_k          = GGML_TYPE_Q4_0;
    cparams.type_v          = GGML_TYPE_Q4_0;
    cparams.k_cache_hadamard = true;
    cparams.v_cache_hadamard = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "[chain-residual] failed to init context\n");
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    const int n_embd = llama_model_n_embd(model);

    const char * prompt_str = "The capital of France is";
    std::vector<llama_token> tokens = tokenize_prompt(model, prompt_str);
    fprintf(stderr, "[chain-residual] prompt n_tokens=%zu\n", tokens.size());

    // Verify decode of prompt; ask for logits and embd at last position.
    llama_batch verify = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        verify.token[i]     = tokens[i];
        verify.pos[i]       = (int32_t) i;
        verify.n_seq_id[i]  = 1;
        verify.seq_id[i][0] = 0;
        verify.logits[i]    = (i + 1 == tokens.size());
    }
    verify.n_tokens = (int32_t) tokens.size();
    llama_set_mtp_op_type(ctx, LLAMA_MTP_OP_NONE);
    if (llama_decode(ctx, verify) != 0) {
        fprintf(stderr, "[chain-residual] verify decode failed\n");
        llama_batch_free(verify);
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    // Snapshot seed_hidden (h_pre_norm at last verify position).
    const float * seed_src = llama_get_embeddings_ith(ctx, -1);
    if (seed_src == nullptr) {
        fprintf(stderr, "[chain-residual] seed embeddings unavailable\n");
        llama_batch_free(verify);
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }
    std::vector<float> seed_hidden(seed_src, seed_src + n_embd);

    // Sample seed_token (greedy argmax of last verify logits).
    const float * seed_logits = llama_get_logits_ith(ctx, (int32_t) tokens.size() - 1);
    if (seed_logits == nullptr) {
        fprintf(stderr, "[chain-residual] seed logits unavailable\n");
        llama_batch_free(verify);
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }
    const int32_t n_vocab = llama_n_vocab(model);
    llama_token seed_token = 0;
    {
        float best = seed_logits[0];
        for (int32_t i = 1; i < n_vocab; ++i) {
            if (seed_logits[i] > best) { best = seed_logits[i]; seed_token = (llama_token) i; }
        }
    }
    const int n_past_after_prompt = (int) tokens.size();
    fprintf(stderr, "[chain-residual] seed_token=%d n_past=%d n_embd=%d\n",
            (int) seed_token, n_past_after_prompt, n_embd);

    llama_batch_free(verify);

    constexpr int n_steps = 3;

    // === FUSED PATH ===
    std::vector<llama_token> fused_tokens;
    {
        llama_set_draft_input_hidden_state(ctx, seed_hidden.data());
        llama_mtp_fused_result fr{};
        const int32_t rc = llama_mtp_fused_draft_invoke(
                ctx, seed_token, /*seed_hidden=*/nullptr, n_steps, &fr);
        if (rc != 0 || fr.n_steps != n_steps) {
            fprintf(stderr, "[chain-residual] fused invoke failed rc=%d n_steps=%d\n",
                    rc, fr.n_steps);
            llama_free(ctx);
            llama_free_model(model);
            llama_backend_free();
            return 1;
        }
        for (int k = 0; k < fr.n_steps; ++k) {
            fused_tokens.push_back(fr.tokens[k]);
        }
    }

    // Roll back KV state so per-step starts from identical pre-state.
    llama_kv_cache_seq_rm(ctx, /*seq_id=*/0, n_past_after_prompt, -1);

    // === PER-STEP PATH ===
    std::vector<llama_token> perstep_tokens;
    if (run_chain_perstep(ctx, seed_hidden, seed_token,
                          n_past_after_prompt, n_steps, perstep_tokens) != 0) {
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    // === COMPARE ===
    fprintf(stderr, "[chain-residual] fused   tokens: ");
    for (auto t : fused_tokens)   fprintf(stderr, "%d ", (int) t);
    fprintf(stderr, "\n[chain-residual] perstep tokens: ");
    for (auto t : perstep_tokens) fprintf(stderr, "%d ", (int) t);
    fprintf(stderr, "\n");

    bool match = (fused_tokens == perstep_tokens);
    fprintf(stderr, "[chain-residual] %s\n", match ? "PASS" : "FAIL");

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return match ? 0 : 1;
}
