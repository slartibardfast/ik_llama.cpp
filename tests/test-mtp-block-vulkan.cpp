/*
 * test-mtp-block-vulkan.cpp — standalone Vulkan reproducer for the MTP
 * chained-rollout heap-corruption bug.
 *
 * Background: with LLAMA_MTP_ROLLOUT>=2, the llama-server Vulkan path crashes
 * inside ggml_gallocr_reserve_n's calloc with "corrupted double-linked list".
 * Individual op tests (concat, argmax, get_rows, mul_mat) all pass on Vulkan at
 * MTP scale, so the suspicion is that the bug is in the *interaction* of ops
 * inside the full MTP block when the scheduler reserves/allocs at multiple
 * n_tokens shapes in sequence.
 *
 * This test replicates the chained-rollout graph without any model weights:
 *   for k in 0..n_rollout:
 *       emb      = get_rows(tok_embd, current_greedy)
 *       e_norm   = rms_norm(emb)
 *       h_norm   = rms_norm(current_hidden)
 *       combined = concat(e_norm, h_norm, dim=0)
 *       cur      = mul_mat(eh_proj, combined)
 *       cur      = flash_attn_ext(q=cur@wq, k=cur@wk, v=cur@wv, kq_mask)
 *       ffn      = mul_mat(ffn_down, silu(mul_mat(ffn_gate, cur)) * mul_mat(ffn_up, cur))
 *       cur_mtp  = ffn
 *       cur      = rms_norm(cur_mtp)
 *       logits_k = mul_mat(lm_head, cur)
 *       if k < n_rollout-1:
 *           current_greedy = argmax(clamp(logits_k, -1e4, 1e4))
 *           current_hidden = cur_mtp
 *   stacked = concat(concat(logits_0, logits_1, 1), logits_2, 1)
 *
 * Then the test exercises the scheduler in the same pattern as llama-server:
 *   1. reserve @ n_tokens=32 (worst-case init)
 *   2. alloc_graph + compute @ n_tokens=5 (runtime, triggers reserve_n on shape change)
 *   3. alloc_graph + compute @ n_tokens=1 (another shape change)
 *   4. alloc_graph + compute @ n_tokens=5 (shape shrink-grow again)
 *
 * Success = clean exit (negative result — bug needs more than the MTP block).
 * Failure = glibc "corrupted double-linked list" crash (positive result — we
 *           have a minimal repro to file against RADV).
 *
 * Build: cmake --build build-vk --target test-mtp-block-vulkan
 * Run:   build-vk/bin/test-mtp-block-vulkan
 */

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

// ============================================================================
// Model dimensions (match MTP block in production)
// ============================================================================
static const int64_t N_EMBD      = 1024;
static const int64_t N_VOCAB     = 32768;
static const int64_t N_FFN       = 3072;
static const int64_t N_HEAD      = 16;
static const int64_t N_HEAD_KV   = 16;
static const int64_t N_HEAD_DIM  = N_EMBD / N_HEAD;   // 64
static const uint32_t N_ROLLOUT  = 3;

// DeltaNet SSM stub dimensions. Chosen to match Qwen3.5 family while keeping
// N_EMBD=1024 compatible (one v-head per 64 dims). Hits the IQK fast path
// (head_k_dim ∈ {64,128}).
static const int64_t SSM_HEAD_K_DIM = 64;   // head_k_dim (= ssm_d_state)
static const int64_t SSM_HEAD_V_DIM = 64;   // head_v_dim
static const int64_t SSM_NUM_K_HEADS = 16;  // num_k_heads (= ssm_n_group)
static const int64_t SSM_NUM_V_HEADS = 16;  // num_v_heads (= ssm_dt_rank)
static const int64_t SSM_KEY_DIM   = SSM_HEAD_K_DIM * SSM_NUM_K_HEADS;   // 1024
static const int64_t SSM_VALUE_DIM = SSM_HEAD_V_DIM * SSM_NUM_V_HEADS;   // 1024

// Larger mem allocation for graph ctx — the chained rollout builds a lot of
// intermediates per iteration.
#define CTX_TENSORS   2048
#define CTX_MEM_BYTES (ggml_tensor_overhead()*CTX_TENSORS + ggml_graph_overhead_custom(CTX_TENSORS, false))

// ============================================================================
// Weight tensors — allocated ONCE on the Vulkan backend, reused across graphs
// ============================================================================
struct mtp_weights {
    ggml_context *       ctx  = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    // embedding + lm head
    ggml_tensor * tok_embd = nullptr;   // [N_EMBD, N_VOCAB]
    ggml_tensor * lm_head  = nullptr;   // [N_EMBD, N_VOCAB]

    // MTP projection
    ggml_tensor * eh_proj  = nullptr;   // [2*N_EMBD, N_EMBD]
    ggml_tensor * enorm_w  = nullptr;   // [N_EMBD]
    ggml_tensor * hnorm_w  = nullptr;   // [N_EMBD]
    ggml_tensor * attn_norm_w = nullptr; // [N_EMBD]
    ggml_tensor * head_norm_w = nullptr; // [N_EMBD]
    ggml_tensor * ffn_norm_w  = nullptr; // [N_EMBD]

    // Attention
    ggml_tensor * wq = nullptr;  // [N_EMBD, N_EMBD]
    ggml_tensor * wk = nullptr;  // [N_EMBD, N_HEAD_KV*N_HEAD_DIM]
    ggml_tensor * wv = nullptr;  // [N_EMBD, N_HEAD_KV*N_HEAD_DIM]
    ggml_tensor * wo = nullptr;  // [N_EMBD, N_EMBD]

    // FFN
    ggml_tensor * ffn_up   = nullptr;  // [N_EMBD, N_FFN]
    ggml_tensor * ffn_gate = nullptr;  // [N_EMBD, N_FFN]
    ggml_tensor * ffn_down = nullptr;  // [N_FFN, N_EMBD]

    // DeltaNet SSM stubs — minimal weights needed to call ggml_delta_net_ext
    // in the Qwen3.5 pattern. We skip ssm_conv1d entirely because the conv
    // path needs inp_s_seq_qnext + state_storage plumbing; the task targets
    // the delta_net_ext op itself and the graph shape before/after it.
    ggml_tensor * ssm_norm_w  = nullptr;  // [N_EMBD]
    ggml_tensor * ssm_in_q    = nullptr;  // [N_EMBD, SSM_KEY_DIM]
    ggml_tensor * ssm_in_k    = nullptr;  // [N_EMBD, SSM_KEY_DIM]
    ggml_tensor * ssm_in_v    = nullptr;  // [N_EMBD, SSM_VALUE_DIM]
    ggml_tensor * ssm_in_g    = nullptr;  // [N_EMBD, SSM_NUM_V_HEADS]
    ggml_tensor * ssm_in_beta = nullptr;  // [N_EMBD, SSM_NUM_V_HEADS]
    ggml_tensor * ssm_out_w   = nullptr;  // [SSM_VALUE_DIM, N_EMBD]
    ggml_tensor * ssm_final_norm_w = nullptr; // [SSM_HEAD_V_DIM]
};

static void alloc_weights(mtp_weights & w, ggml_backend_t backend) {
    // Enough for ~20 weight tensors
    struct ggml_init_params params = { ggml_tensor_overhead() * 64, nullptr, true };
    w.ctx = ggml_init(params);

    w.tok_embd = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, N_EMBD, N_VOCAB);
    w.lm_head  = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, N_EMBD, N_VOCAB);

    w.eh_proj  = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, 2*N_EMBD, N_EMBD);
    w.enorm_w  = ggml_new_tensor_1d(w.ctx, GGML_TYPE_F32, N_EMBD);
    w.hnorm_w  = ggml_new_tensor_1d(w.ctx, GGML_TYPE_F32, N_EMBD);
    w.attn_norm_w = ggml_new_tensor_1d(w.ctx, GGML_TYPE_F32, N_EMBD);
    w.head_norm_w = ggml_new_tensor_1d(w.ctx, GGML_TYPE_F32, N_EMBD);
    w.ffn_norm_w  = ggml_new_tensor_1d(w.ctx, GGML_TYPE_F32, N_EMBD);

    w.wq = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, N_EMBD, N_EMBD);
    w.wk = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, N_EMBD, N_HEAD_KV*N_HEAD_DIM);
    w.wv = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, N_EMBD, N_HEAD_KV*N_HEAD_DIM);
    w.wo = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, N_EMBD, N_EMBD);

    w.ffn_up   = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, N_EMBD, N_FFN);
    w.ffn_gate = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, N_EMBD, N_FFN);
    w.ffn_down = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, N_FFN,  N_EMBD);

    // SSM stub weights
    w.ssm_norm_w  = ggml_new_tensor_1d(w.ctx, GGML_TYPE_F32, N_EMBD);
    w.ssm_in_q    = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, N_EMBD, SSM_KEY_DIM);
    w.ssm_in_k    = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, N_EMBD, SSM_KEY_DIM);
    w.ssm_in_v    = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, N_EMBD, SSM_VALUE_DIM);
    w.ssm_in_g    = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, N_EMBD, SSM_NUM_V_HEADS);
    w.ssm_in_beta = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, N_EMBD, SSM_NUM_V_HEADS);
    w.ssm_out_w   = ggml_new_tensor_2d(w.ctx, GGML_TYPE_F32, SSM_VALUE_DIM, N_EMBD);
    w.ssm_final_norm_w = ggml_new_tensor_1d(w.ctx, GGML_TYPE_F32, SSM_HEAD_V_DIM);

    w.buf = ggml_backend_alloc_ctx_tensors(w.ctx, backend);
    if (!w.buf) {
        fprintf(stderr, "FATAL: failed to allocate weight buffer\n");
        exit(1);
    }
    ggml_backend_buffer_set_usage(w.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Fill with small deterministic values — don't want NaN propagation to
    // mask the real bug.
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-0.02f, 0.02f);

    auto fill = [&](ggml_tensor * t, float scale) {
        size_t n = ggml_nelements(t);
        std::vector<float> data(n);
        for (size_t i = 0; i < n; i++) data[i] = dist(rng) * scale;
        ggml_backend_tensor_set(t, data.data(), 0, n * sizeof(float));
    };

    fill(w.tok_embd, 1.0f);
    fill(w.lm_head,  1.0f);
    fill(w.eh_proj,  1.0f);

    // RMSNorm weights ~1.0
    auto fill_ones = [&](ggml_tensor * t) {
        size_t n = ggml_nelements(t);
        std::vector<float> data(n, 1.0f);
        ggml_backend_tensor_set(t, data.data(), 0, n * sizeof(float));
    };
    fill_ones(w.enorm_w);
    fill_ones(w.hnorm_w);
    fill_ones(w.attn_norm_w);
    fill_ones(w.head_norm_w);
    fill_ones(w.ffn_norm_w);

    fill(w.wq, 1.0f);
    fill(w.wk, 1.0f);
    fill(w.wv, 1.0f);
    fill(w.wo, 1.0f);
    fill(w.ffn_up,   1.0f);
    fill(w.ffn_gate, 1.0f);
    fill(w.ffn_down, 1.0f);

    fill_ones(w.ssm_norm_w);
    fill_ones(w.ssm_final_norm_w);
    fill(w.ssm_in_q,    1.0f);
    fill(w.ssm_in_k,    1.0f);
    fill(w.ssm_in_v,    1.0f);
    fill(w.ssm_in_g,    1.0f);
    fill(w.ssm_in_beta, 1.0f);
    fill(w.ssm_out_w,   1.0f);
}

// ============================================================================
// Graph builder — mirrors src/llama-build-context.cpp:4890-4976 chained
// rollout, simplified (no inp_out_ids, no model.output fallback, no KV cache
// store — just flash_attn_ext over per-graph K/V tensors).
// ============================================================================
struct mtp_graph {
    ggml_context * ctx = nullptr;
    ggml_cgraph *  gf  = nullptr;

    // Inputs — set before compute
    ggml_tensor * hidden_in    = nullptr;   // [N_EMBD, n_tokens]
    ggml_tensor * greedy_in    = nullptr;   // [n_tokens]  (i32)
    ggml_tensor * kq_mask      = nullptr;   // [n_kv, n_batch_pad]

    // One SSM state tensor per hybrid SSM layer, marked as graph input so the
    // scheduler allocates its memory. Zero-filled in set_graph_inputs to
    // mirror the reset_state_local path (position 0 batch) in
    // build_qkv/llama-delta-net.cpp.
    std::vector<ggml_tensor *> ssm_states;

    // Output
    ggml_tensor * stacked_out  = nullptr;   // [N_VOCAB, n_rollout * n_tokens]
};

static ggml_tensor * build_rms_norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w) {
    ggml_tensor * n = ggml_rms_norm(ctx, x, 1e-5f);
    return ggml_mul(ctx, n, w);
}

// Build one "stub" transformer layer (norm + flash_attn + norm + SiLU-PAR FFN).
// Reuses the MTP block's weights — we don't care about numerical accuracy
// here, only about stressing the scheduler with N attention+FFN subgraphs
// preceding the MTP block. Returns the output hidden tensor.
static ggml_tensor * build_stub_transformer_layer(
        ggml_context * ctx, const mtp_weights & w,
        ggml_tensor * hidden, ggml_tensor * kq_mask, int n_tokens, int layer_idx) {
    const float kq_scale = 1.0f / sqrtf((float) N_HEAD_DIM);

    // Attention
    ggml_tensor * attn_in = build_rms_norm(ctx, hidden, w.attn_norm_w);
    ggml_tensor * q_cur = ggml_mul_mat(ctx, w.wq, attn_in);
    ggml_tensor * k_cur = ggml_mul_mat(ctx, w.wk, attn_in);
    ggml_tensor * v_cur = ggml_mul_mat(ctx, w.wv, attn_in);

    ggml_tensor * q = ggml_reshape_3d(ctx, q_cur, N_HEAD_DIM, N_HEAD, n_tokens);
    q = ggml_permute(ctx, q, 0, 2, 1, 3);
    q = ggml_cont(ctx, q);

    ggml_tensor * kv_k = ggml_reshape_3d(ctx, k_cur, N_HEAD_DIM, N_HEAD_KV, n_tokens);
    kv_k = ggml_permute(ctx, kv_k, 0, 2, 1, 3);
    kv_k = ggml_cont(ctx, kv_k);
    kv_k = ggml_cast(ctx, kv_k, GGML_TYPE_F16);

    ggml_tensor * kv_v = ggml_reshape_3d(ctx, v_cur, N_HEAD_DIM, N_HEAD_KV, n_tokens);
    kv_v = ggml_permute(ctx, kv_v, 0, 2, 1, 3);
    kv_v = ggml_cont(ctx, kv_v);
    kv_v = ggml_cast(ctx, kv_v, GGML_TYPE_F16);

    ggml_tensor * kqv = ggml_flash_attn_ext(ctx, q, kv_k, kv_v, kq_mask, kq_scale, 0.0f, 0.0f);
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    ggml_tensor * attn_out = ggml_cont_2d(ctx, kqv, N_EMBD, n_tokens);
    attn_out = ggml_mul_mat(ctx, w.wo, attn_out);
    ggml_tensor * h1 = ggml_add(ctx, hidden, attn_out);

    // FFN
    ggml_tensor * ffn_in = build_rms_norm(ctx, h1, w.ffn_norm_w);
    ggml_tensor * gate   = ggml_mul_mat(ctx, w.ffn_gate, ffn_in);
    ggml_tensor * up     = ggml_mul_mat(ctx, w.ffn_up,   ffn_in);
    ggml_tensor * act    = ggml_silu(ctx, gate);
    ggml_tensor * mixed  = ggml_mul(ctx, act, up);
    ggml_tensor * down   = ggml_mul_mat(ctx, w.ffn_down, mixed);
    ggml_tensor * h2     = ggml_add(ctx, h1, down);
    (void) layer_idx;
    return h2;
}

// Build one "stub" DeltaNet SSM layer that calls ggml_delta_net_ext in the
// same shape as Qwen3.5's recurrent layers. Skips the conv1d path (which
// requires inp_s_seq_qnext plumbing) but preserves the critical permute+cont
// pattern from build_fused_delta_net — that pattern is where the CPU
// correctness fix at llama-delta-net.cpp:108-121 lives, and it's what the
// Vulkan backend has to re-materialize into contiguous memory at dispatch
// time. Returns the output hidden tensor.
//
// Dimension summary (pre-permute, ik_llama layout):
//   q: [head_k_dim, n_tokens, num_k_heads, n_seqs=1]
//   k: [head_k_dim, n_tokens, num_k_heads, n_seqs=1]
//   v: [head_v_dim, num_v_heads, n_tokens, n_seqs=1]  (permuted to this)
//   g: [num_v_heads, n_tokens, 1, n_seqs=1]           (permuted)
//   beta: [1, num_v_heads, n_tokens, n_seqs=1]        (permuted)
//   state: [head_v_dim, head_v_dim * num_v_heads, 1, n_seqs=1]
static ggml_tensor * build_stub_ssm_layer(
        ggml_context * ctx, const mtp_weights & w,
        ggml_tensor * hidden, int n_tokens, int layer_idx,
        std::vector<ggml_tensor *> * out_states) {
    const int64_t H_k = SSM_NUM_K_HEADS;
    const int64_t H_v = SSM_NUM_V_HEADS;
    const int64_t S_k = SSM_HEAD_K_DIM;
    const int64_t S_v = SSM_HEAD_V_DIM;
    const int64_t n_seqs = 1;
    GGML_ASSERT(H_v % H_k == 0);
    const float eps_norm = 1e-6f;
    (void) layer_idx;

    // Pre-SSM norm (RMSNorm, mirrors build_layer_attn_linear_core).
    ggml_tensor * cur = build_rms_norm(ctx, hidden, w.ssm_norm_w);

    // Linear projections to q/k/v/g/beta.
    //   q_raw: [SSM_KEY_DIM, n_tokens]   → reshape → [S_k, H_k, n_tokens, n_seqs]
    //   k_raw: [SSM_KEY_DIM, n_tokens]   → reshape → [S_k, H_k, n_tokens, n_seqs]
    //   v_raw: [SSM_VALUE_DIM, n_tokens] → reshape → [S_v, H_v, n_tokens, n_seqs]
    ggml_tensor * q_raw = ggml_mul_mat(ctx, w.ssm_in_q, cur);
    ggml_tensor * k_raw = ggml_mul_mat(ctx, w.ssm_in_k, cur);
    ggml_tensor * v_raw = ggml_mul_mat(ctx, w.ssm_in_v, cur);
    ggml_tensor * g_raw = ggml_mul_mat(ctx, w.ssm_in_g,    cur);  // [H_v, n_tokens]
    ggml_tensor * b_raw = ggml_mul_mat(ctx, w.ssm_in_beta, cur);  // [H_v, n_tokens]

    // Reshape into the pre-delta_net layout (matches build_qkvz outputs).
    ggml_tensor * q = ggml_reshape_4d(ctx, q_raw, S_k, H_k, n_tokens, n_seqs);
    ggml_tensor * k = ggml_reshape_4d(ctx, k_raw, S_k, H_k, n_tokens, n_seqs);

    // Match build_fused_delta_net input convention: q/k are [head_dim,
    // n_tokens, n_heads, n_seqs]. We produced [S_k, H_k, n_tokens, n_seqs],
    // so permute axes 1↔2 and force contiguous.
    q = ggml_permute(ctx, q, 0, 2, 1, 3);  // → [S_k, n_tokens, H_k, n_seqs]
    q = ggml_cont(ctx, q);
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    k = ggml_cont(ctx, k);

    // Match l2_norm applied by build_qkv post-conv. Without it, q/k are raw
    // and the downstream op still runs. Keep l2_norm for fidelity.
    q = ggml_l2_norm(ctx, q, eps_norm);
    k = ggml_l2_norm(ctx, k, eps_norm);

    // v goes in as [S_v, H_v, n_tokens, n_seqs] — build_fused_delta_net will
    // permute it to [S_v, n_tokens, H_v, n_seqs] and force contiguous.
    ggml_tensor * v = ggml_reshape_4d(ctx, v_raw, S_v, H_v, n_tokens, n_seqs);

    // g: [H_v, n_tokens] → [H_v, n_tokens, n_seqs=1] (3D)
    // build_fused_delta_net permutes g as (2,0,3,1), i.e. from its expected
    // input shape [H_v, n_tokens, n_seqs] → [n_tokens, H_v, 1, n_seqs].
    ggml_tensor * g = ggml_reshape_3d(ctx, g_raw, H_v, n_tokens, n_seqs);

    // beta: [H_v, n_tokens] → [H_v, 1, n_tokens, n_seqs=1] (4D).
    // build_fused_delta_net permutes beta as (2,0,1,3).
    ggml_tensor * beta = ggml_reshape_4d(ctx, b_raw, H_v, 1, n_tokens, n_seqs);

    // State tensor: DeltaNet recurrent state. In production, this is a view
    // of the KV cache row for this sequence. For the stub, we synthesize a
    // fresh zero state per graph build — matches the "reset_state_local"
    // path in build_qkv, which is what happens at pos==0.
    //
    // Shape pre-reshape: [S_v, S_v, H_v, n_seqs]. build_fused_delta_net
    // reshapes it to [S_v, S_v*H_v, 1, n_seqs] and cont's if needed.
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S_v, S_v, H_v, n_seqs);
    ggml_set_input(state);
    char sname[64];
    snprintf(sname, sizeof(sname), "ssm_stub_state_%d", layer_idx);
    ggml_set_name(state, sname);
    if (out_states) out_states->push_back(state);

    // Now replicate build_fused_delta_net's body: permute+cont of v/g/beta,
    // reshape state to 2D per-seq, call ggml_delta_net_ext with
    // emit_intermediates=false (simple inference path).
    v    = ggml_permute(ctx, v, 0, 2, 1, 3);
    g    = ggml_permute(ctx, g, 2, 0, 3, 1);
    beta = ggml_permute(ctx, beta, 2, 0, 1, 3);
    v    = ggml_cont(ctx, v);
    g    = ggml_cont(ctx, g);
    beta = ggml_cont(ctx, beta);

    ggml_tensor * state_flat = ggml_reshape_4d(ctx, state, S_v, S_v * H_v, 1, n_seqs);
    if (!ggml_is_contiguous(state_flat)) {
        state_flat = ggml_cont_4d(ctx, state_flat, S_v, S_v * H_v, 1, n_seqs);
    }

    ggml_tensor * fused = ggml_delta_net_ext(ctx, q, k, v, g, beta, state_flat, /*emit_intermediates=*/false);

    // Extract output tokens [S_v, H_v, n_tokens, n_seqs] from the 1D result.
    ggml_tensor * output_tokens = ggml_view_4d(ctx, fused,
            S_v, H_v, n_tokens, n_seqs,
            ggml_row_size(fused->type, S_v),
            ggml_row_size(fused->type, S_v * H_v),
            ggml_row_size(fused->type, S_v * H_v * n_tokens), 0);

    // Reshape back to [SSM_VALUE_DIM, n_tokens] for the output projection.
    ggml_tensor * attn_out_2d = ggml_reshape_2d(ctx, output_tokens, S_v * H_v, n_tokens);

    // Final RMS norm over the v-head dimension (per ik_llama build_gated_output).
    // We skip the gated z path to keep the stub minimal — the target op is
    // the delta_net_ext + surrounding scheduler pressure, not the gate.
    ggml_tensor * normed = ggml_reshape_2d(ctx, attn_out_2d, S_v, H_v * n_tokens);
    normed = build_rms_norm(ctx, normed, w.ssm_final_norm_w);
    normed = ggml_reshape_2d(ctx, normed, S_v * H_v, n_tokens);

    // Output projection back into the embedding space and residual add.
    ggml_tensor * out = ggml_mul_mat(ctx, w.ssm_out_w, normed);
    return ggml_add(ctx, hidden, out);
}

// Dispatch a single hybrid layer: attention at Qwen3.5 indices {3,7,11,...},
// SSM elsewhere. Matches the (il + 1) % 4 != 0 rule in llama-hparams.cpp:482.
static bool is_attention_layer_qwen35(int layer_idx) {
    return ((layer_idx + 1) % 4) == 0;  // 3, 7, 11, 15, 19, 23, ...
}

static ggml_tensor * build_hybrid_stub_layer(
        ggml_context * ctx, const mtp_weights & w,
        ggml_tensor * hidden, ggml_tensor * kq_mask, int n_tokens, int layer_idx,
        std::vector<ggml_tensor *> * out_states) {
    if (is_attention_layer_qwen35(layer_idx)) {
        return build_stub_transformer_layer(ctx, w, hidden, kq_mask, n_tokens, layer_idx);
    }
    return build_stub_ssm_layer(ctx, w, hidden, n_tokens, layer_idx, out_states);
}

// num_stub_layers: number of stub transformer layers to prepend before the MTP
// block. Zero = original behavior (pure MTP). Larger = approximates the full
// model's pre-MTP graph.
static void build_mtp_graph(mtp_graph & mg, const mtp_weights & w, int n_tokens,
                            int num_stub_layers = 0, bool use_hybrid_pattern = false) {
    struct ggml_init_params params = { CTX_MEM_BYTES, nullptr, true };
    mg.ctx = ggml_init(params);
    mg.gf  = ggml_new_graph_custom(mg.ctx, CTX_TENSORS, false);

    // Inputs
    mg.hidden_in = ggml_new_tensor_2d(mg.ctx, GGML_TYPE_F32, N_EMBD, n_tokens);
    ggml_set_input(mg.hidden_in);
    ggml_set_name(mg.hidden_in, "hidden_in");

    mg.greedy_in = ggml_new_tensor_1d(mg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_input(mg.greedy_in);
    ggml_set_name(mg.greedy_in, "greedy_in");

    // Flash attention mask: shape [n_kv, n_batch_pad]. We use n_kv=n_tokens
    // since we synthesize K,V from the current batch — matches MTP's within-
    // batch attention where kv_head is freshly written.
    const int n_batch_pad = GGML_PAD(n_tokens, GGML_KQ_MASK_PAD);
    mg.kq_mask = ggml_new_tensor_2d(mg.ctx, GGML_TYPE_F16, n_tokens, n_batch_pad);
    ggml_set_input(mg.kq_mask);
    ggml_set_name(mg.kq_mask, "kq_mask");

    // Optional: prepend N stub transformer layers. This mimics the pre-MTP
    // context in the full model. Reuses the same weights — we only care about
    // stressing the scheduler, not numerical accuracy.
    //
    // When use_hybrid_pattern is true, dispatch each layer through the
    // Qwen3.5 hybrid rule (attention at indices {3,7,11,...}, SSM elsewhere).
    // Matches llama-hparams.cpp:482: recurrent_layer_arr[i] = (i+1) % 4 != 0.
    ggml_tensor * pre_mtp_hidden = mg.hidden_in;
    for (int li = 0; li < num_stub_layers; ++li) {
        if (use_hybrid_pattern) {
            pre_mtp_hidden = build_hybrid_stub_layer(
                mg.ctx, w, pre_mtp_hidden, mg.kq_mask, n_tokens, li, &mg.ssm_states);
        } else {
            pre_mtp_hidden = build_stub_transformer_layer(
                mg.ctx, w, pre_mtp_hidden, mg.kq_mask, n_tokens, li);
        }
    }

    ggml_tensor * current_hidden = pre_mtp_hidden;
    ggml_tensor * current_greedy = mg.greedy_in;

    std::vector<ggml_tensor *> rollout_logits;
    rollout_logits.reserve(N_ROLLOUT);

    const float kq_scale = 1.0f / sqrtf((float) N_HEAD_DIM);

    for (uint32_t k = 0; k < N_ROLLOUT; ++k) {
        // 1. emb = get_rows(tok_embd, current_greedy)
        ggml_tensor * emb = ggml_get_rows(mg.ctx, w.tok_embd, current_greedy);
        ggml_set_name(emb, "emb");

        // 2/3. RMSnorm of emb and current_hidden
        ggml_tensor * e_norm = build_rms_norm(mg.ctx, emb,            w.enorm_w);
        ggml_tensor * h_norm = build_rms_norm(mg.ctx, current_hidden, w.hnorm_w);

        // 4. concat along dim=0  →  [2*N_EMBD, n_tokens]
        ggml_tensor * combined = ggml_concat(mg.ctx, e_norm, h_norm, 0);
        ggml_set_name(combined, "combined");

        // 5. cur = mul_mat(eh_proj, combined)  →  [N_EMBD, n_tokens]
        ggml_tensor * cur = ggml_mul_mat(mg.ctx, w.eh_proj, combined);
        ggml_set_name(cur, "eh_proj_out");

        // 6. Self-attention via flash_attn_ext
        ggml_tensor * attn_in = build_rms_norm(mg.ctx, cur, w.attn_norm_w);

        ggml_tensor * q_cur = ggml_mul_mat(mg.ctx, w.wq, attn_in);
        ggml_tensor * k_cur = ggml_mul_mat(mg.ctx, w.wk, attn_in);
        ggml_tensor * v_cur = ggml_mul_mat(mg.ctx, w.wv, attn_in);

        // q/k/v shapes for flash_attn_ext:
        //   q: [head_dim, n_tokens, n_head, 1]
        //   k: [head_dim, n_kv,     n_head_kv, 1]
        //   v: [head_dim, n_kv,     n_head_kv, 1]
        ggml_tensor * q = ggml_reshape_3d(mg.ctx, q_cur, N_HEAD_DIM, N_HEAD,    n_tokens);
        q = ggml_permute(mg.ctx, q, 0, 2, 1, 3);  // [hd, n_tok, n_head]
        q = ggml_cont(mg.ctx, q);

        // Vulkan flash_attn_ext requires K/V to be f16 (or quantized). Cast
        // from f32 to match the production path (where KV cache is f16).
        ggml_tensor * kv_k = ggml_reshape_3d(mg.ctx, k_cur, N_HEAD_DIM, N_HEAD_KV, n_tokens);
        kv_k = ggml_permute(mg.ctx, kv_k, 0, 2, 1, 3);
        kv_k = ggml_cont(mg.ctx, kv_k);
        kv_k = ggml_cast(mg.ctx, kv_k, GGML_TYPE_F16);

        ggml_tensor * kv_v = ggml_reshape_3d(mg.ctx, v_cur, N_HEAD_DIM, N_HEAD_KV, n_tokens);
        kv_v = ggml_permute(mg.ctx, kv_v, 0, 2, 1, 3);
        kv_v = ggml_cont(mg.ctx, kv_v);
        kv_v = ggml_cast(mg.ctx, kv_v, GGML_TYPE_F16);

        ggml_tensor * kqv = ggml_flash_attn_ext(mg.ctx, q, kv_k, kv_v, mg.kq_mask,
                                                 kq_scale, 0.0f, 0.0f);
        // flash_attn_ext returns [head_dim, n_head, n_tokens, 1] permuted
        kqv = ggml_permute(mg.ctx, kqv, 0, 2, 1, 3);   // undo
        ggml_tensor * attn_out = ggml_cont_2d(mg.ctx, kqv, N_EMBD, n_tokens);
        attn_out = ggml_mul_mat(mg.ctx, w.wo, attn_out);

        // residual
        cur = ggml_add(mg.ctx, cur, attn_out);

        // 7. FFN with SiLU gate (PAR)
        ggml_tensor * ffn_in = build_rms_norm(mg.ctx, cur, w.ffn_norm_w);
        ggml_tensor * gate   = ggml_mul_mat(mg.ctx, w.ffn_gate, ffn_in);
        ggml_tensor * up     = ggml_mul_mat(mg.ctx, w.ffn_up,   ffn_in);
        ggml_tensor * act    = ggml_silu(mg.ctx, gate);
        ggml_tensor * mixed  = ggml_mul(mg.ctx, act, up);
        ggml_tensor * down   = ggml_mul_mat(mg.ctx, w.ffn_down, mixed);

        cur = ggml_add(mg.ctx, cur, down);

        // 8. capture pre-norm hidden for next iter
        ggml_tensor * mtp_hidden_next = cur;

        // 9. shared head norm
        ggml_tensor * cur_head = build_rms_norm(mg.ctx, cur, w.head_norm_w);

        // 10. logits
        ggml_tensor * logits_k = ggml_mul_mat(mg.ctx, w.lm_head, cur_head);
        ggml_set_output(logits_k);
        ggml_build_forward_expand(mg.gf, logits_k);
        rollout_logits.push_back(logits_k);

        // 11. greedy for next iter
        if (k + 1 < N_ROLLOUT) {
            ggml_tensor * clamped = ggml_clamp(mg.ctx, logits_k, -1e4f, 1e4f);
            current_greedy = ggml_argmax(mg.ctx, clamped);
            ggml_set_input(current_greedy);
            ggml_set_output(current_greedy);
            current_hidden = mtp_hidden_next;
        }
    }

    // 12. stack iteration-major
    ggml_tensor * stacked = rollout_logits[0];
    for (size_t i = 1; i < rollout_logits.size(); ++i) {
        stacked = ggml_concat(mg.ctx, stacked, rollout_logits[i], 1);
    }
    mg.stacked_out = stacked;
    ggml_set_output(stacked);
    ggml_build_forward_expand(mg.gf, stacked);
}

static void free_graph(mtp_graph & mg) {
    if (mg.ctx) ggml_free(mg.ctx);
    mg.ctx = nullptr;
    mg.gf  = nullptr;
}

// ============================================================================
// Set graph inputs with random data
// ============================================================================
static void set_graph_inputs(mtp_graph & mg, int n_tokens) {
    std::mt19937 rng(9001 + n_tokens);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // hidden_in
    {
        size_t n = ggml_nelements(mg.hidden_in);
        std::vector<float> data(n);
        for (size_t i = 0; i < n; i++) data[i] = dist(rng);
        ggml_backend_tensor_set(mg.hidden_in, data.data(), 0, n * sizeof(float));
    }

    // greedy_in — valid i32 token ids < N_VOCAB
    {
        std::vector<int32_t> ids(n_tokens);
        for (int i = 0; i < n_tokens; i++) ids[i] = i % (int) N_VOCAB;
        ggml_backend_tensor_set(mg.greedy_in, ids.data(), 0, ids.size() * sizeof(int32_t));
    }

    // kq_mask — all zeros (no masking) in f16
    {
        size_t n = ggml_nelements(mg.kq_mask);
        std::vector<ggml_fp16_t> mask(n, ggml_fp32_to_fp16(0.0f));
        ggml_backend_tensor_set(mg.kq_mask, mask.data(), 0, n * sizeof(ggml_fp16_t));
    }

    // SSM states — zero-fill (mirrors reset_state_local path in build_qkv).
    for (ggml_tensor * st : mg.ssm_states) {
        size_t n = ggml_nelements(st);
        std::vector<float> zeros(n, 0.0f);
        ggml_backend_tensor_set(st, zeros.data(), 0, n * sizeof(float));
    }
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char ** argv) {
    (void) argc; (void) argv;
    printf("=== test-mtp-block-vulkan: MTP chained-rollout graph reproducer ===\n");

    // Number of stub transformer layers to prepend before the MTP block.
    // Configurable via STUB_LAYERS env var. Default 0 = pure MTP block only
    // (original behavior — confirmed clean). Increase to bisect the crash:
    //   STUB_LAYERS=4 ./test-mtp-block-vulkan
    //   STUB_LAYERS=8 ./test-mtp-block-vulkan
    //   STUB_LAYERS=16 ./test-mtp-block-vulkan
    //   STUB_LAYERS=32 ./test-mtp-block-vulkan  (production layer count)
    int num_stub_layers = 0;
    if (const char * e = std::getenv("STUB_LAYERS")) {
        num_stub_layers = std::atoi(e);
    }
    printf("num_stub_layers = %d (set STUB_LAYERS env to change)\n", num_stub_layers);

    // HYBRID_SSM=1 switches the stub layers to Qwen3.5's hybrid pattern:
    // attention at layer indices {3, 7, 11, 15, 19, 23, ...}, DeltaNet SSM
    // elsewhere (via ggml_delta_net_ext). This tests whether the hybrid
    // SSM+attention graph shape alone reproduces the Vulkan rollout>=2 crash.
    bool use_hybrid_pattern = false;
    if (const char * e = std::getenv("HYBRID_SSM")) {
        use_hybrid_pattern = std::atoi(e) != 0;
    }
    printf("use_hybrid_pattern = %d (set HYBRID_SSM env to change)\n", use_hybrid_pattern);
    if (use_hybrid_pattern && num_stub_layers > 0) {
        printf("  hybrid layout: ");
        for (int li = 0; li < num_stub_layers; ++li) {
            printf("%s%s", is_attention_layer_qwen35(li) ? "A" : "S",
                   (li + 1 < num_stub_layers) ? "" : "\n");
        }
    }
    fflush(stdout);

#ifndef GGML_USE_VULKAN
    printf("GGML_USE_VULKAN not defined — skipping.\n");
    return 0;
#else
    if (ggml_backend_vk_get_device_count() == 0) {
        printf("No Vulkan devices found — skipping.\n");
        return 0;
    }

    char desc[256];
    ggml_backend_vk_get_device_description(0, desc, sizeof(desc));
    printf("Vulkan device 0: %s\n", desc);
    fflush(stdout);

    ggml_backend_t backend = ggml_backend_vk_init(0);
    if (!backend) {
        fprintf(stderr, "FATAL: ggml_backend_vk_init(0) failed\n");
        return 1;
    }
    printf("Backend initialized: %s\n", ggml_backend_name(backend));
    fflush(stdout);

    // Optional second Vulkan GPU — matches production setup (NAVI21 + Vega).
    // When present, the scheduler has to place tensors across 3 backends and
    // insert cross-backend copy nodes, which may be what triggers the RADV
    // heap corruption on rollout>=2.
    //    MULTI_GPU=1 (default): init Vulkan0 + Vulkan1 if available
    //    MULTI_GPU=0:           single Vulkan GPU only
    int multi_gpu = 1;
    if (const char * e = std::getenv("MULTI_GPU")) multi_gpu = std::atoi(e);

    ggml_backend_t backend2 = nullptr;
    if (multi_gpu && ggml_backend_vk_get_device_count() >= 2) {
        backend2 = ggml_backend_vk_init(1);
        if (backend2) {
            char desc2[256];
            ggml_backend_vk_get_device_description(1, desc2, sizeof(desc2));
            printf("Vulkan device 1: %s\n", desc2);
            printf("Multi-GPU Vulkan mode active\n");
        }
    }
    fflush(stdout);

    // Allocate weights on primary backend
    mtp_weights w;
    alloc_weights(w, backend);
    printf("Weights allocated (buffer size = %zu bytes)\n",
           ggml_backend_buffer_get_size(w.buf));
    fflush(stdout);

    // Create scheduler. The sched API requires the last backend to be CPU (for
    // ops unsupported by Vulkan to fall back). This matches how llama.cpp
    // creates its scheduler.
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (!cpu_backend) {
        fprintf(stderr, "FATAL: ggml_backend_cpu_init() failed\n");
        return 1;
    }

    const size_t graph_size = 8192;
    int n_backends = backend2 ? 3 : 2;
    ggml_backend_t backends[3] = { backend, backend2 ? backend2 : cpu_backend, backend2 ? cpu_backend : nullptr };
    ggml_backend_buffer_type_t bufts[3] = {
        ggml_backend_get_default_buffer_type(backend),
        ggml_backend_get_default_buffer_type(backend2 ? backend2 : cpu_backend),
        backend2 ? ggml_backend_get_default_buffer_type(cpu_backend) : nullptr,
    };
    // Match production: pipeline_parallel=true when we have 2+ GPUs.
    // This enables the scheduler's pipeline-parallel path which creates
    // multiple event/copy buffers per backend.
    int pipeline_parallel = 0;
    if (const char * e = std::getenv("PIPELINE_PARALLEL")) pipeline_parallel = std::atoi(e);
    else pipeline_parallel = backend2 ? 1 : 0;
    printf("pipeline_parallel = %d\n", pipeline_parallel);
    fflush(stdout);
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, n_backends, graph_size, pipeline_parallel != 0);
    if (!sched) {
        fprintf(stderr, "FATAL: ggml_backend_sched_new failed\n");
        return 1;
    }
    printf("Scheduler created\n");
    fflush(stdout);

    // Step 1: reserve at n_tokens=32 (worst-case init, matches llama.cpp)
    {
        printf("\n--- Step 1: reserve @ n_tokens=32 (worst-case init) ---\n");
        fflush(stdout);
        mtp_graph mg;
        build_mtp_graph(mg, w, 32, num_stub_layers, use_hybrid_pattern);
        printf("  graph built: n_nodes=%d n_leafs=%d\n", mg.gf->n_nodes, mg.gf->n_leafs);
        fflush(stdout);

        bool ok = ggml_backend_sched_reserve(sched, mg.gf);
        if (!ok) {
            fprintf(stderr, "FATAL: reserve @ n_tokens=32 failed\n");
            return 1;
        }
        printf("  reserve OK (compute buf = %zu bytes)\n",
               ggml_backend_sched_get_buffer_size(sched, backend));
        fflush(stdout);
        free_graph(mg);
    }

    // Steps 2..4: cycle through runtime shapes, triggering reserve_n calls
    const int runtime_shapes[] = { 5, 1, 5, 2, 5, 1, 8, 5 };
    const int n_steps = sizeof(runtime_shapes) / sizeof(runtime_shapes[0]);

    for (int step = 0; step < n_steps; step++) {
        int n_tokens = runtime_shapes[step];
        printf("\n--- Step %d: alloc+compute @ n_tokens=%d ---\n", step + 2, n_tokens);
        fflush(stdout);

        mtp_graph mg;
        build_mtp_graph(mg, w, n_tokens, num_stub_layers, use_hybrid_pattern);
        printf("  graph built: n_nodes=%d n_leafs=%d\n", mg.gf->n_nodes, mg.gf->n_leafs);
        fflush(stdout);

        ggml_backend_sched_reset(sched);

        // This is where reserve_n is triggered internally when shape grows
        bool alloc_ok = ggml_backend_sched_alloc_graph(sched, mg.gf);
        if (!alloc_ok) {
            fprintf(stderr, "FATAL: sched_alloc_graph @ n_tokens=%d failed\n", n_tokens);
            return 1;
        }
        printf("  alloc_graph OK (compute buf = %zu bytes)\n",
               ggml_backend_sched_get_buffer_size(sched, backend));
        fflush(stdout);

        // Set input data AFTER alloc (inputs live in scheduler buffer)
        set_graph_inputs(mg, n_tokens);

        enum ggml_status st = ggml_backend_sched_graph_compute(sched, mg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FATAL: compute @ n_tokens=%d failed, status=%d\n", n_tokens, (int) st);
            return 1;
        }
        printf("  compute OK\n");
        fflush(stdout);

        // Read back a few output values to force sync
        std::vector<float> out(std::min<size_t>(16, ggml_nelements(mg.stacked_out)));
        ggml_backend_tensor_get(mg.stacked_out, out.data(), 0, out.size() * sizeof(float));
        printf("  stacked[0..3] = %.4f %.4f %.4f %.4f\n",
               out[0], out[1], out[2], out[3]);
        fflush(stdout);

        free_graph(mg);
    }

    printf("\n=== All %d steps completed WITHOUT crash ===\n", n_steps + 1);
    printf("Result: MTP block alone does NOT reproduce the heap corruption.\n");
    printf("        The bug requires additional graph context (e.g., full\n");
    printf("        transformer layers running first).\n");
    fflush(stdout);

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(w.buf);
    ggml_free(w.ctx);
    ggml_backend_free(backend);
    if (backend2) ggml_backend_free(backend2);
    ggml_backend_free(cpu_backend);
    return 0;
#endif
}
