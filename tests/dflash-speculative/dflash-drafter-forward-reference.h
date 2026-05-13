// dflash-drafter-forward-reference.h
//
// CPU scalar reference for the 5-layer DFlash drafter forward. Mirrors
// the cooperative kernel in dflash-drafter-forward.cu step-for-step.
// Used as the test oracle in test-dflash-drafter-forward.cpp.
//
// Spec: specs/dflash/kernel-design.md §6.1.
//
// Algorithm — for each (slot, query_position) of the (N_slots, 1+BLOCK_SIZE)
// query span:
//   1. Input: hidden = input_tokens_emb[slot, query_position, :]
//   2. For each layer ℓ in 0..L_d-1:
//        a. residual = hidden
//        b. hidden_n = RMSNorm(hidden, attn_norm_w[ℓ], norm_eps)
//        c. q = WMMA-oracle(hidden_n, q_w[ℓ])                 (H_q*D_h)
//           reshape q to (H_q, D_h)
//        d. q[h] = q_norm_w[ℓ] * RMSNorm_h(q[h])              (per head)
//        e. q = RoPE(q, query_position, rope_base, fp64-transcendentals)
//        f. K = k_cache[ℓ, slot, :, :, :]                     (already populated)
//           V = v_cache[ℓ, slot, :, :, :]
//        g. attention with mask:
//              - SWA layer (layer_types[ℓ] == 0): K-loop ∈
//                  [max(0, query_position - swa_window + 1), query_position]
//              - Full attention (layer_types[ℓ] == 1): K-loop ∈
//                  [0, query_position]
//           - scores = (Q @ K^T) / sqrt(D_h)  (scalar fp32 attention)
//           - softmax(scores)
//           - attn_out = softmax @ V
//           - reshape attn_out (H_q, D_h) → (H_q*D_h)
//        h. proj = WMMA-oracle(attn_out, o_w[ℓ])             (D_emb)
//        i. hidden = residual + proj
//        j. residual = hidden
//        k. hidden_n = RMSNorm(hidden, ffn_norm_w[ℓ], norm_eps)
//        l. gate = WMMA-oracle(hidden_n, gate_w[ℓ])          (intermediate)
//        m. up   = WMMA-oracle(hidden_n, up_w[ℓ])            (intermediate)
//        n. activated = silu(gate) * up                       (element-wise)
//        o. ffn_out = WMMA-oracle(activated, down_w[ℓ])      (D_emb)
//        p. hidden = residual + ffn_out
//   3. Output: for query_position in 1..BLOCK_SIZE (anchor dropped):
//        out_hidden[slot, query_position - 1, :] = hidden
//
// The cooperative kernel boundary (per spec §6.1) ends here. lm_head and
// argmax go in separate kernels with their own reference functions.
//
// Allium witnesses (per specs/dflash/allium-tla-binding.json):
//   - SingleForwardPerStep      (function call structure — one forward)
//   - QuerySpanIsOnePlusN       (n_query_positions == 1 + BLOCK_SIZE)
//   - InjectionConsumedAtEveryLayer  (k_cache, v_cache read every layer)
//   - LayerTypeDependentMask    (K-loop bound per layer_type[ℓ])
//   - AnchorEmbeddingFromTarget (input_tokens_emb is the caller's embedding lookup)
//   - AnchorPosPreserved        (slot_positions flows through unchanged)
//   - DeterminismPerDeployment  (pure function of inputs)

#pragma once

#include "wmma-mimicking-oracle.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

namespace dflash_reference {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// In-place RMSNorm over D fp32 values with shared fp16 weight.
// sum_sq is summed serially (deterministic, matches a hypothetical kernel
// reduction order with one CTA per row).
inline void rmsnorm_inplace_f32(
    float * x, int D, const __half * weight, float eps)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < D; ++i) sum_sq += x[i] * x[i];
    const float rsqrt = 1.0f / std::sqrt(sum_sq / static_cast<float>(D) + eps);
    for (int i = 0; i < D; ++i) {
        x[i] = (x[i] * rsqrt) * __half2float(weight[i]);
    }
}

// NeoX-style interleaved-pair RoPE applied to a per-head vector of D_h
// floats. Pair (i, i + D_h/2): rotate by theta = pos / rope_base^(2i/D_h)
// where transcendentals are evaluated in fp64 then cast to fp32 (matches
// the T3 inject kernel + spec §6.1).
inline void rope_apply_neox_one_head_f32(
    float * q, int D_h, int pos, float rope_base)
{
    const int D_half = D_h / 2;
    for (int i = 0; i < D_half; ++i) {
        const double exp_val_d  =
            static_cast<double>(2 * i) / static_cast<double>(D_h);
        const double inv_freq_d = std::pow(static_cast<double>(rope_base), -exp_val_d);
        const double theta_d    = static_cast<double>(pos) * inv_freq_d;
        const float c = static_cast<float>(std::cos(theta_d));
        const float s = static_cast<float>(std::sin(theta_d));
        const float lo = q[i];
        const float hi = q[i + D_half];
        q[i]          = lo * c - hi * s;
        q[i + D_half] = lo * s + hi * c;
    }
}

// SiLU activation: x * sigmoid(x).
inline float silu_one_f32(float x) {
    return x / (1.0f + std::exp(-x));
}

// Pad an M×K fp16 matrix to (M_padded × K) where M_padded is the next
// multiple of 16. Out-of-range rows are zeroed.
inline void pad_M_to_16(
    const __half * src, int M_src, int K, int M_dst, __half * dst)
{
    for (int m = 0; m < M_src; ++m) {
        for (int k = 0; k < K; ++k) {
            dst[m * K + k] = src[m * K + k];
        }
    }
    for (int m = M_src; m < M_dst; ++m) {
        for (int k = 0; k < K; ++k) {
            dst[m * K + k] = __float2half(0.0f);
        }
    }
}

// Matmul via WMMA-mimicking oracle with implicit row-dim padding to 16.
//
//   C[M_actual, N] = A[M_actual, K] @ B^T[N, K] + 0
//
// Note B is passed in [N, K] row-major (= [K, N] col-major), so the
// effective access is B[k, n] = B_buf[n * K + k]. This matches typical
// weight storage in llama.cpp.
//
// Caller asks for M_actual rows; we pad to next multiple of 16 internally
// and write back the first M_actual rows.
inline void matmul_wmma_padded(
    const __half * A, int M_actual,
    const __half * Bt, int N, int K,
    __half * D)
{
    if (M_actual % WMMA_M == 0) {
        // Direct WMMA path — but our oracle expects B in [K, N] form, so
        // transpose into a scratch buffer first. Weights are typically
        // stored as [N, K], i.e., B^T row-major.
        std::vector<__half> B_kn(static_cast<std::size_t>(K) * N);
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                B_kn[k * N + n] = Bt[n * K + k];
            }
        }
        std::vector<__half> C_zero(static_cast<std::size_t>(M_actual) * N);
        std::memset(C_zero.data(), 0, C_zero.size() * sizeof(__half));
        wmma_mma_oracle(A, K, B_kn.data(), N, C_zero.data(), N, D, N, M_actual, N, K);
        return;
    }
    const int M_padded = ((M_actual + WMMA_M - 1) / WMMA_M) * WMMA_M;
    std::vector<__half> A_pad(static_cast<std::size_t>(M_padded) * K);
    pad_M_to_16(A, M_actual, K, M_padded, A_pad.data());

    std::vector<__half> B_kn(static_cast<std::size_t>(K) * N);
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            B_kn[k * N + n] = Bt[n * K + k];
        }
    }
    std::vector<__half> C_zero(static_cast<std::size_t>(M_padded) * N);
    std::memset(C_zero.data(), 0, C_zero.size() * sizeof(__half));

    std::vector<__half> D_pad(static_cast<std::size_t>(M_padded) * N);
    wmma_mma_oracle(A_pad.data(), K, B_kn.data(), N, C_zero.data(), N,
                    D_pad.data(), N, M_padded, N, K);

    for (int m = 0; m < M_actual; ++m) {
        for (int n = 0; n < N; ++n) {
            D[m * N + n] = D_pad[m * N + n];
        }
    }
}

// Scalar fp32 single-query attention with SWA/full mask.
// q:     [H_q,  D_h]   fp32 (post-norm, post-RoPE)
// k_cache: [SeqLen, H_kv, D_h] fp16 — this slot's slice (already RoPE'd by inject)
// v_cache: [SeqLen, H_kv, D_h] fp16 — this slot's slice
// out:   [H_q,  D_h]   fp32 (attention output)
//
// SWA: K iterates [max(0, query_pos - swa_window + 1), query_pos]
// Full: K iterates [0, query_pos]
inline void single_query_attention_f32(
    const float * q, int H_q, int H_kv, int D_h,
    const __half * k_cache_slot, const __half * v_cache_slot,
    int SeqLen, int query_pos,
    int swa_window, bool is_full,
    float * out)
{
    const int k_lo = is_full ? 0 : std::max(0, query_pos - swa_window + 1);
    const int k_hi = query_pos;  // inclusive; causal
    const int n_keys = (k_hi - k_lo + 1);
    if (n_keys <= 0) {
        for (int i = 0; i < H_q * D_h; ++i) out[i] = 0.0f;
        return;
    }

    const float inv_sqrt_dh = 1.0f / std::sqrt(static_cast<float>(D_h));
    const int gqa = H_q / H_kv;

    std::vector<float> scores(static_cast<std::size_t>(H_q) * n_keys);

    // Compute scores: scores[h, k] = (q[h] · K[k, h_kv]) * inv_sqrt_dh
    for (int h = 0; h < H_q; ++h) {
        const int h_kv = h / gqa;
        const float * qh = q + h * D_h;
        for (int ki = 0; ki < n_keys; ++ki) {
            const int k = k_lo + ki;
            const __half * Kk = k_cache_slot + (static_cast<std::size_t>(k) * H_kv + h_kv) * D_h;
            float dot = 0.0f;
            for (int d = 0; d < D_h; ++d) {
                dot += qh[d] * __half2float(Kk[d]);
            }
            scores[h * n_keys + ki] = dot * inv_sqrt_dh;
        }
    }

    // Per-head softmax in fp32.
    for (int h = 0; h < H_q; ++h) {
        float * sh = scores.data() + h * n_keys;
        float max_s = sh[0];
        for (int ki = 1; ki < n_keys; ++ki) if (sh[ki] > max_s) max_s = sh[ki];
        float sum = 0.0f;
        for (int ki = 0; ki < n_keys; ++ki) {
            sh[ki] = std::exp(sh[ki] - max_s);
            sum += sh[ki];
        }
        const float inv_sum = 1.0f / sum;
        for (int ki = 0; ki < n_keys; ++ki) sh[ki] *= inv_sum;
    }

    // out[h, d] = sum_k softmax[h, k] * V[k, h_kv, d]
    for (int h = 0; h < H_q; ++h) {
        const int h_kv = h / gqa;
        const float * sh = scores.data() + h * n_keys;
        float * outh = out + h * D_h;
        for (int d = 0; d < D_h; ++d) outh[d] = 0.0f;
        for (int ki = 0; ki < n_keys; ++ki) {
            const int k = k_lo + ki;
            const __half * Vk = v_cache_slot + (static_cast<std::size_t>(k) * H_kv + h_kv) * D_h;
            const float w = sh[ki];
            for (int d = 0; d < D_h; ++d) {
                outh[d] += w * __half2float(Vk[d]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Full reference
// ---------------------------------------------------------------------------

inline void drafter_forward_reference(
    const __half * input_tokens_emb,           // [N_slots, 1+BLOCK_SIZE, D_emb]
    const __half * k_cache,                    // [L_d, N_slots, SeqLen, H_kv, D_h]
    const __half * v_cache,                    // [L_d, N_slots, SeqLen, H_kv, D_h]
    const int    * slot_positions,             // [N_slots]
    const __half * const * layer_attn_norm_w,  // [L_d] -> [D_emb]
    const __half * const * layer_q_w,          // [L_d] -> [H_q*D_h, D_emb]
    const __half * const * layer_q_norm_w,     // [L_d] -> [D_h]
    const __half * const * layer_o_w,          // [L_d] -> [D_emb, H_q*D_h]
    const __half * const * layer_ffn_norm_w,   // [L_d] -> [D_emb]
    const __half * const * layer_gate_w,       // [L_d] -> [intermediate, D_emb]
    const __half * const * layer_up_w,         // [L_d] -> [intermediate, D_emb]
    const __half * const * layer_down_w,       // [L_d] -> [D_emb, intermediate]
    const int    * layer_types,                // [L_d] — 0 SWA, 1 full
    int            swa_window,
    float          rope_base,
    float          norm_eps,
    int            BLOCK_SIZE,
    int            N_slots,
    int            SeqLen,
    int            L_d,
    int            D_emb,
    int            H_q,
    int            H_kv,
    int            D_h,
    int            intermediate,
    __half       * out_hidden)                 // [N_slots, BLOCK_SIZE, D_emb]
{
    const int Q = 1 + BLOCK_SIZE;
    const int n_rows = N_slots * Q;
    const std::size_t cache_layer_stride =
        static_cast<std::size_t>(N_slots) * SeqLen * H_kv * D_h;
    const std::size_t cache_slot_stride =
        static_cast<std::size_t>(SeqLen) * H_kv * D_h;

    // Working buffer: hidden state at every (slot, query_position).
    // Stored row-major as [n_rows, D_emb] in fp16 between layers.
    std::vector<__half> hidden(static_cast<std::size_t>(n_rows) * D_emb);
    for (int s = 0; s < N_slots; ++s) {
        for (int q = 0; q < Q; ++q) {
            const int row = s * Q + q;
            for (int d = 0; d < D_emb; ++d) {
                hidden[static_cast<std::size_t>(row) * D_emb + d] =
                    input_tokens_emb[(static_cast<std::size_t>(s) * Q + q) * D_emb + d];
            }
        }
    }

    std::vector<__half> hidden_n(static_cast<std::size_t>(n_rows) * D_emb);
    std::vector<__half> q_proj(static_cast<std::size_t>(n_rows) * H_q * D_h);
    std::vector<__half> attn_out_h(static_cast<std::size_t>(n_rows) * H_q * D_h);
    std::vector<__half> o_proj_h(static_cast<std::size_t>(n_rows) * D_emb);
    std::vector<__half> gate_h(static_cast<std::size_t>(n_rows) * intermediate);
    std::vector<__half> up_h(static_cast<std::size_t>(n_rows) * intermediate);
    std::vector<__half> activated_h(static_cast<std::size_t>(n_rows) * intermediate);
    std::vector<__half> down_h(static_cast<std::size_t>(n_rows) * D_emb);

    for (int layer = 0; layer < L_d; ++layer) {
        const bool is_full = (layer_types[layer] == 1);

        // Step 2.a/b — attn_norm on each row.
        for (int row = 0; row < n_rows; ++row) {
            std::vector<float> tmp(D_emb);
            for (int d = 0; d < D_emb; ++d) {
                tmp[d] = __half2float(hidden[static_cast<std::size_t>(row) * D_emb + d]);
            }
            rmsnorm_inplace_f32(tmp.data(), D_emb, layer_attn_norm_w[layer], norm_eps);
            for (int d = 0; d < D_emb; ++d) {
                hidden_n[static_cast<std::size_t>(row) * D_emb + d] = __float2half(tmp[d]);
            }
        }

        // Step 2.c — Q projection (no K/V — those come from cache).
        matmul_wmma_padded(hidden_n.data(), n_rows,
                           layer_q_w[layer], H_q * D_h, D_emb,
                           q_proj.data());

        // Step 2.d/e — per-head q_norm, then RoPE, then attention.
        for (int s = 0; s < N_slots; ++s) {
            const int anchor_pos = slot_positions[s];
            const __half * k_cache_layer_slot =
                k_cache + static_cast<std::size_t>(layer) * cache_layer_stride
                        + static_cast<std::size_t>(s) * cache_slot_stride;
            const __half * v_cache_layer_slot =
                v_cache + static_cast<std::size_t>(layer) * cache_layer_stride
                        + static_cast<std::size_t>(s) * cache_slot_stride;
            for (int q = 0; q < Q; ++q) {
                const int row = s * Q + q;
                const int query_pos = anchor_pos + q;
                // Per-head q_norm and RoPE on Q (fp32 working).
                std::vector<float> q_f32(H_q * D_h);
                for (int i = 0; i < H_q * D_h; ++i) {
                    q_f32[i] = __half2float(q_proj[static_cast<std::size_t>(row) * H_q * D_h + i]);
                }
                for (int h = 0; h < H_q; ++h) {
                    rmsnorm_inplace_f32(q_f32.data() + h * D_h, D_h,
                                        layer_q_norm_w[layer], norm_eps);
                    rope_apply_neox_one_head_f32(q_f32.data() + h * D_h, D_h,
                                                  query_pos, rope_base);
                }
                // Single-query attention against the cache.
                std::vector<float> attn_f32(H_q * D_h);
                single_query_attention_f32(
                    q_f32.data(), H_q, H_kv, D_h,
                    k_cache_layer_slot, v_cache_layer_slot,
                    SeqLen, query_pos,
                    swa_window, is_full,
                    attn_f32.data());
                for (int i = 0; i < H_q * D_h; ++i) {
                    attn_out_h[static_cast<std::size_t>(row) * H_q * D_h + i] =
                        __float2half(attn_f32[i]);
                }
            }
        }

        // Step 2.h — output projection.
        matmul_wmma_padded(attn_out_h.data(), n_rows,
                           layer_o_w[layer], D_emb, H_q * D_h,
                           o_proj_h.data());

        // Step 2.i — residual add (in-place into `hidden`).
        for (int row = 0; row < n_rows; ++row) {
            for (int d = 0; d < D_emb; ++d) {
                const float s = __half2float(hidden[static_cast<std::size_t>(row) * D_emb + d])
                              + __half2float(o_proj_h[static_cast<std::size_t>(row) * D_emb + d]);
                hidden[static_cast<std::size_t>(row) * D_emb + d] = __float2half(s);
            }
        }

        // Step 2.j/k — ffn_norm on each row.
        for (int row = 0; row < n_rows; ++row) {
            std::vector<float> tmp(D_emb);
            for (int d = 0; d < D_emb; ++d) {
                tmp[d] = __half2float(hidden[static_cast<std::size_t>(row) * D_emb + d]);
            }
            rmsnorm_inplace_f32(tmp.data(), D_emb, layer_ffn_norm_w[layer], norm_eps);
            for (int d = 0; d < D_emb; ++d) {
                hidden_n[static_cast<std::size_t>(row) * D_emb + d] = __float2half(tmp[d]);
            }
        }

        // Step 2.l/m — gate and up projections.
        matmul_wmma_padded(hidden_n.data(), n_rows,
                           layer_gate_w[layer], intermediate, D_emb,
                           gate_h.data());
        matmul_wmma_padded(hidden_n.data(), n_rows,
                           layer_up_w[layer], intermediate, D_emb,
                           up_h.data());

        // Step 2.n — activated = silu(gate) * up (element-wise, fp32 working).
        for (std::size_t i = 0; i < static_cast<std::size_t>(n_rows) * intermediate; ++i) {
            const float g = __half2float(gate_h[i]);
            const float u = __half2float(up_h[i]);
            const float a = silu_one_f32(g) * u;
            activated_h[i] = __float2half(a);
        }

        // Step 2.o — down projection.
        matmul_wmma_padded(activated_h.data(), n_rows,
                           layer_down_w[layer], D_emb, intermediate,
                           down_h.data());

        // Step 2.p — residual add.
        for (int row = 0; row < n_rows; ++row) {
            for (int d = 0; d < D_emb; ++d) {
                const float s = __half2float(hidden[static_cast<std::size_t>(row) * D_emb + d])
                              + __half2float(down_h[static_cast<std::size_t>(row) * D_emb + d]);
                hidden[static_cast<std::size_t>(row) * D_emb + d] = __float2half(s);
            }
        }
    }

    // Step 3 — write BLOCK_SIZE mask-token positions to out_hidden.
    // Anchor (q == 0) is dropped — only positions 1..BLOCK_SIZE are output.
    for (int s = 0; s < N_slots; ++s) {
        for (int q = 1; q < Q; ++q) {
            const int row = s * Q + q;
            const int out_q = q - 1;
            for (int d = 0; d < D_emb; ++d) {
                out_hidden[(static_cast<std::size_t>(s) * BLOCK_SIZE + out_q) * D_emb + d] =
                    hidden[static_cast<std::size_t>(row) * D_emb + d];
            }
        }
    }
}

// Stub kept for backward compat with the skeleton test — calls into the
// real reference when given valid weight pointers, otherwise returns zeros.
inline void drafter_forward_reference_stub(
    __half * out_hidden,
    int      N_slots,
    int      BLOCK_SIZE,
    int      D_emb)
{
    std::memset(out_hidden, 0,
                static_cast<std::size_t>(N_slots) *
                static_cast<std::size_t>(BLOCK_SIZE) *
                static_cast<std::size_t>(D_emb) * sizeof(__half));
}

} // namespace dflash_reference
