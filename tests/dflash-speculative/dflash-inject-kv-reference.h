// dflash-inject-kv-reference.h
//
// Scalar fp32 host reference for dflash_inject_kv_fused.
//
// Used as the test oracle for the fused CUDA kernel that will live at
//   ggml/src/ggml-cuda/dflash/dflash-inject-kv.cu
//
// Algorithm — mirrors specs/dflash/kernel-design.md §6.2 step-for-step.
// Handles ONE drafter layer per call (caller loops L_d times — one
// invocation per drafter layer, per @PerLayerArity).
//
// For each (slot, anchor):
//   1. K_proj — fp32 accumulator GEMV:
//        K[h, d] = sum_k k_weight[h*D + d, k] * context_states[slot, anchor, k]
//        for h in 0..H_kv-1, d in 0..D-1.
//   2. V_proj — same pattern, separate weight:
//        V[h, d] = sum_k v_weight[h*D + d, k] * context_states[slot, anchor, k]
//   3. K_norm — per-head RMSNorm with shared per-layer k_norm_weight:
//        For each head h:
//          sum_sq = sum_d K[h, d]^2
//          rsqrt  = 1 / sqrt(sum_sq / D + norm_eps)
//          K[h, d] = (K[h, d] * rsqrt) * k_norm_weight[d]
//      V is NOT normed (per @KAsymmetricallyNormedVNot).
//   4. RoPE on K only — NeoX-style interleaved pairs at base = rope_base:
//        position = anchor_positions[slot, anchor]
//        For each head h, for each pair (i, i + D/2):
//          theta = position / rope_base^(2 i / D)
//          c, s  = cos(theta), sin(theta)
//          k_lo  = K[h, i]
//          k_hi  = K[h, i + D/2]
//          K[h, i]         = k_lo * c - k_hi * s
//          K[h, i + D/2]   = k_lo * s + k_hi * c
//      V is NOT RoPE'd.
//   5. Cache write:
//        k_cache_layer[slot, position, h, d] = K[h, d]
//        v_cache_layer[slot, position, h, d] = V[h, d]
//      where position = anchor_positions[slot, anchor].
//
// Cache layout: [N_slots, SeqLen, H_kv, D] row-major. Caller pre-allocates
// and is responsible for the per-layer base pointer offset (this oracle
// handles ONE layer; per-layer launches happen in the caller's L_d loop).
//
// Deterministic — plain serial loops, no parallel reductions.
//
// Allium witnesses (per specs/dflash/allium-tla-binding.json):
//   - PerLayerArity              (caller loops L_d times — one launch per layer)
//   - HeadShapeMatchesDraft       (H_kv, D parameters match drafter declared shape)
//   - KAsymmetricallyNormedVNot   (K_norm + RoPE on K; V untouched after V_proj)
//   - InjectedAnchorAlignment     (cache writes at anchor_positions[slot, anchor])
//   - InjectPerLayerLaunches      (caller dispatches one launch per drafter layer)
//
// Spec: specs/dflash/kernel-design.md §6.2
//       specs/dflash/dflash.allium ProjectAndFuse contract

#pragma once

#include <cmath>
#include <cstddef>

namespace dflash_reference {

inline void inject_kv_fused_scalar_ref_f32(
    const float * context_states,      // [N_slots, MAL_anchors, D_d]
    const float * k_weight,            // [H_kv * D, D_d] row-major  (for this layer)
    const float * v_weight,            // [H_kv * D, D_d] row-major  (for this layer)
    const float * k_norm_weight,       // [D]                        (for this layer)
    float         rope_base,           // e.g., 10000000.0
    float         norm_eps,            // 1e-6
    float       * k_cache_layer,       // [N_slots, SeqLen, H_kv, D] (for this layer)
    float       * v_cache_layer,       // [N_slots, SeqLen, H_kv, D] (for this layer)
    const int   * anchor_positions,    // [N_slots, MAL_anchors] — seq_pos per anchor
    int           N_slots,
    int           MAL_anchors,
    int           H_kv,                // = 8  for Qwen3.6-27B-DFlash
    int           D,                   // = 128
    int           D_d,                 // = 5120
    int           SeqLen)              // KV cache capacity per slot
{
    const int   D_kv     = H_kv * D;
    const int   D_half   = D / 2;
    const float inv_D    = 1.0f / static_cast<float>(D);

    // Per-(slot, anchor) scratch — H_kv × D up to 8 × 128 = 1024 floats = 4 KiB.
    // Allocated on the stack inside the loop to avoid carrying state across
    // anchors.
    for (int slot = 0; slot < N_slots; ++slot) {
        for (int anchor = 0; anchor < MAL_anchors; ++anchor) {
            const float * ctx = context_states
                              + (static_cast<std::size_t>(slot) * MAL_anchors + anchor)
                                * static_cast<std::size_t>(D_d);
            const int position = anchor_positions[slot * MAL_anchors + anchor];

            // K[h*D + d] and V[h*D + d] — row-major within each head.
            // Heap-free; sized for the locked drafter shape (8×128 = 1024).
            float K[1024];
            float V[1024];

            // Step 1: K_proj.
            for (int row = 0; row < D_kv; ++row) {
                float acc = 0.0f;
                const float * w_row = k_weight + static_cast<std::size_t>(row) * D_d;
                for (int k = 0; k < D_d; ++k) {
                    acc += w_row[k] * ctx[k];
                }
                K[row] = acc;
            }

            // Step 2: V_proj — same pattern. V is never normed or RoPE'd.
            for (int row = 0; row < D_kv; ++row) {
                float acc = 0.0f;
                const float * w_row = v_weight + static_cast<std::size_t>(row) * D_d;
                for (int k = 0; k < D_d; ++k) {
                    acc += w_row[k] * ctx[k];
                }
                V[row] = acc;
            }

            // Step 3: K_norm — per-head RMSNorm with shared per-layer
            // k_norm_weight. Each head's 128-element vector is normed
            // independently; reduction is within-head only.
            for (int h = 0; h < H_kv; ++h) {
                float * Kh = K + h * D;
                float sum_sq = 0.0f;
                for (int d = 0; d < D; ++d) sum_sq += Kh[d] * Kh[d];
                const float rsqrt = 1.0f / std::sqrt(sum_sq * inv_D + norm_eps);
                for (int d = 0; d < D; ++d) {
                    Kh[d] = (Kh[d] * rsqrt) * k_norm_weight[d];
                }
            }

            // Step 4: RoPE on K only — NeoX interleaved pairs.
            // pair (i, i + D/2):
            //   theta = position / rope_base^(2 i / D)
            //   k_lo  = K[h, i]
            //   k_hi  = K[h, i + D/2]
            //   K[h, i]          = k_lo * cos - k_hi * sin
            //   K[h, i + D/2]    = k_lo * sin + k_hi * cos
            //
            // pow/cos/sin computed in fp64, cast to fp32 at use. This
            // keeps the scalar reference byte-aligned with the CUDA
            // kernel's transcendental path — both call libm/libdevice
            // pow/cos/sin at fp64 (each ≤ 1 fp64 ULP), and the fp32
            // cast absorbs the residual delta. fp32 versions diverge
            // enough (≤ 6 ULP powf, ≤ 2 ULP cosf/sinf) to push final
            // fp16 outputs past the ≤ 2 ULP test gate.
            for (int h = 0; h < H_kv; ++h) {
                float * Kh = K + h * D;
                for (int i = 0; i < D_half; ++i) {
                    const double exp_val_d  =
                        static_cast<double>(2 * i) / static_cast<double>(D);
                    const double inv_freq_d = std::pow(static_cast<double>(rope_base), -exp_val_d);
                    const double theta_d    = static_cast<double>(position) * inv_freq_d;
                    const float c = static_cast<float>(std::cos(theta_d));
                    const float s = static_cast<float>(std::sin(theta_d));
                    const float k_lo = Kh[i];
                    const float k_hi = Kh[i + D_half];
                    Kh[i]            = k_lo * c - k_hi * s;
                    Kh[i + D_half]   = k_lo * s + k_hi * c;
                }
            }

            // Step 5: Write to cache at the anchor position.
            // Layout: k_cache_layer[slot, position, h, d]
            //       = k_cache_layer[((slot * SeqLen + position) * H_kv + h) * D + d]
            const std::size_t base =
                ((static_cast<std::size_t>(slot) * SeqLen + position)
                  * static_cast<std::size_t>(H_kv)) * static_cast<std::size_t>(D);
            for (int row = 0; row < D_kv; ++row) {
                k_cache_layer[base + row] = K[row];
                v_cache_layer[base + row] = V[row];
            }
        }
    }
}

} // namespace dflash_reference
