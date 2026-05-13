// dflash-combine-features-reference.h
//
// Scalar fp32 host reference for dflash_combine_features.
//
// Used as the test oracle for the fused CUDA kernel that will live at
//   ggml/src/ggml-cuda/dflash/dflash-combine-features.cu
//
// Algorithm — mirrors specs/dflash/kernel-design.md §6.6 step-for-step:
//
//   For each (slot, anchor):
//     1. concat: ctx_concat[0..L_src*D_d] = source_hiddens[slot, anchor, *, *]
//        (channel-wise concat is just the flattened inner dims)
//     2. FC matmul (row-major, fp32 accumulator):
//          for i in 0..D_d:
//            out_pre[i] = sum over k=0..L_src*D_d-1
//                          fc_weight[i, k] * ctx_concat[k]
//     3. hidden_norm RMSNorm:
//          sum_sq = sum over i: out_pre[i] * out_pre[i]
//          rsqrt  = 1 / sqrt(sum_sq / D_d + norm_eps)
//          context_states[slot, anchor, i] =
//              (out_pre[i] * rsqrt) * hidden_norm_weight[i]
//
// Deterministic — plain serial loops, no parallel reductions. The test
// driver computes the oracle in fp32, then byte-compares against the
// fused kernel's fp16 output via a tolerance gate (the kernel keeps
// fp32 accumulators internally; the fp32→fp16 cast at write time is
// round-to-nearest-even and matches the oracle's cast in nearly all
// cases — outlier ULP differences are documented at the test level).
//
// Allium witnesses (per specs/dflash/allium-tla-binding.json):
//   - FuseProjectionFcWeight      (the FC step IS this invariant)
//   - FeatureWidthMatchesTarget   (D_d parameter = target hidden_size)
//   - CombineOrderFCThenHiddenNorm  (FC then RMSNorm; reversing the
//                                    order yields visibly different
//                                    output for non-identity
//                                    hidden_norm_weight)
//   - ContextStatesAnchorLevel    (output is [N_slots, MAL, D_d];
//                                  caller never replicates per layer)
//
// Spec: specs/dflash/kernel-design.md §6.6
//       specs/dflash/dflash.allium @FuseProjectionFcWeight,
//                                  @CombineOrderFCThenHiddenNorm,
//                                  @ContextStatesAnchorLevel,
//                                  @FeatureWidthMatchesTarget

#pragma once

#include <cmath>
#include <cstddef>

namespace dflash_reference {

inline void combine_features_scalar_ref_f32(
    const float * source_hiddens,           // [N_slots, MAL_anchors, L_src, D_d]
    const float * fc_weight,                // [D_d, L_src * D_d] row-major
    const float * hidden_norm_weight,       // [D_d]
    float         norm_eps,
    float       * context_states,           // [N_slots, MAL_anchors, D_d] output
    int N_slots,
    int MAL_anchors,
    int L_src,
    int D_d
) {
    const int K = L_src * D_d;        // FC input dim = concat width = 25600
    const float inv_D_d = 1.0f / static_cast<float>(D_d);

    for (int slot = 0; slot < N_slots; ++slot) {
        for (int anchor = 0; anchor < MAL_anchors; ++anchor) {
            const float * ctx_concat = source_hiddens
                                     + (static_cast<std::size_t>(slot) * MAL_anchors + anchor)
                                       * static_cast<std::size_t>(K);
            float * out = context_states
                        + (static_cast<std::size_t>(slot) * MAL_anchors + anchor)
                          * static_cast<std::size_t>(D_d);

            // Step 1 implicit: ctx_concat is the natural flattened view of
            // source_hiddens[slot, anchor, :, :] — the channel-wise concat
            // is just contiguous memory.

            // Step 2: FC matmul, row-major. fp32 accumulator.
            // out_pre[i] = sum_k fc_weight[i, k] * ctx_concat[k]
            // We write into out[] directly and overwrite in-place during
            // step 3.
            for (int i = 0; i < D_d; ++i) {
                float acc = 0.0f;
                const float * w_row = fc_weight + static_cast<std::size_t>(i) * K;
                for (int k = 0; k < K; ++k) {
                    acc += w_row[k] * ctx_concat[k];
                }
                out[i] = acc;
            }

            // Step 3: hidden_norm RMSNorm.
            // sum_sq accumulated separately so step-2 output is fully
            // materialized before reduction (same as kernel ordering).
            float sum_sq = 0.0f;
            for (int i = 0; i < D_d; ++i) {
                sum_sq += out[i] * out[i];
            }
            const float rsqrt = 1.0f / std::sqrt(sum_sq * inv_D_d + norm_eps);
            for (int i = 0; i < D_d; ++i) {
                out[i] = (out[i] * rsqrt) * hidden_norm_weight[i];
            }
        }
    }
}

} // namespace dflash_reference
