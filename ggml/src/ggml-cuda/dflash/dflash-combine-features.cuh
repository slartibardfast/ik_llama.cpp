// dflash-combine-features.cuh
//
// Launcher declaration for the fused dflash_combine_features kernel.
// Implementation: dflash-combine-features.cu.
// Spec: specs/dflash/kernel-design.md §6.6.
//
// Allium witnesses: see specs/dflash/allium-tla-binding.json
// (bindings_external for FuseProjectionFcWeight,
// FeatureWidthMatchesTarget, CombineOrderFCThenHiddenNorm,
// ContextStatesAnchorLevel).

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// All pointer arguments are device pointers (cudaMalloc'd by caller).
// Computes, per (slot, anchor):
//   1. context = fc_weight @ flatten(source_hiddens[slot, anchor, :, :])
//   2. context = RMSNorm(context, hidden_norm_weight, norm_eps)
//   3. context_states[slot, anchor, :] = context
void dflash_combine_features_launch(
    const __half * d_source_hiddens,        // [N_slots, MAL_anchors, L_src, D_d]
    const __half * d_fc_weight,             // [D_d, L_src * D_d] row-major
    const __half * d_hidden_norm_weight,    // [D_d]
    float          norm_eps,
    __half       * d_context_states,        // [N_slots, MAL_anchors, D_d] output
    int            N_slots,
    int            MAL_anchors,
    int            L_src,
    int            D_d,
    cudaStream_t   stream
);

#ifdef __cplusplus
}
#endif
