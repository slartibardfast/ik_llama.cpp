// dflash-inject-kv.cuh
//
// Launcher declaration for the fused dflash_inject_kv_fused kernel.
// Implementation: dflash-inject-kv.cu.
// Spec: specs/dflash/kernel-design.md §6.2.
//
// Allium witnesses: see specs/dflash/allium-tla-binding.json
// (bindings_external for PerLayerArity, HeadShapeMatchesDraft,
// KAsymmetricallyNormedVNot, InjectedAnchorAlignment,
// InjectPerLayerLaunches).

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// All pointer arguments are device pointers (cudaMalloc'd by caller).
// Computes, per (slot, anchor), for ONE drafter layer:
//   1. K = k_weight @ context_states          → [H_kv, D]
//   2. V = v_weight @ context_states          → [H_kv, D]    (V is never normed / RoPE'd)
//   3. K = per-head RMSNorm(K, k_norm_weight) → [H_kv, D]
//   4. K = NeoX RoPE(K, anchor_position, rope_base)
//   5. k_cache_layer[slot, anchor_position, h, d] = K[h, d]
//      v_cache_layer[slot, anchor_position, h, d] = V[h, d]
//
// Caller loops L_d times (one launch per drafter layer), advancing the
// k_weight / v_weight / k_norm_weight / cache base pointers each launch.
//
// Locked compile-time shape: H_kv=8, D=128, D_d=5120 (Qwen3.6-27B-DFlash).
// SeqLen is runtime so the cache can be sized per deployment.
void dflash_inject_kv_fused_launch(
    const __half * d_context_states,        // [N_slots, MAL_anchors, D_d]
    const __half * d_k_weight,              // [H_kv*D, D_d]  for this drafter layer
    const __half * d_v_weight,              // [H_kv*D, D_d]  for this drafter layer
    const __half * d_k_norm_weight,         // [D]            for this drafter layer
    float          rope_base,
    float          norm_eps,
    __half       * d_k_cache_layer,         // [N_slots, SeqLen, H_kv, D] for this drafter layer
    __half       * d_v_cache_layer,         // [N_slots, SeqLen, H_kv, D]
    const int    * d_anchor_positions,      // [N_slots, MAL_anchors]
    int            N_slots,
    int            MAL_anchors,
    int            H_kv,
    int            D,
    int            D_d,
    int            SeqLen,
    cudaStream_t   stream
);

#ifdef __cplusplus
}
#endif
