// dflash-inject-kv.cu
//
// Per-(slot, anchor) per-drafter-layer fused KV projection:
//   K_proj + V_proj + per-head K_norm + RoPE(K) + cache write.
// V is never normed and never RoPE'd (per @KAsymmetricallyNormedVNot).
//
// Spec:
//   - specs/dflash/kernel-design.md §6.2 (kernel contract)
//   - specs/dflash/dflash.allium @PerLayerArity, @HeadShapeMatchesDraft,
//                                @KAsymmetricallyNormedVNot,
//                                @InjectedAnchorAlignment,
//                                @InjectPerLayerLaunches
//
// RED-first state (this commit): launcher zero-writes both K and V
// caches at the anchor positions. The unit test detects all-zero
// writes at anchor positions and exits 77 (SKIP). Subsequent commits
// fill in the kernel body step by step:
//   - K_proj WMMA GEMV
//   - V_proj WMMA GEMV
//   - K_norm per-head RMSNorm
//   - NeoX RoPE on K
//   - Vectorized half4 cache writes

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "dflash-inject-kv.cuh"

extern "C" void dflash_inject_kv_fused_launch(
    const __half * d_context_states,
    const __half * d_k_weight,
    const __half * d_v_weight,
    const __half * d_k_norm_weight,
    float          rope_base,
    float          norm_eps,
    __half       * d_k_cache_layer,
    __half       * d_v_cache_layer,
    const int    * d_anchor_positions,
    int            N_slots,
    int            MAL_anchors,
    int            H_kv,
    int            D,
    int            D_d,
    int            SeqLen,
    cudaStream_t   stream)
{
    // Suppress unused-parameter warnings — this is a deliberate stub.
    (void) d_context_states;
    (void) d_k_weight;
    (void) d_v_weight;
    (void) d_k_norm_weight;
    (void) rope_base;
    (void) norm_eps;
    (void) d_anchor_positions;
    (void) MAL_anchors;
    (void) H_kv;
    (void) D;
    (void) D_d;

    // Stub: zero the entire K and V cache slabs for this layer so the
    // test driver detects "not implemented" and SKIPs rather than
    // reporting noisy FAILs on uninitialized memory.
    const std::size_t n_cells_bytes =
        static_cast<std::size_t>(N_slots) *
        static_cast<std::size_t>(SeqLen) *
        static_cast<std::size_t>(H_kv) *
        static_cast<std::size_t>(D) * sizeof(__half);
    cudaMemsetAsync(d_k_cache_layer, 0, n_cells_bytes, stream);
    cudaMemsetAsync(d_v_cache_layer, 0, n_cells_bytes, stream);
}
