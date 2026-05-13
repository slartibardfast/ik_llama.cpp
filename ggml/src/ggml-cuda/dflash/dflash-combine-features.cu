// dflash-combine-features.cu
//
// Fused anchor-level FC + hidden_norm kernel. Per drafter cycle this
// runs ONCE per anchor batch and produces the context_states consumed
// by dflash_inject_kv_fused.
//
// Spec:
//   - specs/dflash/kernel-design.md §6.6 (kernel contract)
//   - specs/dflash/dflash.allium @FuseProjectionFcWeight,
//                                @FeatureWidthMatchesTarget,
//                                @CombineOrderFCThenHiddenNorm,
//                                @ContextStatesAnchorLevel
//
// RED-first state (this commit): the launcher zero-initializes the
// output and returns. The unit test detects all-zero output and exits
// 77 (SKIP). The next commit (T3 step "combine kernel: FC + hidden_norm")
// fills in the WMMA GEMM and RMSNorm body.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "dflash-combine-features.cuh"

extern "C" void dflash_combine_features_launch(
    const __half * d_source_hiddens,
    const __half * d_fc_weight,
    const __half * d_hidden_norm_weight,
    float          norm_eps,
    __half       * d_context_states,
    int            N_slots,
    int            MAL_anchors,
    int            L_src,
    int            D_d,
    cudaStream_t   stream)
{
    // Suppress unused-parameter warnings — this is a deliberate stub.
    (void) d_source_hiddens;
    (void) d_fc_weight;
    (void) d_hidden_norm_weight;
    (void) norm_eps;
    (void) L_src;

    // Stub: zero the output buffer so the test driver detects "not
    // implemented" and SKIPs rather than reporting a noisy FAIL on
    // uninitialized memory.
    const std::size_t n_out_bytes =
        static_cast<std::size_t>(N_slots) *
        static_cast<std::size_t>(MAL_anchors) *
        static_cast<std::size_t>(D_d) * sizeof(__half);
    cudaMemsetAsync(d_context_states, 0, n_out_bytes, stream);
}
