// dflash-combine-features.cu
//
// Anchor-level FC + hidden_norm pipeline. Per drafter cycle this runs ONCE
// over all (slot, anchor) pairs and produces the context_states consumed by
// dflash_inject_kv_fused.
//
// Spec: specs/dflash/kernel-design.md §6.6 + §6.6.A
//
// Dispatch (revised 2026-05-19, spec §6.6.A): batched pinned-HMMA GEMM via
// `dflash_gemm_npc` followed by a thin per-row RMSNorm + hidden_norm sub-
// kernel. The scalar fp32 FC K-loop is retired; cuBLAS HGEMM forbidden.
//
// Layout coincidence (no pack kernel needed):
//   source_hiddens [N_slots, MAL_anchors, L_src=5, D_d=5120] row-major
//   is byte-identical to [M, L_src*D_d=25600] when (slot, anchor) collapses
//   to M = N_slots * MAL_anchors. The channel-wise concat of the 5 source-
//   layer hiddens IS the contiguous K-axis.
//
// Allium bindings (unchanged):
//   @FuseProjectionFcWeight, @FeatureWidthMatchesTarget,
//   @CombineOrderFCThenHiddenNorm, @ContextStatesAnchorLevel

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>

#include "dflash-combine-features.cuh"
#include "dflash-gemm.cuh"

namespace {

constexpr int NORM_THREADS = 256;
constexpr int D_d_LOCKED   = 5120;
constexpr int L_src_LOCKED = 5;
constexpr int K_LOCKED     = L_src_LOCKED * D_d_LOCKED;  // 25600

// Per-row RMSNorm + hidden_norm-weight multiply + F16 store.
// One CTA per row of fc_out_f32 [M, D_d]; threads stride across D_d.
__global__ void combine_features_norm_kernel(
    const float  * __restrict__ fc_out_f32,       // [M, D_d=5120]
    const __half * __restrict__ hidden_norm_w,    // [D_d=5120]
    float                       norm_eps,
    __half       * __restrict__ context_states,   // [M, D_d=5120] F16 (output)
    int                         D_d)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float warp_sums[8];

    const float * row_in  = fc_out_f32     + static_cast<std::size_t>(row) * D_d;
    __half      * row_out = context_states + static_cast<std::size_t>(row) * D_d;

    // Pass 1: sum_sq across D_d in fp32.
    float sum_sq = 0.0f;
    for (int i = tid; i < D_d; i += blockDim.x) {
        const float v = row_in[i];
        sum_sq += v * v;
    }
    // Warp-shuffle reduce within warp.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFFu, sum_sq, offset);
    }
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();
    float total_sq;
    if (warp_id == 0) {
        const int n_warps = blockDim.x >> 5;
        float v = (lane < n_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
        }
        if (lane == 0) warp_sums[0] = v;
    }
    __syncthreads();
    total_sq = warp_sums[0];

    const float rsq = rsqrtf(total_sq / static_cast<float>(D_d) + norm_eps);

    // Pass 2: scale by rsq * hidden_norm_weight, cast to F16, store.
    for (int i = tid; i < D_d; i += blockDim.x) {
        const float v = row_in[i];
        const float w = __half2float(hidden_norm_w[i]);
        row_out[i] = __float2half((v * rsq) * w);
    }
}

} // anonymous namespace

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
    // Hard-validate the locked shape. Cross-shape support is out of scope
    // for the production Qwen3.6-27B-DFlash deployment.
    if (D_d != D_d_LOCKED || L_src != L_src_LOCKED) {
        const std::size_t n_out_bytes =
            static_cast<std::size_t>(N_slots) *
            static_cast<std::size_t>(MAL_anchors) *
            static_cast<std::size_t>(D_d) * sizeof(__half);
        cudaMemsetAsync(d_context_states, 0, n_out_bytes, stream);
        return;
    }

    const int M = N_slots * MAL_anchors;
    if (M <= 0) return;

    // K-divisibility precondition for pinned HMMA (TILE_K=16).
    if (K_LOCKED % 16) {
        std::fprintf(stderr,
            "[dflash_combine_features_launch] pinned-HMMA requires K%%16==0, "
            "got K=%d\n", K_LOCKED);
        const std::size_t n_out_bytes =
            static_cast<std::size_t>(M) * D_d * sizeof(__half);
        cudaMemsetAsync(d_context_states, 0, n_out_bytes, stream);
        return;
    }

    // F32 scratch for pinned GEMM output: [M, D_d=5120].
    float * fc_out_f32 = nullptr;
    const std::size_t fc_out_bytes = static_cast<std::size_t>(M) * D_d * sizeof(float);
    cudaMallocAsync(&fc_out_f32, fc_out_bytes, stream);

    // Pinned HMMA GEMM:
    //   weight = fc_weight [N_cols=D_d, K=25600]
    //   act    = source_hiddens [M, K=25600]   (byte-identical to
    //                                            [N_slots, MAL_anchors, L_src, D_d])
    //   dst    = fc_out_f32 [M, D_d]           (F32)
    dflash_gemm_npc(
        /*weight =*/d_fc_weight,
        /*act    =*/d_source_hiddens,
        /*dst_f32=*/fc_out_f32,
        /*K      =*/K_LOCKED,
        /*N_cols =*/D_d,
        /*n_rows =*/M,
        stream);

    // Per-row RMSNorm + hidden_norm-weight multiply + F16 store.
    const dim3 grid(M);
    const dim3 block(NORM_THREADS);
    combine_features_norm_kernel<<<grid, block, 0, stream>>>(
        fc_out_f32, d_hidden_norm_weight, norm_eps, d_context_states, D_d);

    cudaFreeAsync(fc_out_f32, stream);
}
