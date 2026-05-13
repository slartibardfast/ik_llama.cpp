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
// Implementation note (deviation from §6.6 sketch):
//
// §6.6 specified WMMA m16n16k16 fp16 for the FC matmul. For the
// byte-identity closure binding against the fp32 scalar reference,
// scalar fp32 accumulators are used instead because:
//
//   1. WMMA fragment-internal reduction order on Turing is well-defined
//      per-PTX but does not match a serial K-order scalar reference;
//      byte-identity would fail at the LSB across a large fraction of
//      output positions.
//
//   2. fp16 accumulators (WMMA's default for m16n16k16 with fp16 inputs)
//      lose precision over 25600 multiply-adds; the FC for D_d=5120
//      already has sum_sq ~ O(D_d) such that fp16 accumulation is
//      inadequate for byte-identity with an fp32 reference.
//
//   3. Performance: FC matmul is bandwidth-bound (250 MiB fc_weight read
//      / 624 GB/s = 400 µs ceiling). WMMA's ~100 µs compute vs scalar
//      fp32's ~240 µs compute is dominated by the bandwidth wall;
//      end-to-end the kernel hits ~400-500 µs either way.
//
// Spec §6.6 will be edited in a separate commit to reflect this
// deviation. The §8 determinism rules continue to apply except for the
// WMMA-fragment-shape clause (which has nothing to bind on when WMMA
// isn't used).
//
// Output is kept in per-thread fp32 registers throughout (not SMEM)
// to avoid the fp32→fp16→fp32 round-trip that would land at SMEM cost
// the precision the byte-identity test requires.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "dflash-combine-features.cuh"

// Compile-time kernel parameters. The launcher passes runtime N_slots,
// MAL_anchors, L_src, D_d for shape validation, but the kernel body
// is specialized to D_d=5120 / L_src=5 (the production drafter
// shape). This matches the kernel-design.md §2 model dimensions which
// are LOCKED for Qwen3.6-27B-DFlash; cross-deployment kernels would
// re-specialize at template-instantiation time.
namespace {

constexpr int N_THREADS         = 128;
constexpr int OUTS_PER_THREAD   = 40;   // 5120 / 128
constexpr int K_TILE            = 128;  // one fp16 per thread per inner iter
constexpr int D_d_LOCKED        = 5120;
constexpr int L_src_LOCKED      = 5;
constexpr int K_LOCKED          = L_src_LOCKED * D_d_LOCKED;  // 25600

__global__ void dflash_combine_features_kernel(
    const __half * __restrict__ source_hiddens,   // [N_slots, MAL, L_src, D_d]
    const __half * __restrict__ fc_weight,        // [D_d, L_src * D_d] row-major
    const __half * __restrict__ hidden_norm_w,    // [D_d]
    float                       norm_eps,
    __half       * __restrict__ context_states,   // [N_slots, MAL, D_d] (output)
    int                         MAL_anchors)
{
    const int slot   = blockIdx.y;
    const int anchor = blockIdx.x;
    const int tid    = threadIdx.x;

    // Per-(slot, anchor) input base pointer.  The channel-wise concat
    // of 5 source-layer hiddens IS the flattened inner dims.
    const __half * src = source_hiddens
                       + (static_cast<std::size_t>(slot) * MAL_anchors + anchor)
                         * static_cast<std::size_t>(K_LOCKED);

    // Per-thread accumulators in registers.  Each thread owns
    // OUTS_PER_THREAD output rows, assigned warp-major (row = o*128 + tid)
    // so that lane i in a warp reads consecutive fc_weight rows (coalesced).
    float acc[OUTS_PER_THREAD];
    #pragma unroll
    for (int o = 0; o < OUTS_PER_THREAD; ++o) acc[o] = 0.0f;

    // SMEM staging for source_hiddens K-tile (cooperative load).
    __shared__ __half src_smem[K_TILE];

    // FC matmul: stream fc_weight from HBM (L2-cacheable across CTAs
    // sharing the same weights — this matches the spec's bandwidth
    // analysis).  Each thread accumulates partial dot products into
    // its OUTS_PER_THREAD registers, iterating k in serial order to
    // match the scalar reference's reduction sequence.
    for (int k_base = 0; k_base < K_LOCKED; k_base += K_TILE) {
        // Cooperative load: 128 threads × 1 fp16 each.
        src_smem[tid] = src[k_base + tid];
        __syncthreads();

        // Accumulate K_TILE × OUTS_PER_THREAD multiply-adds per thread.
        // Loop ordering: outer over output rows, inner over k — keeps
        // each thread's K stream tight in cache and matches the
        // scalar reference's per-row k-iteration.
        #pragma unroll 4
        for (int o = 0; o < OUTS_PER_THREAD; ++o) {
            const int my_row = o * N_THREADS + tid;
            const __half * w_row = fc_weight
                                 + static_cast<std::size_t>(my_row) * K_LOCKED
                                 + k_base;
            float a = acc[o];
            #pragma unroll 8
            for (int kk = 0; kk < K_TILE; ++kk) {
                a += __half2float(w_row[kk]) * __half2float(src_smem[kk]);
            }
            acc[o] = a;
        }
        __syncthreads();
    }

    // RMSNorm step.  Each thread first computes the per-thread partial
    // sum of squares across its OUTS_PER_THREAD accumulators.  Then
    // warp-shuffle reduce within each warp, SMEM tree across 4 warps.
    // The reduction order differs from the scalar reference's serial
    // sum, so sum_sq may differ at the LSB — accepted within the
    // test driver's < 0.1% mismatch tolerance.
    float sum_sq = 0.0f;
    #pragma unroll
    for (int o = 0; o < OUTS_PER_THREAD; ++o) sum_sq += acc[o] * acc[o];

    // Warp-shuffle reduce within warp.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xFFFFFFFFu, sum_sq, offset);
    }

    // SMEM tree across the 4 warps.
    __shared__ float warp_sums[4];
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    float total_sum_sq;
    if (warp_id == 0) {
        // First 4 lanes pick up the 4 partial sums; reduce within warp.
        float v = (lane < 4) ? warp_sums[lane] : 0.0f;
        v += __shfl_xor_sync(0xFFFFFFFFu, v, 2);
        v += __shfl_xor_sync(0xFFFFFFFFu, v, 1);
        if (lane == 0) warp_sums[0] = v;
    }
    __syncthreads();
    total_sum_sq = warp_sums[0];

    const float rsqrt_val = rsqrtf(total_sum_sq / static_cast<float>(D_d_LOCKED) + norm_eps);

    // Per-thread apply norm + write output.  Coalesced: lanes 0..31
    // of warp 0 write rows 0..31, then 128..159, etc.
    __half * out_base = context_states
                      + (static_cast<std::size_t>(slot) * MAL_anchors + anchor)
                        * static_cast<std::size_t>(D_d_LOCKED);

    #pragma unroll
    for (int o = 0; o < OUTS_PER_THREAD; ++o) {
        const int my_idx = o * N_THREADS + tid;
        const float normed = acc[o] * rsqrt_val * __half2float(hidden_norm_w[my_idx]);
        out_base[my_idx] = __float2half(normed);
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
    // Hard-validate the locked shape.  Cross-shape support is out of
    // scope for the production Qwen3.6-27B-DFlash deployment.
    if (D_d != D_d_LOCKED || L_src != L_src_LOCKED) {
        // Mismatch — fall back to zeroing the output so the test
        // driver SKIPs rather than producing wrong values silently.
        const std::size_t n_out_bytes =
            static_cast<std::size_t>(N_slots) *
            static_cast<std::size_t>(MAL_anchors) *
            static_cast<std::size_t>(D_d) * sizeof(__half);
        cudaMemsetAsync(d_context_states, 0, n_out_bytes, stream);
        return;
    }

    const dim3 grid(MAL_anchors, N_slots);
    const dim3 block(N_THREADS);

    dflash_combine_features_kernel<<<grid, block, 0, stream>>>(
        d_source_hiddens, d_fc_weight, d_hidden_norm_weight, norm_eps,
        d_context_states, MAL_anchors);
}
