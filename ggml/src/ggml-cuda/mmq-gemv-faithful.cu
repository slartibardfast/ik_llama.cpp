//
// PHASE_LAUNCH_FUSION_SWEEP #195 — bit-identical read-once GEMV family (Q4_0).
// See mmq-gemv-faithful.cuh for the contract. SPDX-License-Identifier: MIT
//
#include "mmq-gemv-faithful.cuh"
#include "common.cuh"

#include <cstdlib>
#include <type_traits>

// ---------------------------------------------------------------------------
// Runtime toggle + carve-out threshold. -1 == "not yet read from env".
// ---------------------------------------------------------------------------
static int g_m1_gemv_enabled   = -1;
static int g_m1_gemv_threshold = -1;

extern "C" void ggml_cuda_m1_gemv_set_enabled(int enabled) { g_m1_gemv_enabled = enabled ? 1 : 0; }
extern "C" int  ggml_cuda_m1_gemv_enabled(void) {
    if (g_m1_gemv_enabled < 0) {
        g_m1_gemv_enabled = std::getenv("GGML_CUDA_M1_GEMV") != nullptr ? 1 : 0;
    }
    return g_m1_gemv_enabled;
}
extern "C" void ggml_cuda_m1_gemv_set_threshold(int k) { g_m1_gemv_threshold = k; }
extern "C" int  ggml_cuda_m1_gemv_threshold(void) {
    if (g_m1_gemv_threshold < 0) {
        const char * e = std::getenv("GGML_CUDA_M1_GEMV_K");
        g_m1_gemv_threshold = e ? atoi(e) : 8;
    }
    return g_m1_gemv_threshold;
}

// Per-32-block partial product X = float(dot32)*dA, where dot32 is the
// integer-exact int dot over the full 32-block. Returns X (NOT X*dB): the fold
// must do `p += X*dB` so the compiler contracts it to fma(X, dB, p) — exactly
// MMQ's `sum += (C.x*dA)*dB` => fma(C.x*dA, dB, sum). Materializing X*dB here
// and adding it separately rounds *dB before the add and diverges by ~1 ULP.
// Q4_0 unpack is natural k-order (w[0..15]=low nibbles, w[16..31]=high).
static __device__ __forceinline__ float gemv_block_partial(
        const block_q4_0 * __restrict__ wb, const int8_t * __restrict__ a) {
    const float dA = __half2float(wb->d);
    int dot = 0;
    #pragma unroll
    for (int kpos = 0; kpos < QK4_0; ++kpos) {
        const int byte = (kpos < 16) ? wb->qs[kpos] : wb->qs[kpos - 16];
        const int w    = ((kpos < 16) ? (byte & 0x0F) : (byte >> 4)) - 8;
        dot += w * (int)a[kpos];
    }
    return (float)dot * dA;
}

// ---------------------------------------------------------------------------
// v1 kernel — one WARP per output row, MWARP columns looped inside (read-once:
// each lane reads its block's weights once per group and reuses across columns).
// Memory-coalesced: at group g, the 32 lanes read the 32 consecutive blocks
// [32g, 32g+32) of row i (consecutive 18-byte Q4_0 blocks). The float fold is
// done ORDERED on lane 0 — gather each lane's per-block term in ascending block
// order via __shfl and accumulate into the 4 split-K chunk partials — because
// byte-identity forbids a tree warp-reduce. dst = p0+((p1+p2)+p3).
// ---------------------------------------------------------------------------
template <int MMAX>
static __global__ void mmq_gemv_faithful_q4_0_kernel(
        const char * __restrict__ x, const char * __restrict__ y, float * __restrict__ dst,
        const int ne00, const int ne01, const int stride01, const int ne11, const int stride11, const int ne0) {

    const int lane = threadIdx.x;                                // 0..31
    const int i    = blockIdx.x*blockDim.y + threadIdx.y;        // output row (one warp per row)
    if (i >= ne01) {
        return;
    }

    const block_q4_0     * wrow = (const block_q4_0 *)(x + (int64_t)i*stride01);
    const block_q8_1_mmq * Y    = (const block_q8_1_mmq *) y;

    const int nb  = ne00 / QK4_0;        // Q4_0 blocks along K
    const int bpi = MMQ_ITER_K / QK4_0;  // 8 (blocks_per_iter)

    // The 4 split-K chunk stops (lo_{s+1} == hi_s, so chunks are contiguous
    // [0,hi0)[hi0,hi1)[hi1,hi2)[hi2,hi3); blocks [hi3,nb) are dropped).
    int hi[4];
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        int h = (nb * (s + 1)) / 4;  h -= h % bpi;
        if (s == 3) h = nb - (nb % bpi);
        hi[s] = h;
    }

    float p[MMAX][4];
    #pragma unroll
    for (int j = 0; j < MMAX; ++j) {
        #pragma unroll
        for (int s = 0; s < 4; ++s) p[j][s] = 0.0f;
    }

    for (int g0 = 0; g0 < nb; g0 += WARP_SIZE) {
        const int kb    = g0 + lane;
        const bool live = kb < nb;
        const block_q4_0 * wb = live ? &wrow[kb] : wrow;  // safe addr; term gated by `live`
        const int k128 = kb >> 2;
        const int sub  = kb & 3;

        #pragma unroll
        for (int j = 0; j < MMAX; ++j) {
            if (j >= ne11) break;
            const block_q8_1_mmq * yb = live ? &Y[(int64_t)k128*stride11 + j] : Y;
            const float    dB = live ? __low2float(yb->ds4[sub]) : 0.0f;
            const int8_t * a  = yb->qs + sub*QK8_1;
            const float    X  = live ? gemv_block_partial(wb, a) : 0.0f;  // float(dot32)*dA

            // Ordered fold on lane 0: ascending block order, into chunk partials.
            // p += X*dB contracts to fma(X, dB, p) — matches MMQ's reduction.
            #pragma unroll
            for (int t = 0; t < WARP_SIZE; ++t) {
                const float Xt  = __shfl_sync(0xFFFFFFFF, X,  t, WARP_SIZE);
                const float dBt = __shfl_sync(0xFFFFFFFF, dB, t, WARP_SIZE);
                const int   kbt = g0 + t;
                if (lane == 0 && kbt < hi[3]) {
                    const int s = (kbt < hi[0]) ? 0 : (kbt < hi[1]) ? 1 : (kbt < hi[2]) ? 2 : 3;
                    p[j][s] += Xt * dBt;
                }
            }
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < MMAX; ++j) {
            if (j >= ne11) break;
            dst[(int64_t)j*ne0 + i] = p[j][0] + ((p[j][1] + p[j][2]) + p[j][3]);
        }
    }
}

void ggml_cuda_mul_mat_q4_0_gemv_faithful(const mmq_args & args, cudaStream_t stream) {
    const int ne01 = (int) args.ne01;  // output rows
    const int ne11 = (int) args.ne11;  // M columns
    const int nwarps = 8;
    const dim3 block(WARP_SIZE, nwarps, 1);
    const dim3 grid((ne01 + nwarps - 1) / nwarps, 1, 1);
    const int K = (int) args.ne00, s01 = (int) args.stride01, s11 = (int) args.stride11, n0 = (int) args.ne0;
    // Instantiate the smallest MMAX >= ne11 (the carve-out threshold caps ne11).
    auto launch = [&](auto mmax_tag) {
        constexpr int MMAX = decltype(mmax_tag)::value;
        mmq_gemv_faithful_q4_0_kernel<MMAX><<<grid, block, 0, stream>>>(
            args.x, args.y, args.dst, K, ne01, s01, ne11, s11, n0);
    };
    if      (ne11 <= 1) launch(std::integral_constant<int,1>{});
    else if (ne11 <= 2) launch(std::integral_constant<int,2>{});
    else if (ne11 <= 4) launch(std::integral_constant<int,4>{});
    else                launch(std::integral_constant<int,8>{});
}
