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
// v2 kernel — one WARP per 32-row tile, lane = row (each lane folds its OWN row
// sequentially, so the byte-identical serial fold runs 32-way parallel — no
// single-lane bottleneck). Weights are staged through SMEM so the GLOBAL reads
// stay coalesced (each row's KTILE contiguous blocks copied by the full warp)
// while the per-row fold reads from SMEM. y-blocks broadcast across the 32 rows.
// Requires nb % KTILE == 0 (true: production K are multiples of 256 => nb%8==0).
// dst(i,j) = p0 + ((p1+p2)+p3), one (float(dot32)*dA)*dB add per 32-block.
// ---------------------------------------------------------------------------
#define GEMV_NWARPS 4
#define GEMV_KTILE  8   // == MMQ_ITER_K/QK4_0 = blocks_per_iter; nb is a multiple

template <int MMAX>
static __global__ void __launch_bounds__(WARP_SIZE*GEMV_NWARPS, 2)
mmq_gemv_faithful_q4_0_kernel(
        const char * __restrict__ x, const char * __restrict__ y, float * __restrict__ dst,
        const int ne00, const int ne01, const int stride01, const int ne11, const int stride11, const int ne0) {

    __shared__ block_q4_0 sw[GEMV_NWARPS][WARP_SIZE][GEMV_KTILE];  // [warp][row_in_tile][block]

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;                                  // 0..31 = row within the 32-row tile
    const int row0 = (blockIdx.x*GEMV_NWARPS + warp)*WARP_SIZE;    // first row of this warp's tile
    const int row  = row0 + lane;                                  // this lane's output row

    const block_q8_1_mmq * Y = (const block_q8_1_mmq *) y;

    const int nb = ne00 / QK4_0;
    const int bpi = GEMV_KTILE;  // blocks_per_iter

    // 4 split-K chunk stops (contiguous; [hi3,nb) dropped).
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

    // 18 bytes/block * KTILE = 144 bytes = 36 ints per row, copied coalesced.
    constexpr int INTS_PER_TILE_ROW = (GEMV_KTILE * sizeof(block_q4_0)) / sizeof(int);  // 36

    for (int kt0 = 0; kt0 < nb; kt0 += GEMV_KTILE) {
        // Coalesced cooperative load: for each of the 32 rows, the full warp
        // copies that row's KTILE contiguous blocks (36 ints) into SMEM.
        #pragma unroll
        for (int r = 0; r < WARP_SIZE; ++r) {
            const int grow = row0 + r;
            if (grow >= ne01) continue;
            const int * gsrc = (const int *)(x + (int64_t)grow*stride01 + (int64_t)kt0*sizeof(block_q4_0));
            int * sdst = (int *) &sw[warp][r][0];
            #pragma unroll
            for (int c = lane; c < INTS_PER_TILE_ROW; c += WARP_SIZE) {
                sdst[c] = gsrc[c];
            }
        }
        __syncwarp();

        // Fold: this lane owns `row`; sum its KTILE blocks into chunk partials.
        if (row < ne01) {
            #pragma unroll
            for (int b = 0; b < GEMV_KTILE; ++b) {
                const int kb = kt0 + b;
                if (kb >= hi[3]) break;                              // dropped tail
                const int s = (kb < hi[0]) ? 0 : (kb < hi[1]) ? 1 : (kb < hi[2]) ? 2 : 3;
                const block_q4_0 * wb = &sw[warp][lane][b];
                const int k128 = kb >> 2, sub = kb & 3;
                #pragma unroll
                for (int j = 0; j < MMAX; ++j) {
                    if (j >= ne11) break;
                    const block_q8_1_mmq * yb = &Y[(int64_t)k128*stride11 + j];
                    const float    dB = __low2float(yb->ds4[sub]);
                    const int8_t * a  = yb->qs + sub*QK8_1;
                    p[j][s] += gemv_block_partial(wb, a) * dB;       // fma(X, dB, p)
                }
            }
        }
        __syncwarp();  // protect SMEM reuse next tile
    }

    if (row < ne01) {
        #pragma unroll
        for (int j = 0; j < MMAX; ++j) {
            if (j >= ne11) break;
            dst[(int64_t)j*ne0 + row] = p[j][0] + ((p[j][1] + p[j][2]) + p[j][3]);
        }
    }
}

void ggml_cuda_mul_mat_q4_0_gemv_faithful(const mmq_args & args, cudaStream_t stream) {
    const int ne01 = (int) args.ne01;  // output rows
    const int ne11 = (int) args.ne11;  // M columns
    const int rows_per_block = GEMV_NWARPS * WARP_SIZE;
    const dim3 block(WARP_SIZE, GEMV_NWARPS, 1);
    const dim3 grid((ne01 + rows_per_block - 1) / rows_per_block, 1, 1);
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
