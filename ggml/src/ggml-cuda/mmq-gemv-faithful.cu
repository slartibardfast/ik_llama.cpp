//
// PHASE_LAUNCH_FUSION_SWEEP #195 — bit-identical read-once GEMV family (Q4_0).
// See mmq-gemv-faithful.cuh for the contract. SPDX-License-Identifier: MIT
//
#include "mmq-gemv-faithful.cuh"
#include "common.cuh"

#include <cstdlib>

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

// ---------------------------------------------------------------------------
// v0 kernel — one thread per output element (i,j). Correctness-first: the
// float-accumulation order is byte-identical to production MMQ
// (mul_mat_q_split_k<Q4_0,...,split_k=4>), per the derived reduction:
//
//   for s in 0..3 (the 4 split-K chunks, boundaries nb*s/4 aligned down to
//                  blocks_per_iter=8, tail dropped on the last slice):
//     p[s] = sequential ascending fold over [lo,hi):
//            for kb in [lo,hi): for g in 0..3: p[s] += (float(dot8)*dA)*dB
//   dst = p[0] + ((p[1]+p[2])+p[3])           (the split-K fixup combine)
//
// Q4_0 unpack is natural k-order (w[0..15]=low nibbles, w[16..31]=high), so
// MMQ's group-of-8 = natural positions [8g,8g+8). int->float is exact.
// ---------------------------------------------------------------------------
static __global__ void mmq_gemv_faithful_q4_0_kernel(
        const char * __restrict__ x, const char * __restrict__ y, float * __restrict__ dst,
        const int ne00, const int ne01, const int stride01, const int ne11, const int stride11, const int ne0) {

    const int i = blockIdx.x*blockDim.x + threadIdx.x;  // output row
    const int j = blockIdx.y;                            // output column
    if (i >= ne01 || j >= ne11) {
        return;
    }

    const block_q4_0      * wrow = (const block_q4_0 *)(x + (int64_t)i*stride01);
    const block_q8_1_mmq  * Y    = (const block_q8_1_mmq *) y;

    const int nb  = ne00 / QK4_0;        // Q4_0 blocks along K
    const int bpi = MMQ_ITER_K / QK4_0;  // 8 (blocks_per_iter)

    float p[4];
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
        int lo = (nb * s)       / 4;  lo -= lo % bpi;
        int hi = (nb * (s + 1)) / 4;  hi -= hi % bpi;
        if (s == 3) {
            hi = nb - (nb % bpi);     // last slice extends; trailing (nb%bpi) blocks dropped
        }

        float acc = 0.0f;
        for (int kb = lo; kb < hi; ++kb) {
            const block_q4_0 * wb = &wrow[kb];
            const float dA = __half2float(wb->d);

            const int k128 = kb >> 2;    // which 128-value q8_1_mmq block
            const int sub  = kb & 3;     // which 32-value sub-block within it
            const block_q8_1_mmq * yb = &Y[(int64_t)k128*stride11 + j];
            const float dB = __low2float(yb->ds4[sub]);
            const int8_t * a = yb->qs + sub*QK8_1;

            // One float add per 32-block: MMQ's k01 loop steps per-block for
            // Q4_0 (dA,dB are one scale per 32), so the int dot over the full
            // 32-block is multiplied by dA*dB once. dot32 is integer-exact.
            int dot = 0;
            #pragma unroll
            for (int kpos = 0; kpos < QK4_0; ++kpos) {
                const int byte = (kpos < 16) ? wb->qs[kpos] : wb->qs[kpos - 16];
                const int w    = ((kpos < 16) ? (byte & 0x0F) : (byte >> 4)) - 8;
                dot += w * (int)a[kpos];
            }
            acc += ((float)dot * dA) * dB;          // ((int->float)*dA)*dB, matches sum += C.x*dA*dB
        }
        p[s] = acc;
    }

    dst[(int64_t)j*ne0 + i] = p[0] + ((p[1] + p[2]) + p[3]);
}

void ggml_cuda_mul_mat_q4_0_gemv_faithful(const mmq_args & args, cudaStream_t stream) {
    const int ne01 = (int) args.ne01;  // output rows
    const int ne11 = (int) args.ne11;  // M columns
    const int block = 128;
    const dim3 grid((ne01 + block - 1) / block, ne11, 1);
    mmq_gemv_faithful_q4_0_kernel<<<grid, block, 0, stream>>>(
        args.x, args.y, args.dst,
        (int) args.ne00, ne01, (int) args.stride01, ne11, (int) args.stride11, (int) args.ne0);
}
