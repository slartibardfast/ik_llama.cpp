#include "mul-mat-f16-pinned.cuh"
#include <mma.h>

using namespace nvcuda;

// Deterministic F16 weight × F16 activation → F32 dst, using nvcuda::wmma 16x16x16.
//
// Layout & shape:
//   weight ggml [K, N]: data[k + n*K_stride_w]  — N rows × K cols ROW-major (logical)
//   act    ggml [K, M]: data[k + m*K_stride_a]  — K rows × M cols COL-major (logical)
//   dst    ggml [N, M]: data[n + m*N_dst_stride] — N rows × M cols COL-major (logical)
//
// Each CTA = 1 warp = computes 16 output rows × 16 output cols of dst at
// (n0, m0). Loop K in fixed chunks of 16. fp32 accumulator. wmma::mma_sync
// uses HMMA.16816 which has a fixed per-fragment reduction order — so cell
// (n, m) is byte-identical regardless of the total M.
//
// The kernel TILE_M is 16 (set by wmma; differs from the .cuh declared
// TILE_M=8). The host launcher passes M and the kernel masks tail columns.

__launch_bounds__(WARP_SIZE, 2)
static __global__ void mul_mat_f16_pinned_kernel_wmma(
        const half  * __restrict__ weight,
        const half  * __restrict__ act,
        float       * __restrict__ dst,
        const int K,
        const int N_rows,
        const int M,
        const int K_stride_w,
        const int K_stride_a,
        const int N_dst_stride) {

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    const int n_tile = blockIdx.x;
    const int m_tile = blockIdx.y;
    const int n0 = n_tile * WMMA_M;
    const int m0 = m_tile * WMMA_N;

    // SMEM tile for the activation. wmma::load_matrix_sync needs col-major
    // for the B fragment, with the leading dim being the column stride.
    // We stage `act` from global through SMEM so we can zero out the tail
    // columns when M < 16. (Weight A can be loaded directly from global; its
    // strides are well-defined and fully covered since N_rows is always
    // padded to TILE_N by the wmma kernel.)
    __shared__ alignas(16) half B_smem[WMMA_K][WMMA_N];

    wmma::fragment<wmma::matrix_a,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    // B_smem is laid out half[k_row][n_col] (row-major), so we use row_major
    // here. wmma transposes internally for the HMMA operand layout.
    wmma::fragment<wmma::matrix_b,    WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>                 c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const int n_valid = min(WMMA_M, N_rows - n0);
    const int m_valid = min(WMMA_N, M       - m0);
    const int tid     = threadIdx.x;

    // Edge: if N_rows is not a multiple of 16, the kernel still launches
    // a full N-tile at the tail (n_valid < 16). For weight loads beyond
    // n_valid we pad rows with zero via a staging SMEM (else we'd read past
    // the buffer or pull arbitrary memory).
    __shared__ alignas(16) half A_smem[WMMA_M][WMMA_K];

    for (int k0 = 0; k0 < K; k0 += WMMA_K) {
        // ===== Load A (weight tile) into A_smem with row-zero pad for n_valid < WMMA_M.
        // 16x16 halves = 256 halves. 32 threads × 8 halves/thread = 256.
        // Each thread does ONE float4 (8 halves) load aligned.
        {
            const int row = tid / 2;       // 0..15 → n_local
            const int col = (tid % 2) * 8; // 0 or 8 → k_local
            if (row < n_valid) {
                const half * src = weight + (k0 + col) + (size_t)(n0 + row) * K_stride_w;
                float4 v = *reinterpret_cast<const float4 *>(src);
                *reinterpret_cast<float4 *>(&A_smem[row][col]) = v;
            } else {
                float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                *reinterpret_cast<float4 *>(&A_smem[row][col]) = z;
            }
        }

        // ===== Load B (act tile) into B_smem col-major with tail-col zero pad.
        // 16x16 halves = 256. Each thread loads 8 halves along K for fixed M.
        // (32 threads × 8 = 256; K=16 → each thread covers 8 K-rows, 1 m-col.)
        // BUT col-major: B_smem[k][m]. Load along K (inner) for a fixed m.
        // Actually: 16 cols × 16 K rows. Per thread: 1 M col × 8 K halves.
        // 32 threads × 8 halves = 256 halves total. 32 ÷ 16 = 2 → 2 threads per M col.
        // Thread layout: (tid/2) = m, (tid%2)*8 = k_start. Each thread loads 8 halves.
        {
            const int m_local = tid / 2;       // 0..15 → m col
            const int k_start = (tid % 2) * 8; // 0 or 8 → k row start
            // 8 halves along K at fixed m: act[k0+k_start..k0+k_start+7, m0+m_local].
            // Ggml: data[(k0+k_start+i) + (m0+m_local)*K_stride_a] for i=0..7.
            // Stride-1 along K → contiguous in memory → one float4 load.
            if (m_local < m_valid) {
                const half * src = act + (k0 + k_start) + (size_t)(m0 + m_local) * K_stride_a;
                float4 v = *reinterpret_cast<const float4 *>(src);
                // Store col-major: B_smem[k_start..k_start+7][m_local].
                // B_smem row stride = WMMA_N halves = 32 bytes. So per-half write.
                // No vectorized store across non-contiguous rows; loop.
                const half * hv = reinterpret_cast<const half *>(&v);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    B_smem[k_start + i][m_local] = hv[i];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    B_smem[k_start + i][m_local] = __float2half(0.0f);
                }
            }
        }

        __syncwarp();

        // ===== wmma load + mma_sync.
        wmma::load_matrix_sync(a_frag, &A_smem[0][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &B_smem[0][0], WMMA_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncwarp();
    }

    // ===== Store C (16x16 fp32, col-major to dst) with tail mask.
    __shared__ alignas(16) float C_smem[WMMA_M][WMMA_N];
    wmma::store_matrix_sync(&C_smem[0][0], c_frag, WMMA_N, wmma::mem_row_major);

    __syncwarp();

    // 16x16 = 256 floats. 32 threads × 8 = 256. Each thread does 1 float4 store.
    // But we need to mask out-of-range (n, m). Simpler: each thread writes 8 cells
    // and checks each.
    #pragma unroll
    for (int li = 0; li < 8; ++li) {
        const int idx = tid * 8 + li;   // 0..255
        const int i   = idx / WMMA_N;   // 0..15 → n_local
        const int j   = idx % WMMA_N;   // 0..15 → m_local
        const int n   = n0 + i;
        const int m   = m0 + j;
        if (n < N_rows && m < M) {
            dst[n + (size_t)m * N_dst_stride] = C_smem[i][j];
        }
    }
}

void ggml_cuda_mul_mat_f16_pinned(
        const half  * weight,
        const half  * act,
        float       * dst,
        int           K,
        int           N_rows,
        int           M,
        int           K_stride_w,
        int           K_stride_a,
        int           N_dst_stride,
        cudaStream_t  stream) {

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    GGML_ASSERT(K % WMMA_K == 0 && "K must be a multiple of 16");

    const int n_tiles = (N_rows + WMMA_M - 1) / WMMA_M;
    const int m_tiles = (M       + WMMA_N - 1) / WMMA_N;

    const dim3 grid(n_tiles, m_tiles, 1);
    const dim3 block(WARP_SIZE, 1, 1);

    mul_mat_f16_pinned_kernel_wmma<<<grid, block, 0, stream>>>(
        weight, act, dst, K, N_rows, M,
        K_stride_w, K_stride_a, N_dst_stride);
}

// ===== F32 path =====
//
// Each CTA = 1 warp, computes ONE output cell dst[n, m].
// K loop strided by 32, per-thread fp32 accumulator, warp-reduce in
// fixed tree order. K-loop order is independent of M → byte-identity
// across batch.

__launch_bounds__(WARP_SIZE, 4)
static __global__ void mul_mat_f32_pinned_kernel(
        const float * __restrict__ weight,
        const float * __restrict__ act,
        float       * __restrict__ dst,
        const int K,
        const int N_rows,
        const int M,
        const int K_stride_w,
        const int K_stride_a,
        const int N_dst_stride) {

    const int n = blockIdx.x;
    const int m = blockIdx.y;
    if (n >= N_rows || m >= M) return;

    const int tid = threadIdx.x;
    float acc = 0.0f;

    // K loop in fixed stride-32 order. Per-thread accumulator covers
    // K positions {tid, tid+32, tid+64, ...}. Order is identical for any
    // (n, m, total_M) — only the underlying values vary.
    const float * w_row = weight + (size_t)n * K_stride_w;
    const float * a_col = act    + (size_t)m * K_stride_a;
    for (int k = tid; k < K; k += WARP_SIZE) {
        acc += w_row[k] * a_col[k];
    }

    // Warp reduce in fixed butterfly order (16, 8, 4, 2, 1).
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
    }

    if (tid == 0) {
        dst[(size_t)n + (size_t)m * N_dst_stride] = acc;
    }
}

void ggml_cuda_mul_mat_f32_pinned(
        const float * weight,
        const float * act,
        float       * dst,
        int           K,
        int           N_rows,
        int           M,
        int           K_stride_w,
        int           K_stride_a,
        int           N_dst_stride,
        cudaStream_t  stream) {

    const dim3 grid(N_rows, M, 1);
    const dim3 block(WARP_SIZE, 1, 1);

    mul_mat_f32_pinned_kernel<<<grid, block, 0, stream>>>(
        weight, act, dst, K, N_rows, M,
        K_stride_w, K_stride_a, N_dst_stride);
}
