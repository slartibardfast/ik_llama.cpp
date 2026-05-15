// test-mmq-q4-0-ar16-dp4a
//
// Phase A.4 binding (PHASE_MMQ_Q4_0_AR16.md §11): verify
// vec_dot_q4_0_ar16_q8_1_dp4a computes a Q4_0_AR16 × Q8_1 dot product
// matching a scalar fp32 reference (cos ≥ 0.9999, NMSE ≤ 1e-4).
//
// Strategy:
//   1. Generate random F32 weights W [mmq_y, K] and quantize → Q4_0_AR16.
//   2. Generate random F32 activations A [K, mmq_x] and quantize → Q8_1
//      mmq-layout (block_q8_1_mmq) using a CPU emulator of quantize_mmq_q8_1.
//   3. Launch a wrapper kernel that performs ONE MMQ-tile worth of compute
//      (load_tiles_q4_0_ar16 + two vec_dot calls) and writes per-(i,j) sums.
//   4. CPU scalar reference: dequantize both sides + serial fp32 dot.
//   5. Compute cosine similarity + NMSE per output column; aggregate.
//
// Tile shape used here: mmq_x=8, mmq_y=64, nwarps=4. K=256 (one MMQ_ITER_K).

#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_CHECK_E(stmt) do { \
    cudaError_t _e = (stmt); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        std::exit(1); \
    } \
} while (0)

// ----- Configuration -----
constexpr int K              = 256;          // one MMQ_ITER_K worth
constexpr int N_AR16_BLOCKS  = K / QK_AR16;  // 16 blocks per row
constexpr int N_Q81_BLOCKS   = K / (4*QK8_1); // K / 128 = 2 block_q8_1_mmq per col

// ----- Q4_0_AR16 quantizer (CPU) -----
static void quantize_row_q4_0_ar16_cpu(const float * x, block_q4_0_ar16 * y, int k) {
    const int nb = k / QK_AR16;
    for (int b = 0; b < nb; ++b) {
        // Find absmax over 16 values.
        float amax = 0.0f;
        for (int j = 0; j < QK_AR16; ++j) {
            const float v = std::fabs(x[b*QK_AR16 + j]);
            if (v > amax) amax = v;
        }
        const float d = amax / -8.0f;   // signed 4-bit, [-8..7]
        const float id = d ? 1.0f/d : 0.0f;
        y[b].d = __float2half(d);
        // Per-K convention: K=2j stored in qs[j].low, K=2j+1 in qs[j].high.
        for (int j = 0; j < QK_AR16/2; ++j) {
            const float v0 = x[b*QK_AR16 + 2*j + 0] * id;
            const float v1 = x[b*QK_AR16 + 2*j + 1] * id;
            int q0 = (int)std::roundf(v0) + 8;   // [-8..7] → [0..15]
            int q1 = (int)std::roundf(v1) + 8;
            if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
            if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
            y[b].qs[j] = (uint8_t)((q1 << 4) | q0);
        }
    }
}

static void dequantize_row_q4_0_ar16_cpu(const block_q4_0_ar16 * x, float * y, int k) {
    const int nb = k / QK_AR16;
    for (int b = 0; b < nb; ++b) {
        const float d = __half2float(x[b].d);
        for (int j = 0; j < QK_AR16/2; ++j) {
            const int q0 = (int)(x[b].qs[j] & 0x0F) - 8;
            const int q1 = (int)((x[b].qs[j] >> 4) & 0x0F) - 8;
            y[b*QK_AR16 + 2*j + 0] = d * (float)q0;
            y[b*QK_AR16 + 2*j + 1] = d * (float)q1;
        }
    }
}

// ----- Q8_1 mmq quantizer (CPU emulation of quantize_mmq_q8_1 with DS4 layout) -----
//
// Layout: for `mmq_x` columns × K rows of activations, we emit
// `(K / (4*QK8_1)) * mmq_x` block_q8_1_mmq structs. Each block_q8_1_mmq
// holds 4*QK8_1 = 128 K positions for ONE column.
//
// The dispatcher accesses y as `y + jt*(mmq_x*sizeof/sizeof(int))` for the j-tile
// base, then per-block within tile. So per-row layout (in the dispatcher's view):
// row `r` (one mmq_x-tile column) = `r * mmq_x` block_q8_1_mmq's, each covering
// 128 K's. For a single tile and K=256 we have 2 block_q8_1_mmq's per column ×
// mmq_x columns = 2*mmq_x block_q8_1_mmq's. Tile-y stride11 = mmq_x.
//
// d and s (per 32-K Q8_1 sub-block): standard ds-pair stored in `ds4[k_sub]`.
static void quantize_col_q8_1_mmq_cpu(const float * a_col, block_q8_1_mmq * y_col, int K_local) {
    // a_col: K_local F32 values for ONE activation column.
    // y_col: K_local/128 block_q8_1_mmq's.
    const int n_q81_blocks = K_local / (4*QK8_1);
    for (int b = 0; b < n_q81_blocks; ++b) {
        // 4 sub-blocks of 32 K's each.
        for (int s = 0; s < 4; ++s) {
            float amax = 0.0f;
            float sum = 0.0f;
            for (int j = 0; j < QK8_1; ++j) {
                const float v = a_col[b*128 + s*32 + j];
                const float av = std::fabs(v);
                if (av > amax) amax = av;
            }
            const float d = amax / 127.0f;
            const float id = d ? 1.0f/d : 0.0f;
            for (int j = 0; j < QK8_1; ++j) {
                const float v = a_col[b*128 + s*32 + j];
                int q = (int)std::roundf(v * id);
                if (q < -128) q = -128; if (q > 127) q = 127;
                y_col[b].qs[s*32 + j] = (int8_t)q;
                sum += (float)q;
            }
            // ds4[s] holds (d, s_partial). Store s as quantized-sum * d (matches GPU path).
            y_col[b].ds4[s] = __floats2half2_rn(d, sum*d);
        }
    }
}

// ----- Q8_1 dequant for the CPU oracle -----
static void dequant_col_q8_1_mmq_cpu(const block_q8_1_mmq * y_col, float * out, int K_local) {
    const int n_q81_blocks = K_local / (4*QK8_1);
    for (int b = 0; b < n_q81_blocks; ++b) {
        for (int s = 0; s < 4; ++s) {
            const float2 ds = __half22float2(y_col[b].ds4[s]);
            const float d  = ds.x;
            for (int j = 0; j < QK8_1; ++j) {
                out[b*128 + s*32 + j] = d * (float)y_col[b].qs[s*32 + j];
            }
        }
    }
}

// ----- The wrapper kernel: one MMQ-tile worth of compute. -----
template <int MMQ_X_T, int MMQ_Y_T, int NWARPS_T>
__global__ void test_vec_dot_q4_0_ar16_dp4a_kernel(
        const char * __restrict__ x_weights,
        int                       stride01_bytes,
        const block_q8_1_mmq *    y_in,
        int                       stride11_ints,
        float *                   dst /* [MMQ_Y, MMQ_X] */) {

    extern __shared__ int test_mmq_smem[];
    int * tile_y = (int *) test_mmq_smem;
    int * tile_x = tile_y + GGML_PAD(MMQ_X_T*(WARP_SIZE + WARP_SIZE/QI8_1), NWARPS_T*WARP_SIZE);

    constexpr int qk = QK_AR16;  // 16
    constexpr int y_block_ints = sizeof(block_q8_1_mmq) / sizeof(int);  // 36

    constexpr int blocks_per_iter = MMQ_ITER_K / qk;  // 16 AR16 blocks per kb0 iter

    float sum[MMQ_X_T*MMQ_Y_T / (NWARPS_T*WARP_SIZE)] = {0.0f};

    // Run one kb0 iter (kb0=0): covers 256 K positions.
    // 1. Load weight tile (16 AR16 blocks per row × MMQ_Y rows).
    load_tiles_q4_0_ar16<MMQ_Y_T, NWARPS_T, /*need_check=*/false>(
        x_weights, tile_x, /*kbx0=*/0, /*i_max=*/MMQ_Y_T-1, stride01_bytes);

    // 2. Load tile_y first block_q8_1_mmq per column (covering K=0..127 of all MMQ_X_T cols).
    //    Mirror dispatcher: by0 = y + stride11 * (kb_iter*y_ints_per_kb0 + 0*y_block_ints).
    //    With kb_iter=0 and stride11=mmq_x: by0 = y for the first stripe, y + mmq_x*y_block_ints for the second.
    const int * by0_a = (const int *) y_in;
    #pragma unroll
    for (int l0 = 0; l0 < MMQ_X_T*MMQ_TILE_Y_K; l0 += NWARPS_T*WARP_SIZE) {
        int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;
        tile_y[l] = by0_a[l];
    }
    __syncthreads();
    vec_dot_q4_0_ar16_q8_1_dp4a<MMQ_X_T, MMQ_Y_T, NWARPS_T>(tile_x, tile_y, sum, /*k00=*/0);
    __syncthreads();

    // 3. Load tile_y second block_q8_1_mmq per column (covering K=128..255 of all MMQ_X_T cols).
    const int * by0_b = (const int *) y_in + stride11_ints * y_block_ints;
    #pragma unroll
    for (int l0 = 0; l0 < MMQ_X_T*MMQ_TILE_Y_K; l0 += NWARPS_T*WARP_SIZE) {
        int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;
        tile_y[l] = by0_b[l];
    }
    __syncthreads();
    vec_dot_q4_0_ar16_q8_1_dp4a<MMQ_X_T, MMQ_Y_T, NWARPS_T>(tile_x, tile_y, sum, /*k00=*/WARP_SIZE);
    __syncthreads();

    // Write per-thread sum slots into dst.
    constexpr int slots_i = MMQ_Y_T / WARP_SIZE;          // i-tiles per warp
    constexpr int slots_j = MMQ_X_T / NWARPS_T;           // j-tiles per warp
    #pragma unroll
    for (int j_tile = 0; j_tile < slots_j; ++j_tile) {
        #pragma unroll
        for (int i_tile = 0; i_tile < slots_i; ++i_tile) {
            const int slot = j_tile*slots_i + i_tile;
            const int i = i_tile*WARP_SIZE + threadIdx.x;
            const int j = j_tile*NWARPS_T  + threadIdx.y;
            if (i < MMQ_Y_T && j < MMQ_X_T) {
                dst[i*MMQ_X_T + j] = sum[slot];
            }
        }
    }

    (void)stride11_ints;
}

template <int MMQ_X, int MMQ_Y, int NWARPS>
static bool run_config(std::mt19937 & rng) {
    std::normal_distribution<float> rand_n(0.0f, 0.3f);

    // ---------- Generate weights and activations ----------
    std::vector<float> W(MMQ_Y * K);
    for (auto & v : W) v = rand_n(rng);
    std::vector<float> A(K * MMQ_X);
    for (auto & v : A) v = rand_n(rng);

    // ---------- Quantize weights row-by-row to Q4_0_AR16 ----------
    const int row_bytes_w = N_AR16_BLOCKS * (int)sizeof(block_q4_0_ar16);
    std::vector<uint8_t> Wq_bytes((size_t)MMQ_Y * row_bytes_w);
    for (int i = 0; i < MMQ_Y; ++i) {
        block_q4_0_ar16 * row = (block_q4_0_ar16 *)(Wq_bytes.data() + (size_t)i * row_bytes_w);
        quantize_row_q4_0_ar16_cpu(&W[i*K], row, K);
    }

    // ---------- Quantize activations col-by-col to Q8_1 mmq layout ----------
    std::vector<block_q8_1_mmq> Aq(N_Q81_BLOCKS * MMQ_X);
    std::vector<block_q8_1_mmq> col_tmp(N_Q81_BLOCKS);
    std::vector<float> col_a(K);
    for (int c = 0; c < MMQ_X; ++c) {
        for (int kk = 0; kk < K; ++kk) col_a[kk] = A[kk*MMQ_X + c];
        quantize_col_q8_1_mmq_cpu(col_a.data(), col_tmp.data(), K);
        for (int b = 0; b < N_Q81_BLOCKS; ++b) {
            std::memcpy(&Aq[b * MMQ_X + c], &col_tmp[b], sizeof(block_q8_1_mmq));
        }
    }

    // ---------- CPU reference: scalar fp32 dot of dequantized values ----------
    std::vector<float> ref(MMQ_Y * MMQ_X, 0.0f);
    {
        std::vector<float> Wdq(K);
        std::vector<float> Adq(K);
        std::vector<block_q8_1_mmq> col_tmp2(N_Q81_BLOCKS);
        for (int i = 0; i < MMQ_Y; ++i) {
            const block_q4_0_ar16 * row = (const block_q4_0_ar16 *)(Wq_bytes.data() + (size_t)i * row_bytes_w);
            dequantize_row_q4_0_ar16_cpu(row, Wdq.data(), K);
            for (int j = 0; j < MMQ_X; ++j) {
                for (int b = 0; b < N_Q81_BLOCKS; ++b)
                    std::memcpy(&col_tmp2[b], &Aq[b*MMQ_X + j], sizeof(block_q8_1_mmq));
                dequant_col_q8_1_mmq_cpu(col_tmp2.data(), Adq.data(), K);
                float s = 0.0f;
                for (int k = 0; k < K; ++k) s += Wdq[k] * Adq[k];
                ref[i*MMQ_X + j] = s;
            }
        }
    }

    // ---------- Device alloc + memcpy ----------
    char * d_W = nullptr;
    CUDA_CHECK_E(cudaMalloc(&d_W, Wq_bytes.size()));
    CUDA_CHECK_E(cudaMemcpy(d_W, Wq_bytes.data(), Wq_bytes.size(), cudaMemcpyHostToDevice));

    block_q8_1_mmq * d_A = nullptr;
    CUDA_CHECK_E(cudaMalloc(&d_A, Aq.size() * sizeof(block_q8_1_mmq)));
    CUDA_CHECK_E(cudaMemcpy(d_A, Aq.data(), Aq.size() * sizeof(block_q8_1_mmq), cudaMemcpyHostToDevice));

    float * d_dst = nullptr;
    CUDA_CHECK_E(cudaMalloc(&d_dst, MMQ_Y * MMQ_X * sizeof(float)));
    CUDA_CHECK_E(cudaMemset(d_dst, 0, MMQ_Y * MMQ_X * sizeof(float)));

    // ---------- Launch ----------
    const size_t tile_y_ints = GGML_PAD(MMQ_X*(WARP_SIZE + WARP_SIZE/QI8_1), NWARPS*WARP_SIZE);
    const size_t tile_x_ints = (size_t)MMQ_Y * MMQ_MMA_TILE_X_K_Q4_0_AR16;
    const size_t smem_bytes = (tile_y_ints + tile_x_ints) * sizeof(int);

    dim3 block(WARP_SIZE, NWARPS, 1);
    dim3 grid(1, 1, 1);
    test_vec_dot_q4_0_ar16_dp4a_kernel<MMQ_X, MMQ_Y, NWARPS><<<grid, block, smem_bytes>>>(
        d_W, row_bytes_w, d_A, MMQ_X, d_dst);
    CUDA_CHECK_E(cudaGetLastError());
    CUDA_CHECK_E(cudaDeviceSynchronize());

    std::vector<float> dst(MMQ_Y * MMQ_X, 0.0f);
    CUDA_CHECK_E(cudaMemcpy(dst.data(), d_dst, MMQ_Y * MMQ_X * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_W);
    cudaFree(d_A);
    cudaFree(d_dst);

    double dot = 0.0, n_d = 0.0, n_r = 0.0, mse = 0.0, sum_sq_r = 0.0;
    for (int i = 0; i < MMQ_Y; ++i) {
        for (int j = 0; j < MMQ_X; ++j) {
            const float a = dst[i*MMQ_X + j];
            const float b = ref[i*MMQ_X + j];
            dot += (double)a * b;
            n_d += (double)a * a;
            n_r += (double)b * b;
            const double e = (double)(a - b);
            mse += e*e;
            sum_sq_r += (double)b * b;
        }
    }
    const double cos_sim = dot / (std::sqrt(n_d) * std::sqrt(n_r) + 1e-30);
    const double nmse    = mse / (sum_sq_r + 1e-30);

    const bool cos_ok  = cos_sim >= 0.9999;
    const bool nmse_ok = nmse    <= 1e-4;

    fprintf(stderr, "  (mmq_x=%2d, mmq_y=%3d, nwarps=%d) K=%d  cos=%.7f  NMSE=%.3e  %s\n",
            MMQ_X, MMQ_Y, NWARPS, K, cos_sim, nmse,
            (cos_ok && nmse_ok) ? "PASS" : "FAIL");

    if (!cos_ok || !nmse_ok) {
        int printed = 0;
        for (int i = 0; i < MMQ_Y && printed < 5; ++i) {
            for (int j = 0; j < MMQ_X && printed < 5; ++j) {
                const float a = dst[i*MMQ_X + j];
                const float b = ref[i*MMQ_X + j];
                if (std::fabs(a - b) > 1e-2f * (std::fabs(b) + 1e-6f)) {
                    fprintf(stderr, "    diff [%2d,%2d]: gpu=%9.4f  ref=%9.4f  delta=%+8.4f\n",
                            i, j, a, b, a-b);
                    ++printed;
                }
            }
        }
        return false;
    }
    return true;
}

int main() {
    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        fprintf(stderr, "no CUDA device available; SKIP\n");
        return 77;
    }
    CUDA_CHECK_E(cudaSetDevice(0));

    std::mt19937 rng(0xDEADBEEFULL);
    bool ok = true;

    // Sweep across (mmq_x, mmq_y, nwarps) configurations consistent with the
    // MMQ kernel templating constraints. NWARPS must divide MMQ_X for full
    // j-tile coverage. WARP_SIZE must divide MMQ_Y for full i-tile coverage.
    ok &= run_config< 4,  32, 4>(rng);
    ok &= run_config< 8,  32, 4>(rng);
    ok &= run_config< 8,  64, 4>(rng);
    ok &= run_config< 8, 128, 4>(rng);
    ok &= run_config<16,  32, 4>(rng);
    ok &= run_config<16,  64, 4>(rng);

    if (ok) {
        fprintf(stderr, "OVERALL: PASS\n");
        return 0;
    } else {
        fprintf(stderr, "OVERALL: FAIL\n");
        return 1;
    }
}
