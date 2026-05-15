// test-mmq-q4-0-ar16-load-tiles
//
// Phase A.3 binding (PHASE_MMQ_Q4_0_AR16.md §11): verify
// load_tiles_q4_0_ar16 produces SMEM contents byte-equivalent to
// dequantize_block_q4_0_ar16 (the validated oracle in convert.cu).
//
// Strategy:
//   1. Allocate random block_q4_0_ar16 source matrix.
//   2. Launch a wrapper __global__ that calls load_tiles_q4_0_ar16 on
//      a global-memory "SMEM-mirror" buffer (lets us avoid SMEM size
//      limits in the test driver).
//   3. CPU oracle: for each (i, kbx, kqsx) tuple, compute the
//      expected x_qs ints (low/high nibble split + sign-recenter) and
//      expected x_df scale.
//   4. Compare byte-for-byte.
//
// Sweep: mmq_y ∈ {16, 32, 64, 128}, nwarps ∈ {4, 8}, need_check ∈ {false, true}.
// PASS = all comparisons exact across all sweep configs.

#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_CHECK(stmt) do { \
    cudaError_t _e = (stmt); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        std::exit(1); \
    } \
} while (0)

// Wrapper kernel: parameterise mmq_y / nwarps / need_check via template, call
// load_tiles_q4_0_ar16 to write into a per-CTA SMEM tile, then copy SMEM to
// the per-CTA output slice of global memory for host verification.
//
// kbx0 = 0 (first WARP_SIZE k positions). Single CTA test (gridDim.x=1).
template <int MMQ_Y, int NWARPS>
__global__ void test_load_tiles_q4_0_ar16_kernel(
        const char * __restrict__ x,
        int          stride_bytes,
        int          i_max,
        bool         need_check_flag,
        int   *      out_x_qs,
        float *      out_x_df) {

    extern __shared__ int smem[];
    int * x_tile = smem;

    // Dispatch on need_check_flag at runtime via template branches.
    if (need_check_flag) {
        load_tiles_q4_0_ar16<MMQ_Y, NWARPS, /*need_check=*/true>(
            x, x_tile, /*kbx0=*/0, i_max, stride_bytes);
    } else {
        load_tiles_q4_0_ar16<MMQ_Y, NWARPS, /*need_check=*/false>(
            x, x_tile, /*kbx0=*/0, i_max, stride_bytes);
    }

    __syncthreads();

    // Copy SMEM x_qs + x_df slices to global memory for host verification.
    // Layout per load_tiles_q4_0_ar16's INT8_MMA path:
    //   x_qs at offset 0, sized 2*WARP_SIZE = 64 ints per row.
    //   x_df at offset 2*WARP_SIZE, sized MMQ_MMA_TILE_X_K_Q4_0_AR16 - 2*WARP_SIZE per row.
    // We dump the full MMQ_MMA_TILE_X_K_Q4_0_AR16 stride per row for completeness.
    const int per_row = MMQ_MMA_TILE_X_K_Q4_0_AR16;
    const int n_total = MMQ_Y * per_row;
    const int tid_flat = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthread  = blockDim.x * blockDim.y;
    for (int j = tid_flat; j < n_total; j += nthread) {
        // x_qs portion: first 2*WARP_SIZE ints per row.
        // x_df portion: starts at 2*WARP_SIZE within each row.
        const int row = j / per_row;
        const int col = j % per_row;
        if (col < 2*WARP_SIZE) {
            out_x_qs[row * 2*WARP_SIZE + col] = x_tile[row * per_row + col];
        } else {
            const int df_col = col - 2*WARP_SIZE;
            const int n_df_per_row = MMQ_MMA_TILE_X_K_Q4_0_AR16 - 2*WARP_SIZE;
            out_x_df[row * n_df_per_row + df_col] = ((const float *)x_tile)[row * per_row + col];
        }
    }
}

// CPU oracle: compute expected x_qs / x_df contents under the §2.5 unified
// linear-K-per-int layout.
//
// AR16 source-byte convention (per dequantize_block_q4_0_ar16):
//   qs[i].low  = K position 2*i  (even K)
//   qs[i].high = K position 2*i+1 (odd  K)
//
// load_tiles_q4_0_ar16 emits to x_qs[i*stride + kbx*4 + s] (s ∈ 0..3) where
// slot s holds 4 sign-recentered int8 quants at K positions [4s..4s+3] of
// block kbx. The unpack regroups even/odd nibbles via __byte_perm.
static void cpu_oracle_x_qs(
        const std::vector<uint8_t> & src_bytes,
        int n_rows, int stride_bytes,
        int kbx0_blocks,
        std::vector<int32_t> & out_x_qs) {
    const int per_row = 2*WARP_SIZE;  // 64 ints
    out_x_qs.assign((size_t)n_rows * per_row, 0);

    for (int i = 0; i < n_rows; ++i) {
        for (int kbx = 0; kbx < 16; ++kbx) {  // 16 AR16 blocks per warp tile
            const size_t block_off = (size_t)i * stride_bytes
                                   + (size_t)(kbx0_blocks + kbx) * sizeof(block_q4_0_ar16);
            // block_q4_0_ar16: 2 bytes d (fp16) + 8 bytes qs.
            const uint8_t * qs = src_bytes.data() + block_off + sizeof(uint16_t);
            // Dequant the 16 K positions of this block: K=2*j is qs[j]&0xF, K=2*j+1 is qs[j]>>4. Sign-recenter.
            int8_t k_vals[16];
            for (int j = 0; j < 8; ++j) {
                k_vals[2*j + 0] = (int8_t)(qs[j] & 0x0F) - 8;
                k_vals[2*j + 1] = (int8_t)((qs[j] >> 4) & 0x0F) - 8;
            }
            // Pack into 4 linear-K-per-int ints: slot s holds [K=4s, K=4s+1, K=4s+2, K=4s+3].
            for (int s = 0; s < 4; ++s) {
                int32_t packed = 0;
                for (int b = 0; b < 4; ++b) {
                    packed |= ((int32_t)(uint8_t)k_vals[4*s + b]) << (b*8);
                }
                out_x_qs[i * per_row + kbx*4 + s] = packed;
            }
        }
    }
}

static void cpu_oracle_x_df(
        const std::vector<uint8_t> & src_bytes,
        int n_rows, int stride_bytes,
        int kbx0_blocks,
        std::vector<float> & out_x_df) {
    const int n_df_per_row = MMQ_MMA_TILE_X_K_Q4_0_AR16 - 2*WARP_SIZE;
    out_x_df.assign((size_t)n_rows * n_df_per_row, 0.0f);

    // load_tiles_q4_0_ar16's scale-load loop writes one scale per block. blocks_per_tile_x_row = 16.
    // Each row writes scales at offsets 0..15 within the x_df portion.
    for (int i = 0; i < n_rows; ++i) {
        for (int kbxd = 0; kbxd < 16; ++kbxd) {
            const size_t block_off = (size_t)i * stride_bytes
                                   + (size_t)(kbx0_blocks + kbxd) * sizeof(block_q4_0_ar16);
            uint16_t d_bits = 0;
            d_bits |= ((uint16_t)src_bytes[block_off + 0]) << 0;
            d_bits |= ((uint16_t)src_bytes[block_off + 1]) << 8;
            // Convert fp16 → float via the same path the kernel uses (bxi->d → float).
            // Kernel stores `bxi->d` (ggml_half) into float slot via implicit conversion.
            const __half h = __ushort_as_half(d_bits);
            out_x_df[i * n_df_per_row + kbxd] = __half2float(h);
        }
    }
}

template <int MMQ_Y, int NWARPS>
static bool run_config(std::mt19937 & rng, bool need_check_flag, int i_max) {
    const int n_rows_src = MMQ_Y;
    const int n_blocks_per_row = 16;  // we test kbx0=0, covering 16 blocks
    const int stride_bytes = n_blocks_per_row * (int)sizeof(block_q4_0_ar16);

    // Random source bytes.
    std::vector<uint8_t> src_bytes((size_t)n_rows_src * stride_bytes);
    for (auto & b : src_bytes) {
        b = (uint8_t)(rng() & 0xff);
    }
    // Force the fp16 scales to representable normal values (avoid NaN/Inf).
    for (int i = 0; i < n_rows_src; ++i) {
        for (int k = 0; k < n_blocks_per_row; ++k) {
            const size_t off = (size_t)i * stride_bytes + (size_t)k * sizeof(block_q4_0_ar16);
            const float d = ((rng() & 0xff) / 256.0f - 0.5f) * 0.1f + 0.05f;
            const uint16_t d_bits = __half_as_ushort(__float2half(d));
            src_bytes[off + 0] = (uint8_t)(d_bits & 0xff);
            src_bytes[off + 1] = (uint8_t)((d_bits >> 8) & 0xff);
        }
    }

    // Allocate device source.
    char * d_src = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, src_bytes.size()));
    CUDA_CHECK(cudaMemcpy(d_src, src_bytes.data(), src_bytes.size(), cudaMemcpyHostToDevice));

    // Output buffers (kernel-side dump).
    const int per_row    = MMQ_MMA_TILE_X_K_Q4_0_AR16;
    const int n_x_qs     = MMQ_Y * 2*WARP_SIZE;
    const int n_df_per_r = per_row - 2*WARP_SIZE;
    const int n_x_df     = MMQ_Y * n_df_per_r;
    int   * d_x_qs = nullptr;
    float * d_x_df = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x_qs, (size_t)n_x_qs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x_df, (size_t)n_x_df * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_x_qs, 0, (size_t)n_x_qs * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_x_df, 0, (size_t)n_x_df * sizeof(float)));

    // SMEM allocation: per-row stride * MMQ_Y rows, in ints.
    const size_t smem_bytes = (size_t)MMQ_Y * per_row * sizeof(int);

    // Launch: 1 CTA, blockDim (WARP_SIZE, NWARPS, 1).
    dim3 block(WARP_SIZE, NWARPS, 1);
    dim3 grid(1, 1, 1);
    test_load_tiles_q4_0_ar16_kernel<MMQ_Y, NWARPS><<<grid, block, smem_bytes>>>(
        d_src, stride_bytes, i_max, need_check_flag, d_x_qs, d_x_df);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back.
    std::vector<int32_t> host_x_qs((size_t)n_x_qs, 0);
    std::vector<float>   host_x_df((size_t)n_x_df, 0.0f);
    CUDA_CHECK(cudaMemcpy(host_x_qs.data(), d_x_qs, (size_t)n_x_qs * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_x_df.data(), d_x_df, (size_t)n_x_df * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_src);
    cudaFree(d_x_qs);
    cudaFree(d_x_df);

    // CPU oracle.
    std::vector<int32_t> ref_x_qs;
    std::vector<float>   ref_x_df;
    cpu_oracle_x_qs(src_bytes, MMQ_Y, stride_bytes, /*kbx0_blocks=*/0, ref_x_qs);
    cpu_oracle_x_df(src_bytes, MMQ_Y, stride_bytes, /*kbx0_blocks=*/0, ref_x_df);

    // When need_check_flag is set, the kernel clamps i to i_max. For our
    // oracle we'd need to similarly clamp; simpler: only compare rows <= i_max.
    const int rows_to_compare = need_check_flag ? std::min(MMQ_Y, i_max + 1) : MMQ_Y;

    // Compare x_qs.
    size_t mismatches_qs = 0;
    for (int i = 0; i < rows_to_compare; ++i) {
        for (int j = 0; j < 2*WARP_SIZE; ++j) {
            if (host_x_qs[i * 2*WARP_SIZE + j] != ref_x_qs[i * 2*WARP_SIZE + j]) {
                if (mismatches_qs < 5) {
                    fprintf(stderr, "  x_qs[%d][%d]: kernel=0x%08x ref=0x%08x\n",
                            i, j, host_x_qs[i*2*WARP_SIZE + j], ref_x_qs[i*2*WARP_SIZE + j]);
                }
                ++mismatches_qs;
            }
        }
    }
    size_t mismatches_df = 0;
    for (int i = 0; i < rows_to_compare; ++i) {
        for (int j = 0; j < 16; ++j) {  // 16 scales per row in AR16
            const float k_val = host_x_df[i * n_df_per_r + j];
            const float r_val = ref_x_df[i * n_df_per_r + j];
            if (k_val != r_val) {
                if (mismatches_df < 5) {
                    fprintf(stderr, "  x_df[%d][%d]: kernel=%.6e ref=%.6e\n",
                            i, j, k_val, r_val);
                }
                ++mismatches_df;
            }
        }
    }

    if (mismatches_qs == 0 && mismatches_df == 0) {
        fprintf(stderr, "  PASS: mmq_y=%d nwarps=%d need_check=%d i_max=%d\n",
                MMQ_Y, NWARPS, (int)need_check_flag, i_max);
        return true;
    } else {
        fprintf(stderr, "  FAIL: mmq_y=%d nwarps=%d need_check=%d i_max=%d  qs_mismatches=%zu df_mismatches=%zu\n",
                MMQ_Y, NWARPS, (int)need_check_flag, i_max,
                mismatches_qs, mismatches_df);
        return false;
    }
}

int main() {
    int n_devices = 0;
    if (cudaGetDeviceCount(&n_devices) != cudaSuccess || n_devices == 0) {
        fprintf(stderr, "no CUDA device available; SKIP\n");
        return 77;
    }
    CUDA_CHECK(cudaSetDevice(0));

    std::mt19937 rng(0xC0FFEEULL);
    bool ok = true;

    // Sweep (mmq_y, nwarps) consistent with MMQ kernel templating.
    // Under unified §2.5 layout: MMQ_MMA_TILE_X_K_Q4_0_AR16 = 84 ints/row.
    // mmq_y=128 → 128*84*4 = 43008 bytes = 42 KB SMEM. Fits Turing default 48 KB.
    // Production MMQ uses mmq_y ∈ {8, 16, 32, 64, 128} per ggml_cuda_mmq_get_y_host.
    ok &= run_config<16,  4>(rng, /*need_check=*/false, /*i_max=*/15);
    ok &= run_config<32,  4>(rng, /*need_check=*/false, /*i_max=*/31);
    ok &= run_config<64,  4>(rng, /*need_check=*/false, /*i_max=*/63);
    ok &= run_config<64,  8>(rng, /*need_check=*/false, /*i_max=*/63);
    ok &= run_config<128, 8>(rng, /*need_check=*/false, /*i_max=*/127);

    // need_check=true with i_max < mmq_y-1.
    ok &= run_config<32, 4>(rng, /*need_check=*/true, /*i_max=*/20);
    ok &= run_config<64, 8>(rng, /*need_check=*/true, /*i_max=*/40);
    ok &= run_config<128, 8>(rng, /*need_check=*/true, /*i_max=*/100);

    if (ok) {
        fprintf(stderr, "OVERALL: PASS\n");
        return 0;
    } else {
        fprintf(stderr, "OVERALL: FAIL\n");
        return 1;
    }
}
