// test-trace-4-warp-reduce-mask-shape.cu
//
// TRACE-4 / TRACE-5 / TRACE-7 — isolate warp_reduce_sum / warp_reduce_max
// non-associativity over mask-shape-dependent lane patterns.
//
// Hypothesis (from data/trace-3-2026-05-16/findings.md):
//   The FA per-slot-kv kernel's softmax warp_reduce_sum is non-associative
//   over fp32 inputs. When two FA queries have the SAME nonzero K-cell
//   values but at DIFFERENT lane positions in their mask, the warp-shuffle
//   tree reduces in a different order → different fp32 sums.
//
// The kernel's reduction matches the production ggml/src/ggml-cuda/common.cuh
// definition exactly:
//
//   static __device__ __forceinline__ float warp_reduce_sum(float x) {
//       #pragma unroll
//       for (int mask = 16; mask > 0; mask >>= 1)
//           x += __shfl_xor_sync(0xffffffff, x, mask, 32);
//       return x;
//   }
//
// Three experiments:
//   T4: warp_reduce_sum across two lane patterns ({0..11, 24} vs {12..23, 25}).
//   T5: same fixture but warp_reduce_max — must be byte-identical (max is
//       associative-commutative under fp32; warp shuffle traversal can't
//       change the result).
//   T7: compact-then-reduce fix candidate — gather valid (nonzero) values
//       into lanes [0..n_valid) BEFORE warp_reduce_sum. Result must be
//       byte-identical between the two patterns.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum_xor(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

__device__ __forceinline__ float warp_reduce_max_xor(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}

__device__ __forceinline__ float warp_reduce_sum_compacted(float x, int valid_mask_bits) {
    // Compact-then-reduce: rebase nonzero lanes to lanes [0..popcount(valid_mask_bits)).
    // Each lane's "compacted_idx" is the popcount of valid_mask_bits below its own bit.
    // For our test we pass valid_mask_bits as the 32-bit pattern of which lanes are valid.
    const int tid = threadIdx.x & 31;
    const int my_bit = 1u << tid;
    const bool is_valid = (valid_mask_bits & my_bit) != 0;
    int rank = 0;
    if (is_valid) {
        unsigned int below = (unsigned int) valid_mask_bits & (my_bit - 1u);
        rank = __popc(below);
    }
    // Shuffle x to lane `rank` if valid, else 0.
    // Use a per-step gather: for each valid source lane, broadcast to dest lane = rank.
    // Simpler approach: build a fully-redistributed value at every lane.
    float compacted = 0.0f;
    for (int src = 0; src < 32; ++src) {
        if (!(valid_mask_bits & (1u << src))) continue;
        unsigned int below = (unsigned int) valid_mask_bits & ((1u << src) - 1u);
        int dst = __popc(below);
        const float vsrc = __shfl_sync(0xffffffff, x, src, 32);
        if (tid == dst) {
            compacted = vsrc;
        }
    }
    // Now compacted holds nonzero values at lanes [0..n_valid); zero elsewhere.
    return warp_reduce_sum_xor(compacted);
}

// Kernel: each block runs ONE warp. Lane data is passed in via the `vals` array
// (32 floats). The kernel also receives `valid_mask_bits` (which lanes count as
// "valid" for the compact-then-reduce variant), but for T4/T5 the input vals
// already encode the slot pattern (zeros in invalid lanes).
__global__ void run_kernels(
        const float * vals_xor_sum,  // 32 floats for T4 sum (raw XOR-shuffle)
        const float * vals_xor_max,  // 32 floats for T5 max
        const float * vals_compact,  // 32 floats for T7 (raw values)
        int valid_mask_bits,         // pattern of valid lanes for T7
        float * out_xor_sum,
        float * out_xor_max,
        float * out_compact) {
    const int tid = threadIdx.x & 31;

    float x_sum = vals_xor_sum[tid];
    float r_sum = warp_reduce_sum_xor(x_sum);

    float x_max = vals_xor_max[tid];
    float r_max = warp_reduce_max_xor(x_max);

    float x_cp  = vals_compact[tid];
    float r_cp  = warp_reduce_sum_compacted(x_cp, valid_mask_bits);

    if (tid == 0) {
        *out_xor_sum = r_sum;
        *out_xor_max = r_max;
        *out_compact = r_cp;
    }
}

static void check_cuda(cudaError_t e, const char * what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    // 13 nonzero values modeling exp(scaled_KQ) at slot's valid K positions.
    // Use distinct values that span fp32 dynamic range; chosen so the sum
    // ordering matters at the bit level.
    const std::vector<float> values = {
        1.234567f, 2.345678f, 3.456789f, 4.567890f, 5.678901f,
        6.789012f, 7.890123f, 8.901234f, 9.012345f, 10.123456f,
        11.234567f, 12.345678f, 13.456789f
    };
    const int n_valid = (int) values.size();

    // Slot 0 lane pattern: valid at {0..11, 24}.
    std::vector<float> slot0(32, 0.0f);
    int sl0_idx = 0;
    int sl0_bits = 0;
    for (int p : {0,1,2,3,4,5,6,7,8,9,10,11,24}) {
        slot0[p] = values[sl0_idx++];
        sl0_bits |= (1 << p);
    }

    // Slot 1 lane pattern: valid at {12..23, 25}.
    std::vector<float> slot1(32, 0.0f);
    int sl1_idx = 0;
    int sl1_bits = 0;
    for (int p : {12,13,14,15,16,17,18,19,20,21,22,23,25}) {
        slot1[p] = values[sl1_idx++];
        sl1_bits |= (1 << p);
    }

    // Reference: serial fp32 sum of the 13 values in input order.
    double ref_d = 0.0;
    for (float v : values) ref_d += v;
    const float ref_f = (float) ref_d;
    fprintf(stderr, "Reference (serial double-precision sum cast to float): %.9g  bits=%08x\n",
            ref_f, *reinterpret_cast<const uint32_t*>(&ref_f));

    // Device buffers.
    float *d_vals0, *d_vals1, *d_out0_sum, *d_out1_sum, *d_out0_max, *d_out1_max, *d_out0_cp, *d_out1_cp;
    check_cuda(cudaMalloc(&d_vals0, 32 * sizeof(float)), "malloc d_vals0");
    check_cuda(cudaMalloc(&d_vals1, 32 * sizeof(float)), "malloc d_vals1");
    check_cuda(cudaMalloc(&d_out0_sum, sizeof(float)), "malloc out0 sum");
    check_cuda(cudaMalloc(&d_out1_sum, sizeof(float)), "malloc out1 sum");
    check_cuda(cudaMalloc(&d_out0_max, sizeof(float)), "malloc out0 max");
    check_cuda(cudaMalloc(&d_out1_max, sizeof(float)), "malloc out1 max");
    check_cuda(cudaMalloc(&d_out0_cp, sizeof(float)), "malloc out0 cp");
    check_cuda(cudaMalloc(&d_out1_cp, sizeof(float)), "malloc out1 cp");
    check_cuda(cudaMemcpy(d_vals0, slot0.data(), 32 * sizeof(float), cudaMemcpyHostToDevice), "memcpy slot0");
    check_cuda(cudaMemcpy(d_vals1, slot1.data(), 32 * sizeof(float), cudaMemcpyHostToDevice), "memcpy slot1");

    run_kernels<<<1, 32>>>(d_vals0, d_vals0, d_vals0, sl0_bits, d_out0_sum, d_out0_max, d_out0_cp);
    run_kernels<<<1, 32>>>(d_vals1, d_vals1, d_vals1, sl1_bits, d_out1_sum, d_out1_max, d_out1_cp);
    check_cuda(cudaDeviceSynchronize(), "sync");

    float h_out0_sum, h_out1_sum, h_out0_max, h_out1_max, h_out0_cp, h_out1_cp;
    check_cuda(cudaMemcpy(&h_out0_sum, d_out0_sum, sizeof(float), cudaMemcpyDeviceToHost), "get out0 sum");
    check_cuda(cudaMemcpy(&h_out1_sum, d_out1_sum, sizeof(float), cudaMemcpyDeviceToHost), "get out1 sum");
    check_cuda(cudaMemcpy(&h_out0_max, d_out0_max, sizeof(float), cudaMemcpyDeviceToHost), "get out0 max");
    check_cuda(cudaMemcpy(&h_out1_max, d_out1_max, sizeof(float), cudaMemcpyDeviceToHost), "get out1 max");
    check_cuda(cudaMemcpy(&h_out0_cp, d_out0_cp, sizeof(float), cudaMemcpyDeviceToHost), "get out0 cp");
    check_cuda(cudaMemcpy(&h_out1_cp, d_out1_cp, sizeof(float), cudaMemcpyDeviceToHost), "get out1 cp");

    uint32_t bits_out0_sum, bits_out1_sum, bits_out0_max, bits_out1_max, bits_out0_cp, bits_out1_cp;
    std::memcpy(&bits_out0_sum, &h_out0_sum, sizeof(uint32_t));
    std::memcpy(&bits_out1_sum, &h_out1_sum, sizeof(uint32_t));
    std::memcpy(&bits_out0_max, &h_out0_max, sizeof(uint32_t));
    std::memcpy(&bits_out1_max, &h_out1_max, sizeof(uint32_t));
    std::memcpy(&bits_out0_cp,  &h_out0_cp,  sizeof(uint32_t));
    std::memcpy(&bits_out1_cp,  &h_out1_cp,  sizeof(uint32_t));

    fprintf(stderr, "\n=== TRACE-4: warp_reduce_sum (XOR shuffle) ===\n");
    fprintf(stderr, "  slot 0 lanes {0..11, 24}: sum = %.9g  bits=%08x\n", h_out0_sum, bits_out0_sum);
    fprintf(stderr, "  slot 1 lanes {12..23, 25}: sum = %.9g  bits=%08x\n", h_out1_sum, bits_out1_sum);
    fprintf(stderr, "  bit-equal? %s  delta = %.3e\n",
            (bits_out0_sum == bits_out1_sum) ? "YES" : "NO",
            h_out0_sum - h_out1_sum);

    fprintf(stderr, "\n=== TRACE-5: warp_reduce_max (XOR shuffle) ===\n");
    fprintf(stderr, "  slot 0 lanes {0..11, 24}: max = %.9g  bits=%08x\n", h_out0_max, bits_out0_max);
    fprintf(stderr, "  slot 1 lanes {12..23, 25}: max = %.9g  bits=%08x\n", h_out1_max, bits_out1_max);
    fprintf(stderr, "  bit-equal? %s\n", (bits_out0_max == bits_out1_max) ? "YES" : "NO");

    fprintf(stderr, "\n=== TRACE-7: warp_reduce_sum (compact-then-reduce) ===\n");
    fprintf(stderr, "  slot 0 lanes {0..11, 24}: sum = %.9g  bits=%08x\n", h_out0_cp, bits_out0_cp);
    fprintf(stderr, "  slot 1 lanes {12..23, 25}: sum = %.9g  bits=%08x\n", h_out1_cp, bits_out1_cp);
    fprintf(stderr, "  bit-equal? %s  delta = %.3e\n",
            (bits_out0_cp == bits_out1_cp) ? "YES" : "NO",
            h_out0_cp - h_out1_cp);

    fprintf(stderr, "\n--- Summary ---\n");
    fprintf(stderr, "  T4 (warp_reduce_sum XOR): %s  (expected: NO — mask-shape non-associativity)\n",
            (bits_out0_sum == bits_out1_sum) ? "BYTE-EQUAL" : "DIFFER");
    fprintf(stderr, "  T5 (warp_reduce_max XOR): %s  (expected: YES — max associative-commutative)\n",
            (bits_out0_max == bits_out1_max) ? "BYTE-EQUAL" : "DIFFER");
    fprintf(stderr, "  T7 (compact-then-reduce): %s  (expected: YES — content-canonical order)\n",
            (bits_out0_cp == bits_out1_cp) ? "BYTE-EQUAL" : "DIFFER");

    cudaFree(d_vals0); cudaFree(d_vals1);
    cudaFree(d_out0_sum); cudaFree(d_out1_sum);
    cudaFree(d_out0_max); cudaFree(d_out1_max);
    cudaFree(d_out0_cp); cudaFree(d_out1_cp);

    // Exit code: 0 if all expected outcomes hold, else 1.
    bool pass = (bits_out0_sum != bits_out1_sum)   // T4 must DIFFER
             && (bits_out0_max == bits_out1_max)   // T5 must EQUAL
             && (bits_out0_cp  == bits_out1_cp);   // T7 must EQUAL
    return pass ? 0 : 1;
}
