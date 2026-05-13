// wmma-mimicking-oracle.h
//
// CPU emulation of CUDA WMMA m16n16k16 matrix-multiply-accumulate as it
// runs on Turing (sm_75) tensor cores via PTX `mma.sync.aligned.m16n8k8`
// (which the WMMA C++ API composes twice along N to realise m16n16k16).
//
// Used as a test oracle for `dflash_drafter_forward` and other DFlash
// kernels that include WMMA paths. Pairs with `serial_fp32_mma_oracle`
// (also defined below) as a wider-tolerance sanity check — any kernel
// passing both the WMMA-mimicking oracle (tight fp16 ULP gate) AND the
// serial fp32 oracle (cosine + NMSE gate) is bound on both reduction-
// order-sensitive and reduction-order-insensitive numerical structure.
//
// Spec: specs/dflash/kernel-design.md §6.1
//       specs/dflash/dflash.allium DraftBlockEmit contract
//
// Internal precision model (mirrors what the tensor core does, refined
// from spec's literal "fp16/fp16" — the Turing tensor core multiplies
// fp16 × fp16 → fp32 internally, accumulates in fp32, and rounds back to
// fp16 only at fragment store):
//   - fp16 multiply lifted to fp32: __half2float(a) * __half2float(b)
//   - Per-tile accumulation: fp32 binary tree over 16 K-elements
//   - Cross-tile accumulation: fp32 serial sum across K-tiles
//   - Final output: fp32 → fp16 cast (__float2half) once per output cell
//
// PTX-reduction-order model:
//   m16n16k16 = two m16n8k8 PTX ops side-by-side along N.
//   Each m16n8k8 op accumulates 8 K-elements per output cell. We model
//   the 16-element K-tile as two 8-element halves (each its own binary
//   tree), summed at the end of the tile.
//
// This produces:
//   tile_acc =
//     (((a0*b0 + a1*b1) + (a2*b2 + a3*b3)) + ((a4*b4 + a5*b5) + (a6*b6 + a7*b7)))
//     +
//     (((a8*b8 + a9*b9) + (a10*b10 + a11*b11)) + ((a12*b12 + a13*b13) + (a14*b14 + a15*b15)))
//
// Reduction order across K-tiles is outer-loop serial in fp32 — matches
// what a sane kernel would do at K-tile granularity.
//
// Limitations: actual tensor-core internal reduction order is partially
// undocumented and may differ from this binary-tree model by sub-ULP at
// fp32. The serial_fp32_mma_oracle sanity-checks at wider tolerance.

#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_fp16.h>

namespace dflash_reference {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Binary tree reduction over 8 fp32 values (one m16n8k8 op's K-dim).
inline float reduce8_tree(const float * p) {
    const float l0 = p[0] + p[1];
    const float l1 = p[2] + p[3];
    const float l2 = p[4] + p[5];
    const float l3 = p[6] + p[7];
    const float ll = l0 + l1;
    const float lr = l2 + l3;
    return ll + lr;
}

// Reduction over 16 fp32 values modelling two m16n8k8 mma.sync ops:
// first half + second half (each via reduce8_tree).
inline float wmma_tile_reduce(const float * products) {
    return reduce8_tree(products) + reduce8_tree(products + 8);
}

// WMMA-mimicking matrix-multiply-accumulate.
//
//   D = A * B + C
//   A: M x K  (row-major, fp16)
//   B: K x N  (row-major, fp16)
//   C: M x N  (row-major, fp16, input accumulator)
//   D: M x N  (row-major, fp16, output)
//
// Strides are in elements, not bytes. M, N, K must all be multiples of
// WMMA_M, WMMA_N, WMMA_K = 16.
//
// Internal accumulator is fp32 (matches Turing tensor core hardware:
// fp16 input lifted to fp32 for accumulation, output rounded to fp16).
//
// Within each 16x16x16 tile, the K-axis reduction follows the
// reduce8_tree+reduce8_tree pattern that mirrors PTX m16n8k8 fragment
// reduction. Across K-tiles, accumulation is serial fp32 sum.
inline void wmma_mma_oracle(
    const __half * A, int A_stride_k,
    const __half * B, int B_stride_n,
    const __half * C, int C_stride_n,
    __half       * D, int D_stride_n,
    int M, int N, int K)
{
    assert(M % WMMA_M == 0);
    assert(N % WMMA_N == 0);
    assert(K % WMMA_K == 0);
    const int K_tiles = K / WMMA_K;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc_fp32 = __half2float(C[m * C_stride_n + n]);
            for (int kt = 0; kt < K_tiles; ++kt) {
                float products[WMMA_K];
                const int k_base = kt * WMMA_K;
                #pragma unroll
                for (int ki = 0; ki < WMMA_K; ++ki) {
                    const int k = k_base + ki;
                    const float a = __half2float(A[m * A_stride_k + k]);
                    const float b = __half2float(B[k * B_stride_n + n]);
                    products[ki] = a * b;
                }
                acc_fp32 += wmma_tile_reduce(products);
            }
            D[m * D_stride_n + n] = __float2half(acc_fp32);
        }
    }
}

// Serial-fp32 GEMM reference for the wider-tolerance sanity check.
//
// Same shape contract as wmma_mma_oracle (M, N, K can be any positive
// integers — does NOT require multiples of 16). Reduction order is
// straight serial over K.
inline void serial_fp32_mma_oracle(
    const __half * A, int A_stride_k,
    const __half * B, int B_stride_n,
    const __half * C, int C_stride_n,
    __half       * D, int D_stride_n,
    int M, int N, int K)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = __half2float(C[m * C_stride_n + n]);
            for (int k = 0; k < K; ++k) {
                const float a = __half2float(A[m * A_stride_k + k]);
                const float b = __half2float(B[k * B_stride_n + n]);
                acc += a * b;
            }
            D[m * D_stride_n + n] = __float2half(acc);
        }
    }
}

// fp32-output variant of the WMMA oracle. Same numerics as wmma_mma_oracle
// but writes fp32 outputs instead of rounding to fp16. Used when the
// caller wants to inspect the un-rounded accumulator (e.g. when the kernel
// outputs fp32 then casts; or when chained with RMSNorm/RoPE that operate
// in fp32 before final fp16 round).
inline void wmma_mma_oracle_fp32_out(
    const __half * A, int A_stride_k,
    const __half * B, int B_stride_n,
    const __half * C, int C_stride_n,    // accumulator input still fp16
    float        * D, int D_stride_n,
    int M, int N, int K)
{
    assert(M % WMMA_M == 0);
    assert(N % WMMA_N == 0);
    assert(K % WMMA_K == 0);
    const int K_tiles = K / WMMA_K;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc_fp32 = __half2float(C[m * C_stride_n + n]);
            for (int kt = 0; kt < K_tiles; ++kt) {
                float products[WMMA_K];
                const int k_base = kt * WMMA_K;
                #pragma unroll
                for (int ki = 0; ki < WMMA_K; ++ki) {
                    const int k = k_base + ki;
                    const float a = __half2float(A[m * A_stride_k + k]);
                    const float b = __half2float(B[k * B_stride_n + n]);
                    products[ki] = a * b;
                }
                acc_fp32 += wmma_tile_reduce(products);
            }
            D[m * D_stride_n + n] = acc_fp32;
        }
    }
}

// Self-consistency check: the oracle must produce identical output
// given identical input (pure function). Also exercises the K-tile loop
// across multiple tiles to confirm the outer-loop accumulation works.
//
// Returns 0 on PASS (deterministic + tile-loop consistent), nonzero on
// FAIL with a per-failure-mode code.
inline int wmma_oracle_self_check() {
    // 16x16 ⨯ 16x16 = 16x16, K_tiles = 1
    {
        std::vector<__half> A(16 * 16), B(16 * 16), C(16 * 16);
        std::vector<__half> D1(16 * 16), D2(16 * 16);
        for (int i = 0; i < 256; ++i) {
            A[i] = __float2half(0.001f * (i - 128));
            B[i] = __float2half(0.0005f * (256 - i));
            C[i] = __float2half(0.0f);
        }
        wmma_mma_oracle(A.data(), 16, B.data(), 16, C.data(), 16, D1.data(), 16, 16, 16, 16);
        wmma_mma_oracle(A.data(), 16, B.data(), 16, C.data(), 16, D2.data(), 16, 16, 16, 16);
        for (int i = 0; i < 256; ++i) {
            if (__half_as_short(D1[i]) != __half_as_short(D2[i])) return 1;
        }
    }

    // 16x16 ⨯ 16x32 = 16x16 padded out via two K-tiles, K_tiles = 2
    // Test that the K-tile outer loop produces deterministic results.
    {
        std::vector<__half> A(16 * 32), B(32 * 16), C(16 * 16);
        std::vector<__half> D1(16 * 16), D2(16 * 16);
        for (int i = 0; i < 16 * 32; ++i) A[i] = __float2half(0.0007f * i - 0.2f);
        for (int i = 0; i < 32 * 16; ++i) B[i] = __float2half(0.0011f * (i + 1));
        for (int i = 0; i < 16 * 16; ++i) C[i] = __float2half(0.001f * i);
        wmma_mma_oracle(A.data(), 32, B.data(), 16, C.data(), 16, D1.data(), 16, 16, 16, 32);
        wmma_mma_oracle(A.data(), 32, B.data(), 16, C.data(), 16, D2.data(), 16, 16, 16, 32);
        for (int i = 0; i < 16 * 16; ++i) {
            if (__half_as_short(D1[i]) != __half_as_short(D2[i])) return 2;
        }
    }

    // Confirm that the WMMA-mimicking oracle and the serial fp32 oracle
    // agree at small K (no reduction order divergence visible at K=16).
    {
        std::vector<__half> A(16 * 16), B(16 * 16), C(16 * 16);
        std::vector<__half> D_wmma(16 * 16), D_serial(16 * 16);
        for (int i = 0; i < 256; ++i) {
            A[i] = __float2half(0.01f);
            B[i] = __float2half(0.01f);
            C[i] = __float2half(0.0f);
        }
        wmma_mma_oracle(A.data(), 16, B.data(), 16, C.data(), 16, D_wmma.data(), 16, 16, 16, 16);
        serial_fp32_mma_oracle(A.data(), 16, B.data(), 16, C.data(), 16, D_serial.data(), 16, 16, 16, 16);
        // All-equal inputs → all-equal outputs; reduction order doesn't matter.
        for (int i = 0; i < 16 * 16; ++i) {
            if (__half_as_short(D_wmma[i]) != __half_as_short(D_serial[i])) return 3;
        }
    }

    return 0;
}

// Compute fp16-ULP delta between two fp16 values, treating them as signed
// magnitudes. For values of opposite sign, returns |bits_a| + |bits_b|
// (the canonical ULP distance across zero on a sign-magnitude format).
inline std::int32_t fp16_ulp_delta(__half a, __half b) {
    const std::int16_t a_bits = static_cast<std::int16_t>(__half_as_short(a));
    const std::int16_t b_bits = static_cast<std::int16_t>(__half_as_short(b));
    auto sign_mag = [](std::int16_t x) -> std::int32_t {
        if (x < 0) {
            // For fp16 sign-magnitude: negative bits ordered by magnitude.
            // Convert to "from zero" distance: bits XOR 0x8000.
            return -static_cast<std::int32_t>(x & 0x7FFF);
        }
        return static_cast<std::int32_t>(x);
    };
    const std::int32_t a_dist = sign_mag(a_bits);
    const std::int32_t b_dist = sign_mag(b_bits);
    return std::abs(a_dist - b_dist);
}

} // namespace dflash_reference
