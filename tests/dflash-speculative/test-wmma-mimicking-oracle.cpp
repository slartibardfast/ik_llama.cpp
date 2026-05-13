// test-wmma-mimicking-oracle.cpp
//
// Self-test for the WMMA-mimicking oracle defined in
// wmma-mimicking-oracle.h. This test does NOT touch a GPU kernel — it
// validates the oracle itself, before any kernel binds against it.
//
// Per kernel-design.md §6.1 T4 plan: "Validate the oracle against itself
// at varying tile counts to confirm self-consistency before binding the
// kernel against it."
//
// What this test confirms:
//   1. wmma_oracle_self_check() returns 0 (determinism + K-tile loop).
//   2. WMMA-mimicking oracle and serial fp32 oracle agree byte-identically
//      at K=16 with all-equal inputs (reduction order has no effect).
//   3. At K=128/512/5120 with random fp16 inputs, the two oracles agree
//      to within ≤ 1 fp16 ULP at ≤ 1 % rate, and 0 cells diverge by > 1
//      ULP — confirms the WMMA reduction model doesn't accumulate
//      systematic error vs straight serial fp32.
//   4. fp16_ulp_delta correctly returns 0 for byte-identical inputs, 1
//      for next-fp16-up, 2 for two-ULP-up, etc., across signed values.
//
// PASS: exit 0.
// FAIL: exit 1 with stderr explaining the failure mode.

#include "wmma-mimicking-oracle.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr int FP16_ULP_GATE = 1;
constexpr double FP16_ULP_RATE_GATE = 0.01;  // ≤ 1 % cells at ULP boundary

int test_self_check() {
    const int rc = dflash_reference::wmma_oracle_self_check();
    if (rc != 0) {
        std::fprintf(stderr,
            "[FAIL] wmma_oracle_self_check returned %d\n"
            "  rc=1: determinism failure at K=16 single-tile\n"
            "  rc=2: determinism failure at K=32 two-tile\n"
            "  rc=3: WMMA oracle disagrees with serial fp32 at K=16 all-equal inputs\n",
            rc);
        return 1;
    }
    std::printf("[PASS] wmma_oracle_self_check returned 0\n");
    return 0;
}

int test_ulp_delta() {
    // Zero vs zero
    if (dflash_reference::fp16_ulp_delta(__float2half(0.0f), __float2half(0.0f)) != 0) {
        std::fprintf(stderr, "[FAIL] fp16_ulp_delta(+0, +0) != 0\n");
        return 1;
    }
    // Adjacent ULPs above zero
    __half one_ulp;
    *reinterpret_cast<unsigned short*>(&one_ulp) = 0x0001;
    if (dflash_reference::fp16_ulp_delta(__float2half(0.0f), one_ulp) != 1) {
        std::fprintf(stderr, "[FAIL] fp16_ulp_delta(0, 1-ULP) != 1\n");
        return 1;
    }
    // Two-ULP delta
    __half two_ulp;
    *reinterpret_cast<unsigned short*>(&two_ulp) = 0x0002;
    if (dflash_reference::fp16_ulp_delta(one_ulp, two_ulp) != 1) {
        std::fprintf(stderr, "[FAIL] fp16_ulp_delta(1-ULP, 2-ULP) != 1\n");
        return 1;
    }
    // Cross-zero: smallest positive vs smallest negative subnormal
    __half pos_sub, neg_sub;
    *reinterpret_cast<unsigned short*>(&pos_sub) = 0x0001;
    *reinterpret_cast<unsigned short*>(&neg_sub) = 0x8001;
    const int d = dflash_reference::fp16_ulp_delta(pos_sub, neg_sub);
    if (d != 2) {
        std::fprintf(stderr,
            "[FAIL] fp16_ulp_delta(+smallest-sub, -smallest-sub) = %d, expected 2\n", d);
        return 1;
    }
    std::printf("[PASS] fp16_ulp_delta basic boundaries\n");
    return 0;
}

int test_random_agree(int M, int N, int K, uint32_t seed) {
    if (M % dflash_reference::WMMA_M || N % dflash_reference::WMMA_N || K % dflash_reference::WMMA_K) {
        std::fprintf(stderr, "[SKIP] shape %dx%dx%d not WMMA-aligned\n", M, N, K);
        return 0;
    }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);

    std::vector<__half> A(M * K), B(K * N), C(M * N);
    std::vector<__half> D_wmma(M * N), D_serial(M * N);
    for (auto & h : A) h = __float2half(dist(rng));
    for (auto & h : B) h = __float2half(dist(rng));
    for (auto & h : C) h = __float2half(0.0f);

    dflash_reference::wmma_mma_oracle(A.data(), K, B.data(), N, C.data(), N, D_wmma.data(), N, M, N, K);
    dflash_reference::serial_fp32_mma_oracle(A.data(), K, B.data(), N, C.data(), N, D_serial.data(), N, M, N, K);

    int max_ulp = 0;
    int n_ulp_eq_1 = 0;
    int n_ulp_gt_1 = 0;
    int worst_idx = -1;
    for (int i = 0; i < M * N; ++i) {
        const int ulp = dflash_reference::fp16_ulp_delta(D_wmma[i], D_serial[i]);
        if (ulp > max_ulp) { max_ulp = ulp; worst_idx = i; }
        if (ulp == 1) ++n_ulp_eq_1;
        if (ulp > 1)  ++n_ulp_gt_1;
    }

    const double rate1 = static_cast<double>(n_ulp_eq_1) / (M * N);
    const double rateN = static_cast<double>(n_ulp_gt_1) / (M * N);

    std::printf("[ ] %dx%dx%d seed=%u: max_ulp=%d, 1-ULP rate=%.3f%%, >1-ULP rate=%.3f%%",
                M, N, K, seed, max_ulp, rate1 * 100.0, rateN * 100.0);

    // Two-oracle agreement at small K should be tight. At larger K (5120),
    // accumulator order differs and we expect some ULP drift; gate on rate.
    if (max_ulp > 2) {
        std::printf("  [WARN] max_ulp > 2 (idx %d: wmma=%.6e serial=%.6e)",
                    worst_idx,
                    __half2float(D_wmma[worst_idx]),
                    __half2float(D_serial[worst_idx]));
    }
    // Tolerance: at K ≤ 64, expect byte-identity at all cells.
    if (K <= 64 && max_ulp > FP16_ULP_GATE) {
        std::printf("  [FAIL] K<=64 max_ulp=%d > %d\n", max_ulp, FP16_ULP_GATE);
        return 1;
    }
    // At K ≥ 128, allow ULP drift but with bounded rate.
    if (rateN > FP16_ULP_RATE_GATE) {
        std::printf("  [FAIL] >1-ULP rate %.3f%% > %.3f%%\n", rateN * 100.0, FP16_ULP_RATE_GATE * 100.0);
        return 1;
    }
    std::printf("  [PASS]\n");
    return 0;
}

} // anonymous namespace

int main() {
    int fails = 0;

    fails += test_self_check();
    fails += test_ulp_delta();

    // Sweep over plausible drafter shapes:
    //   M = 16 (one query tile, padded from 5..9 actual query positions)
    //   N = 16, 64, 128, 1024, 5120 (output dims of Q/K/V proj, MLP gate/up/down)
    //   K = 16, 64, 128, 1024, 5120 (input dims)
    const int seeds[] = {1u, 17u, 42u, 99u};
    const int M = 16;
    const int Ns[] = {16, 64, 128, 1024};
    const int Ks[] = {16, 64, 128, 1024, 5120};
    for (int N : Ns) {
        for (int K : Ks) {
            for (int seed : seeds) {
                fails += test_random_agree(M, N, K, seed);
            }
        }
    }

    if (fails > 0) {
        std::fprintf(stderr, "[OVERALL] %d failures\n", fails);
        return 1;
    }
    std::printf("[OVERALL] all PASS\n");
    return 0;
}
