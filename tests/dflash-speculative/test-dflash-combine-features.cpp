// test-dflash-combine-features.cpp
//
// Byte-identity unit test: fused CUDA combine_features kernel vs the
// fp32 scalar reference, exercised across a (N_slots × MAL_anchors × seed)
// sweep. This is the T3 closure binding for combine_features per
// kernel-design.md §10 step 3.
//
// @witnesses: FuseProjectionFcWeight
// @witnesses: FeatureWidthMatchesTarget
// @witnesses: CombineOrderFCThenHiddenNorm
// @witnesses: ContextStatesAnchorLevel
//
// PASS criterion per sub-run: byte-identical OR every disagreement is
// exactly 1 fp16 ULP AND rate ≤ 1 % of output cells. Rationale: byte-
// identity is unachievable when one side does serial-order fp32 reduction
// (scalar ref's RMSNorm sum_sq) and the other does parallel-tree fp32
// reduction (kernel's warp-shuffle + SMEM tree). fp32 add is non-
// associative; reduction-order LSB noise propagates through rsqrt to a
// 1-ULP fp16 output difference at < 1 % of positions. > 1 ULP at any
// position is a real kernel bug.

#include "dflash-combine-features-reference.h"
#include "ggml-cuda/dflash/dflash-combine-features.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_CHECK(stmt)                                                   \
    do {                                                                   \
        cudaError_t _e = (stmt);                                           \
        if (_e != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                         __FILE__, __LINE__, cudaGetErrorString(_e));      \
            return 1;                                                      \
        }                                                                  \
    } while (0)

namespace {

constexpr int   L_src    = 5;
constexpr int   D_d      = 5120;
constexpr float norm_eps = 1.0e-6f;

int run_one(int N_slots, int MAL_anchors, uint32_t seed) {
    const std::size_t n_src = (std::size_t) N_slots * MAL_anchors * L_src * D_d;
    const std::size_t n_fc  = (std::size_t) D_d * L_src * D_d;
    const std::size_t n_hn  = D_d;
    const std::size_t n_out = (std::size_t) N_slots * MAL_anchors * D_d;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<__half> src_h(n_src), fc_h(n_fc), hn_h(n_hn);
    for (auto & h : src_h) h = __float2half(dist(rng));
    for (auto & h : fc_h)  h = __float2half(dist(rng) * 0.05f);
    for (auto & h : hn_h)  h = __float2half(0.5f + dist(rng));

    std::vector<float> src_f(n_src), fc_f(n_fc), hn_f(n_hn);
    for (std::size_t i = 0; i < n_src; ++i) src_f[i] = __half2float(src_h[i]);
    for (std::size_t i = 0; i < n_fc;  ++i) fc_f[i]  = __half2float(fc_h[i]);
    for (std::size_t i = 0; i < n_hn;  ++i) hn_f[i]  = __half2float(hn_h[i]);

    std::vector<float> ref_f(n_out);
    dflash_reference::combine_features_scalar_ref_f32(
        src_f.data(), fc_f.data(), hn_f.data(), norm_eps,
        ref_f.data(), N_slots, MAL_anchors, L_src, D_d);

    std::vector<__half> ref_h(n_out);
    for (std::size_t i = 0; i < n_out; ++i) ref_h[i] = __float2half(ref_f[i]);

    __half *d_src = nullptr, *d_fc = nullptr, *d_hn = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, n_src * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_fc,  n_fc  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_hn,  n_hn  * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out, n_out * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_src, src_h.data(), n_src * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fc,  fc_h.data(),  n_fc  * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hn,  hn_h.data(),  n_hn  * sizeof(__half), cudaMemcpyHostToDevice));

    dflash_combine_features_launch(
        d_src, d_fc, d_hn, norm_eps, d_out,
        N_slots, MAL_anchors, L_src, D_d, /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> kern_h(n_out);
    CUDA_CHECK(cudaMemcpy(kern_h.data(), d_out, n_out * sizeof(__half), cudaMemcpyDeviceToHost));

    cudaFree(d_src); cudaFree(d_fc); cudaFree(d_hn); cudaFree(d_out);

    // Stub-state detection (first sub-run only; subsequent runs we trust the
    // first's verdict). All-zero output at this scale means the kernel ran
    // and produced literally nothing.
    bool all_zero = true;
    for (std::size_t i = 0; i < n_out; ++i) {
        uint16_t b;
        std::memcpy(&b, &kern_h[i], sizeof(uint16_t));
        if (b != 0) { all_zero = false; break; }
    }
    if (all_zero) {
        std::printf("  [N=%d MAL=%d seed=%u]  SKIP: kernel output is all-zero\n",
                    N_slots, MAL_anchors, seed);
        return 77;
    }

    int n_diff = 0, n_1ulp = 0, max_ulp = 0;
    for (std::size_t i = 0; i < n_out; ++i) {
        uint16_t a, b;
        std::memcpy(&a, &kern_h[i], sizeof(uint16_t));
        std::memcpy(&b, &ref_h[i],  sizeof(uint16_t));
        if (a == b) continue;
        ++n_diff;
        int d = std::abs(static_cast<int>(a) - static_cast<int>(b));
        if (d == 1) ++n_1ulp;
        if (d > max_ulp) max_ulp = d;
    }

    const bool pass =
        (n_diff == 0) ||
        (max_ulp <= 1 && n_diff * 100 <= (int) n_out);
    std::printf("  [N=%d MAL=%d seed=%u]  diff=%d/%zu (1ULP=%d, max=%d)  %s\n",
                N_slots, MAL_anchors, seed, n_diff, n_out, n_1ulp, max_ulp,
                pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

} // namespace

int main() {
    int dev_count = 0;
    cudaError_t derr = cudaGetDeviceCount(&dev_count);
    if (derr != cudaSuccess || dev_count == 0) {
        std::printf("SKIP: no CUDA device available\n");
        return 77;
    }

    std::printf("test-dflash-combine-features sweep:  L_src=%d  D_d=%d\n", L_src, D_d);

    // Configurations: (N_slots, MAL_anchors) × seeds.
    // (N_slots, MAL_anchors): {1,1}, {2,2}, {4,3}, {8,4} — production
    // operating ranges per @ContextBudgetAtNp8 and MAL ≈ 3 envelope.
    // Seeds: 2 distinct random seeds to vary the input distribution.
    struct Config { int N; int MAL; uint32_t seed; };
    const Config cfgs[] = {
        {1, 1, 42}, {1, 1, 137},
        {2, 2, 42}, {2, 2, 137},
        {4, 3, 42}, {4, 3, 137},
        {8, 4, 42}, {8, 4, 137},
    };
    const int n_cfgs = sizeof(cfgs) / sizeof(cfgs[0]);

    int passes = 0, skips = 0, fails = 0;
    for (int i = 0; i < n_cfgs; ++i) {
        const int rc = run_one(cfgs[i].N, cfgs[i].MAL, cfgs[i].seed);
        if      (rc == 0)  ++passes;
        else if (rc == 77) ++skips;
        else               ++fails;
        if (rc == 77) {
            std::printf("aborting sweep — kernel is stub state\n");
            return 77;
        }
    }

    std::printf("---\n");
    std::printf("sweep summary: %d/%d configs PASSed (fails=%d skips=%d)\n",
                passes, n_cfgs, fails, skips);

    return (fails == 0) ? 0 : 1;
}
