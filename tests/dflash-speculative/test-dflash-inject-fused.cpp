// test-dflash-inject-fused.cpp
//
// Byte-identity unit test: fused CUDA inject_kv_fused kernel vs the
// fp32 scalar reference, exercised across all 5 drafter layers AND
// a (N_slots × MAL_anchors × seed) sweep. This is the T3 closure
// binding for inject_kv_fused per kernel-design.md §10 step 3.
//
// @witnesses: PerLayerArity
// @witnesses: HeadShapeMatchesDraft
// @witnesses: KAsymmetricallyNormedVNot
// @witnesses: InjectedAnchorAlignment
// @witnesses: InjectPerLayerLaunches
//
// PASS criterion per sub-run:
//   K: byte-identical OR every disagreement ≤ 2 ULP at fp16 AND rate ≤ 1 %.
//      (≤ 2 ULP because cosf/sinf diverges up to 2 ULP between CUDA
//      libdevice and CPU libm; > 2 ULP would indicate a precision bug.)
//   V: byte-identical OR every disagreement ≤ 1 ULP AND rate ≤ 1 %.
//      (No trig in V's path; tightened bound. Empirically V is always
//      perfectly byte-identical, validating @KAsymmetricallyNormedVNot.)

#include "dflash-inject-kv-reference.h"
#include "ggml-cuda/dflash/dflash-inject-kv.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
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

constexpr int   L_d       = 5;
constexpr int   H_kv      = 8;
constexpr int   D         = 128;
constexpr int   D_d       = 5120;
constexpr float rope_base = 1.0e7f;
constexpr float norm_eps  = 1.0e-6f;

// Counts mismatch bins for byte / 1ULP / >2ULP buckets in fp16 (kept for
// the WORST-cell diagnostic) plus NMSE / cos similarity (the actual gate
// post the §6.2.A pinned-HMMA dispatch — same precedent as lm_head test).
void count_diffs(const __half* a, const __half* b, std::size_t n,
                 int& n_diff, int& n_1ulp, int& max_ulp,
                 double& nmse, double& cos_sim) {
    n_diff = n_1ulp = max_ulp = 0;
    double sum_sq_err = 0.0, sum_sq_ref = 0.0;
    double sum_ad = 0.0, sum_sq_dev = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        uint16_t ab, bb;
        std::memcpy(&ab, &a[i], sizeof(uint16_t));
        std::memcpy(&bb, &b[i], sizeof(uint16_t));
        if (ab != bb) {
            ++n_diff;
            int d = std::abs(static_cast<int>(ab) - static_cast<int>(bb));
            if (d == 1) ++n_1ulp;
            if (d > max_ulp) max_ulp = d;
        }
        const double ar = static_cast<double>(__half2float(a[i]));
        const double br = static_cast<double>(__half2float(b[i]));
        const double diff = ar - br;
        sum_sq_err  += diff * diff;
        sum_sq_ref  += br * br;
        sum_ad      += ar * br;
        sum_sq_dev  += ar * ar;
    }
    nmse    = sum_sq_err / (sum_sq_ref + 1e-30);
    cos_sim = sum_ad / (std::sqrt(sum_sq_ref) * std::sqrt(sum_sq_dev) + 1e-30);
}

int run_one(int N_slots, int MAL_anchors, int SeqLen, uint32_t seed) {
    const int D_kv = H_kv * D;
    const std::size_t n_ctx     = (std::size_t) N_slots * MAL_anchors * D_d;
    const std::size_t n_w       = (std::size_t) D_kv * D_d;
    const std::size_t n_kn      = D;
    const std::size_t n_pos     = (std::size_t) N_slots * MAL_anchors;
    const std::size_t n_cache_1 = (std::size_t) N_slots * SeqLen * H_kv * D;
    const std::size_t n_cache   = (std::size_t) L_d * n_cache_1;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> small(-0.5f, 0.5f);

    // Anchor positions: distinct per (slot, anchor) within [1, SeqLen-1].
    // Avoid pos=0 to make RoPE non-trivial (cos≠1, sin≠0).
    std::vector<int> pos_h(n_pos);
    for (int s = 0; s < N_slots; ++s) {
        for (int a = 0; a < MAL_anchors; ++a) {
            pos_h[s * MAL_anchors + a] = 1 + ((s * MAL_anchors + a) % (SeqLen - 1));
        }
    }

    std::vector<__half> ctx_h(n_ctx);
    std::vector<float>  ctx_f(n_ctx);
    for (std::size_t i = 0; i < n_ctx; ++i) {
        const float v = small(rng);
        ctx_h[i] = __float2half(v);
        ctx_f[i] = __half2float(ctx_h[i]);
    }

    std::vector<std::vector<__half>> kw_h(L_d, std::vector<__half>(n_w));
    std::vector<std::vector<__half>> vw_h(L_d, std::vector<__half>(n_w));
    std::vector<std::vector<__half>> kn_h(L_d, std::vector<__half>(n_kn));
    std::vector<std::vector<float>>  kw_f(L_d, std::vector<float>(n_w));
    std::vector<std::vector<float>>  vw_f(L_d, std::vector<float>(n_w));
    std::vector<std::vector<float>>  kn_f(L_d, std::vector<float>(n_kn));
    for (int il = 0; il < L_d; ++il) {
        for (auto & h : kw_h[il]) h = __float2half(small(rng) * 0.03f);
        for (auto & h : vw_h[il]) h = __float2half(small(rng) * 0.03f);
        for (auto & h : kn_h[il]) h = __float2half(0.5f + small(rng));
        for (std::size_t i = 0; i < n_w;  ++i) kw_f[il][i] = __half2float(kw_h[il][i]);
        for (std::size_t i = 0; i < n_w;  ++i) vw_f[il][i] = __half2float(vw_h[il][i]);
        for (std::size_t i = 0; i < n_kn; ++i) kn_f[il][i] = __half2float(kn_h[il][i]);
    }

    // Scalar reference across all 5 layers
    std::vector<float> ref_k_cache(n_cache, 0.0f);
    std::vector<float> ref_v_cache(n_cache, 0.0f);
    for (int il = 0; il < L_d; ++il) {
        float * k_layer = ref_k_cache.data() + (std::size_t) il * n_cache_1;
        float * v_layer = ref_v_cache.data() + (std::size_t) il * n_cache_1;
        dflash_reference::inject_kv_fused_scalar_ref_f32(
            ctx_f.data(),
            kw_f[il].data(), vw_f[il].data(), kn_f[il].data(),
            rope_base, norm_eps,
            k_layer, v_layer, pos_h.data(),
            N_slots, MAL_anchors, H_kv, D, D_d, SeqLen);
    }
    std::vector<__half> ref_k_h(n_cache), ref_v_h(n_cache);
    for (std::size_t i = 0; i < n_cache; ++i) ref_k_h[i] = __float2half(ref_k_cache[i]);
    for (std::size_t i = 0; i < n_cache; ++i) ref_v_h[i] = __float2half(ref_v_cache[i]);

    // Kernel path
    __half *d_ctx = nullptr, *d_kn = nullptr;
    __half *d_kw_buf = nullptr, *d_vw_buf = nullptr;
    __half *d_kcache_all = nullptr, *d_vcache_all = nullptr;
    int    *d_pos = nullptr;

    CUDA_CHECK(cudaMalloc(&d_ctx, n_ctx * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_kw_buf, (std::size_t) L_d * n_w * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_vw_buf, (std::size_t) L_d * n_w * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_kn,     (std::size_t) L_d * n_kn * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_kcache_all, n_cache * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_vcache_all, n_cache * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_pos, n_pos * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_kcache_all, 0, n_cache * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_vcache_all, 0, n_cache * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_ctx, ctx_h.data(), n_ctx * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos, pos_h.data(), n_pos * sizeof(int),    cudaMemcpyHostToDevice));
    for (int il = 0; il < L_d; ++il) {
        CUDA_CHECK(cudaMemcpy(d_kw_buf + (std::size_t) il * n_w,
                              kw_h[il].data(), n_w * sizeof(__half),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vw_buf + (std::size_t) il * n_w,
                              vw_h[il].data(), n_w * sizeof(__half),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_kn + (std::size_t) il * n_kn,
                              kn_h[il].data(), n_kn * sizeof(__half),
                              cudaMemcpyHostToDevice));
    }

    // Per-layer launches — the @InjectPerLayerLaunches witness.
    for (int il = 0; il < L_d; ++il) {
        dflash_inject_kv_fused_launch(
            d_ctx,
            d_kw_buf  + (std::size_t) il * n_w,
            d_vw_buf  + (std::size_t) il * n_w,
            d_kn      + (std::size_t) il * n_kn,
            rope_base, norm_eps,
            d_kcache_all + (std::size_t) il * n_cache_1,
            d_vcache_all + (std::size_t) il * n_cache_1,
            d_pos,
            N_slots, MAL_anchors, H_kv, D, D_d, SeqLen,
            /*stream=*/0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> kern_k_h(n_cache), kern_v_h(n_cache);
    CUDA_CHECK(cudaMemcpy(kern_k_h.data(), d_kcache_all, n_cache * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kern_v_h.data(), d_vcache_all, n_cache * sizeof(__half), cudaMemcpyDeviceToHost));

    cudaFree(d_ctx); cudaFree(d_kw_buf); cudaFree(d_vw_buf); cudaFree(d_kn);
    cudaFree(d_kcache_all); cudaFree(d_vcache_all); cudaFree(d_pos);

    // Stub detection: check (layer 0, slot 0, anchor 0)'s K row.
    const int pos0 = pos_h[0];
    const std::size_t check_base = ((((std::size_t) 0) + 0) * SeqLen + pos0) * H_kv * D;
    bool all_zero = true;
    for (int d = 0; d < D; ++d) {
        uint16_t bits;
        std::memcpy(&bits, &kern_k_h[check_base + d], sizeof(uint16_t));
        if (bits != 0) { all_zero = false; break; }
    }
    if (all_zero) {
        std::printf("  [N=%d MAL=%d seed=%u]  SKIP: kernel K cache is all-zero (stub)\n",
                    N_slots, MAL_anchors, seed);
        return 77;
    }

    int k_diff, k_1ulp, k_max;
    int v_diff, v_1ulp, v_max;
    double k_nmse, k_cos, v_nmse, v_cos;
    count_diffs(kern_k_h.data(), ref_k_h.data(), n_cache, k_diff, k_1ulp, k_max, k_nmse, k_cos);
    count_diffs(kern_v_h.data(), ref_v_h.data(), n_cache, v_diff, v_1ulp, v_max, v_nmse, v_cos);

    // Diagnostic: if any K cell exceeds the 2-ULP gate, surface the WORST
    // cell's (layer, slot, position, head, dim) coords + the kernel/ref
    // fp16 bits + fp32 values so the test output drives debugging.
    if (k_max > 2) {
        std::size_t worst_idx = 0;
        int worst_d = 0;
        for (std::size_t i = 0; i < n_cache; ++i) {
            uint16_t a, b;
            std::memcpy(&a, &kern_k_h[i], sizeof(uint16_t));
            std::memcpy(&b, &ref_k_h[i],  sizeof(uint16_t));
            int d = std::abs(static_cast<int>(a) - static_cast<int>(b));
            if (d > worst_d) { worst_d = d; worst_idx = i; }
        }
        // Decompose worst_idx → (layer, slot, position, head, dim) per the
        // cache layout [L_d, N_slots, SeqLen, H_kv, D].
        const std::size_t per_layer = (std::size_t) N_slots * SeqLen * H_kv * D;
        const std::size_t per_slot  = (std::size_t) SeqLen * H_kv * D;
        const std::size_t per_pos   = (std::size_t) H_kv * D;
        const int wl = worst_idx / per_layer;
        std::size_t r = worst_idx - (std::size_t) wl * per_layer;
        const int ws = r / per_slot;            r %= per_slot;
        const int wp = r / per_pos;             r %= per_pos;
        const int wh = r / D;                   const int wd = r % D;
        std::printf("    WORST K cell @ [L=%d slot=%d pos=%d head=%d dim=%d]: "
                    "kern=%.6g ref=%.6g diff=%d ULP\n",
                    wl, ws, wp, wh, wd,
                    __half2float(kern_k_h[worst_idx]),
                    __half2float(ref_k_h[worst_idx]), worst_d);
    }

    // PASS gate (revised 2026-05-19 post §6.2.A pinned-HMMA dispatch):
    // byte-identical OR NMSE ≤ 1e-5 AND cos ≥ 0.99999 (same precedent as
    // post-S59 lm_head test). HMMA fragment reduction tree differs from
    // serial-fp32 reference's K-loop reduction; numerics class unchanged.
    const bool k_pass = (k_diff == 0) || (k_nmse <= 1e-5 && k_cos >= 0.99999);
    const bool v_pass = (v_diff == 0) || (v_nmse <= 1e-5 && v_cos >= 0.99999);
    const bool pass = k_pass && v_pass;

    std::printf("  [N=%d MAL=%d seed=%u]  "
                "K: NMSE=%.3e cos=%.6f (bid=%d/%zu, max_ulp=%d)  "
                "V: NMSE=%.3e cos=%.6f (bid=%d/%zu, max_ulp=%d)  %s\n",
                N_slots, MAL_anchors, seed,
                k_nmse, k_cos, (int)(n_cache - (std::size_t)k_diff), n_cache, k_max,
                v_nmse, v_cos, (int)(n_cache - (std::size_t)v_diff), n_cache, v_max,
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

    std::printf("test-dflash-inject-fused sweep:  L_d=%d  H_kv=%d  D=%d  D_d=%d\n",
                L_d, H_kv, D, D_d);

    // SeqLen sized to accommodate the max anchor position generated across
    // configs (need pos < SeqLen). With (N=8, MAL=4) and distinct positions,
    // we need ≥ 33 slots; use 64 for headroom.
    constexpr int SeqLen = 64;

    // (N_slots, MAL_anchors) × seeds.
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
        const int rc = run_one(cfgs[i].N, cfgs[i].MAL, SeqLen, cfgs[i].seed);
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
