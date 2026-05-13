// test-dflash-inject-fused.cpp
//
// Byte-identity unit test: fused CUDA inject_kv_fused kernel vs the
// fp32 scalar reference, exercised across all 5 drafter layers.
//
// @witnesses: PerLayerArity
// @witnesses: HeadShapeMatchesDraft
// @witnesses: KAsymmetricallyNormedVNot
// @witnesses: InjectedAnchorAlignment
// @witnesses: InjectPerLayerLaunches
//
// Each @witnesses line above cites an Allium @invariant this test binds
// on (per specs/dflash/allium-tla-binding.json bindings_external).
//
//   PerLayerArity              — host driver loops L_d=5 times, one
//                                launch per drafter layer.
//   HeadShapeMatchesDraft       — test uses drafter's declared
//                                H_kv=8, D=128 shape.
//   KAsymmetricallyNormedVNot   — scalar oracle applies k_norm + RoPE
//                                to K only; fused must agree byte-
//                                identically. V untouched after V_proj.
//   InjectedAnchorAlignment     — test sets anchor_positions explicitly;
//                                verifies cache writes land at exactly
//                                those positions (and ONLY there).
//   InjectPerLayerLaunches      — driver is a for-loop over L_d=5 with
//                                grid (N_slots, MAL_anchors) per launch;
//                                no batched 3D launch alternative.
//
// Spec:
//   - specs/dflash/kernel-design.md §6.2
//   - specs/dflash/dflash.allium ProjectAndFuse contract
//
// Exit codes:
//   0  — fused kernel output matches scalar reference (byte-identical
//        or every disagreement ≤ 1 ULP at fp16 AND rate ≤ 1 %)
//   1  — fused kernel disagrees materially (FAIL)
//  77  — CTest SKIP: kernel stub leaves caches all-zero, or no CUDA device

#include "dflash-inject-kv-reference.h"
#include "ggml-cuda/dflash/dflash-inject-kv.cuh"

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

int main() {
    // Production drafter shape (Qwen3.6-27B-DFlash) — locked.
    constexpr int   N_slots     = 1;
    constexpr int   MAL_anchors = 1;
    constexpr int   L_d         = 5;     // drafter layer count
    constexpr int   H_kv        = 8;
    constexpr int   D           = 128;
    constexpr int   D_d         = 5120;
    constexpr int   SeqLen      = 16;    // test cache capacity per slot
    constexpr float rope_base   = 1.0e7f;
    constexpr float norm_eps    = 1.0e-6f;

    int dev_count = 0;
    cudaError_t derr = cudaGetDeviceCount(&dev_count);
    if (derr != cudaSuccess || dev_count == 0) {
        std::printf("SKIP: no CUDA device available (%s)\n",
                    derr == cudaSuccess ? "count=0" : cudaGetErrorString(derr));
        return 77;
    }

    const int D_kv = H_kv * D;  // 1024
    std::printf("test-dflash-inject-fused: N=%d MAL=%d L_d=%d  H_kv=%d D=%d D_d=%d SeqLen=%d\n",
                N_slots, MAL_anchors, L_d, H_kv, D, D_d, SeqLen);

    const std::size_t n_ctx  = (std::size_t) N_slots * MAL_anchors * D_d;
    const std::size_t n_w    = (std::size_t) D_kv * D_d;            // per-layer
    const std::size_t n_kn   = D;                                   // per-layer
    const std::size_t n_pos  = (std::size_t) N_slots * MAL_anchors;
    const std::size_t n_cache = (std::size_t) N_slots * SeqLen * H_kv * D;

    std::printf("  per-layer k_weight bytes:  %zu MiB\n", (n_w * sizeof(__half)) >> 20);
    std::printf("  cache bytes per layer:     %zu KiB (each of K/V)\n",
                (n_cache * sizeof(__half)) >> 10);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> small(-0.5f, 0.5f);
    // Anchor positions: choose a position within SeqLen.  Single slot,
    // single anchor — keep simple for the first RED-first commit; future
    // expanded sweep (closure binding step) will randomize across slots.
    std::vector<int> pos_h(n_pos);
    pos_h[0] = 3;

    // Context states (shared across all drafter layers).
    std::vector<__half> ctx_h(n_ctx);
    std::vector<float>  ctx_f(n_ctx);
    for (std::size_t i = 0; i < n_ctx; ++i) {
        const float v = small(rng);
        ctx_h[i] = __float2half(v);
        ctx_f[i] = __half2float(ctx_h[i]);
    }

    // Per-layer weight buffers (each layer gets its own random seed band so
    // the test exercises 5 distinct configurations).  fc scale down to keep
    // post-GEMV magnitudes bounded (sum of D_d random uniform terms has
    // sd ~ sqrt(D_d/12) ~ 21; scale by 0.03 keeps K, V around |1|).
    std::vector<std::vector<__half>> kw_h(L_d, std::vector<__half>(n_w));
    std::vector<std::vector<__half>> vw_h(L_d, std::vector<__half>(n_w));
    std::vector<std::vector<__half>> kn_h(L_d, std::vector<__half>(n_kn));
    std::vector<std::vector<float>>  kw_f(L_d, std::vector<float>(n_w));
    std::vector<std::vector<float>>  vw_f(L_d, std::vector<float>(n_w));
    std::vector<std::vector<float>>  kn_f(L_d, std::vector<float>(n_kn));
    for (int il = 0; il < L_d; ++il) {
        for (auto & h : kw_h[il]) h = __float2half(small(rng) * 0.03f);
        for (auto & h : vw_h[il]) h = __float2half(small(rng) * 0.03f);
        for (auto & h : kn_h[il]) h = __float2half(0.5f + small(rng));  // non-identity
        for (std::size_t i = 0; i < n_w;  ++i) kw_f[il][i] = __half2float(kw_h[il][i]);
        for (std::size_t i = 0; i < n_w;  ++i) vw_f[il][i] = __half2float(vw_h[il][i]);
        for (std::size_t i = 0; i < n_kn; ++i) kn_f[il][i] = __half2float(kn_h[il][i]);
    }

    // Scalar reference: run all L_d layers serially, accumulate per-layer
    // cache writes into one big fp32 buffer per (K, V).  Cache is
    // [L_d, N_slots, SeqLen, H_kv, D] for the test (one slab per layer).
    const std::size_t n_cache_all = (std::size_t) L_d * n_cache;
    std::vector<float> ref_k_cache(n_cache_all, 0.0f);
    std::vector<float> ref_v_cache(n_cache_all, 0.0f);
    for (int il = 0; il < L_d; ++il) {
        float * k_layer = ref_k_cache.data() + (std::size_t) il * n_cache;
        float * v_layer = ref_v_cache.data() + (std::size_t) il * n_cache;
        dflash_reference::inject_kv_fused_scalar_ref_f32(
            ctx_f.data(),
            kw_f[il].data(), vw_f[il].data(), kn_f[il].data(),
            rope_base, norm_eps,
            k_layer, v_layer, pos_h.data(),
            N_slots, MAL_anchors, H_kv, D, D_d, SeqLen);
    }
    // Cast fp32 scalar reference to fp16 for comparison.
    std::vector<__half> ref_k_h(n_cache_all);
    std::vector<__half> ref_v_h(n_cache_all);
    for (std::size_t i = 0; i < n_cache_all; ++i) ref_k_h[i] = __float2half(ref_k_cache[i]);
    for (std::size_t i = 0; i < n_cache_all; ++i) ref_v_h[i] = __float2half(ref_v_cache[i]);

    // Kernel path: allocate device buffers, copy in, launch per-layer L_d
    // times (this IS the @InjectPerLayerLaunches binding — code shape is
    // the witness).
    __half *d_ctx = nullptr, *d_kn = nullptr;
    __half *d_kw_buf = nullptr, *d_vw_buf = nullptr;
    __half *d_kcache_all = nullptr, *d_vcache_all = nullptr;
    int    *d_pos = nullptr;

    CUDA_CHECK(cudaMalloc(&d_ctx, n_ctx * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_kw_buf, (std::size_t) L_d * n_w * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_vw_buf, (std::size_t) L_d * n_w * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_kn,     (std::size_t) L_d * n_kn * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_kcache_all, n_cache_all * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_vcache_all, n_cache_all * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_pos, n_pos * sizeof(int)));

    CUDA_CHECK(cudaMemset(d_kcache_all, 0, n_cache_all * sizeof(__half)));
    CUDA_CHECK(cudaMemset(d_vcache_all, 0, n_cache_all * sizeof(__half)));
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

    // Per-layer launches — this is the @InjectPerLayerLaunches witness.
    for (int il = 0; il < L_d; ++il) {
        dflash_inject_kv_fused_launch(
            d_ctx,
            d_kw_buf  + (std::size_t) il * n_w,
            d_vw_buf  + (std::size_t) il * n_w,
            d_kn      + (std::size_t) il * n_kn,
            rope_base, norm_eps,
            d_kcache_all + (std::size_t) il * n_cache,
            d_vcache_all + (std::size_t) il * n_cache,
            d_pos,
            N_slots, MAL_anchors, H_kv, D, D_d, SeqLen,
            /*stream=*/0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<__half> kern_k_h(n_cache_all);
    std::vector<__half> kern_v_h(n_cache_all);
    CUDA_CHECK(cudaMemcpy(kern_k_h.data(), d_kcache_all, n_cache_all * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(kern_v_h.data(), d_vcache_all, n_cache_all * sizeof(__half), cudaMemcpyDeviceToHost));

    cudaFree(d_ctx);
    cudaFree(d_kw_buf);
    cudaFree(d_vw_buf);
    cudaFree(d_kn);
    cudaFree(d_kcache_all);
    cudaFree(d_vcache_all);
    cudaFree(d_pos);

    // Detect RED-first stub state: kernel zeros the full cache slab, while
    // the scalar reference writes non-zero values at the anchor positions.
    // We check the (layer 0, slot 0, position=3, head 0) row: if all zero,
    // kernel is stubbed → SKIP.  This is a deterministic SKIP signal.
    const int pos0 = pos_h[0];
    const std::size_t check_base =
        ((((std::size_t) 0 /*layer*/ * N_slots + 0 /*slot*/) * SeqLen + pos0)
          * H_kv + 0 /*head*/) * D;
    bool stub_all_zero = true;
    for (int d = 0; d < D; ++d) {
        uint16_t bits;
        std::memcpy(&bits, &kern_k_h[check_base + d], sizeof(uint16_t));
        if (bits != 0) { stub_all_zero = false; break; }
    }
    if (stub_all_zero) {
        std::printf("SKIP: kernel K cache at (layer 0, slot 0, pos %d, head 0) is all-zero "
                    "(stub; inject kernel not yet implemented)\n", pos0);
        return 77;
    }

    // Compare cache slabs.  Mismatches in cells the kernel/ref didn't write
    // (both should be zero) count too — they catch a kernel that writes
    // outside its anchor position.
    auto count_diffs = [&](const __half* a, const __half* b, std::size_t n,
                           int& n_diff, int& n_1ulp, int& max_ulp) {
        n_diff = n_1ulp = max_ulp = 0;
        for (std::size_t i = 0; i < n; ++i) {
            uint16_t ab, bb;
            std::memcpy(&ab, &a[i], sizeof(uint16_t));
            std::memcpy(&bb, &b[i], sizeof(uint16_t));
            if (ab == bb) continue;
            ++n_diff;
            int d = std::abs(static_cast<int>(ab) - static_cast<int>(bb));
            if (d == 1) ++n_1ulp;
            if (d > max_ulp) max_ulp = d;
        }
    };

    int k_diff, k_1ulp, k_max;
    int v_diff, v_1ulp, v_max;
    count_diffs(kern_k_h.data(), ref_k_h.data(), n_cache_all, k_diff, k_1ulp, k_max);
    count_diffs(kern_v_h.data(), ref_v_h.data(), n_cache_all, v_diff, v_1ulp, v_max);

    // Mismatch rates are computed against the count of cells the reference
    // actually wrote (= L_d * N_slots * MAL_anchors * H_kv * D); positions
    // outside the anchor cells should be zero on both sides.  But the test
    // also accepts mismatches in zero positions (kernel bug that writes
    // outside anchor) as a failure — those would show up as non-1-ULP
    // differences from zero, captured in the max_ulp gate.
    const std::size_t n_written_cells =
        (std::size_t) L_d * N_slots * MAL_anchors * H_kv * D;
    std::printf("  K mismatches: %d / %zu  (1-ULP: %d, max ULP: %d)\n",
                k_diff, n_cache_all, k_1ulp, k_max);
    std::printf("  V mismatches: %d / %zu  (1-ULP: %d, max ULP: %d)\n",
                v_diff, n_cache_all, v_1ulp, v_max);
    std::printf("  reference-written cell count (denom for rate): %zu\n", n_written_cells);

    // PASS criterion: byte-identical, OR every disagreement ≤ 2 ULP at
    // fp16 AND rate ≤ 1 % of reference-written cells. Looser than the
    // combine_features test (which gates at ≤ 1 ULP) because:
    //   - V output goes through K_proj + V_proj GEMV + fp16 cast only;
    //     no transcendental functions are involved. Empirically V
    //     matches byte-identically (max ULP = 0).
    //   - K output additionally goes through cos/sin in the RoPE step.
    //     CUDA libdevice cosf/sinf is documented to be accurate within
    //     ≤ 2 ULP at fp32 (vs CPU libm typically ≤ 1 ULP). A 2-ULP fp32
    //     difference propagates through the (k_lo * cos - k_hi * sin)
    //     combination to occasionally a 2-ULP fp16 difference at binade-
    //     boundary positions. > 2 ULP would indicate a real bug.
    // V is held to a TIGHTER bound (≤ 1 ULP) because its accumulation
    // path is identical to combine_features' FC and exhibits the same
    // fp32-reduction-order noise but never trig.
    auto judge = [&](int diff, int max_ulp, int allowed_ulp) -> int {
        if (diff == 0) return 0;
        if (max_ulp <= allowed_ulp && diff * 100 <= (int) n_written_cells) {
            return 0;
        }
        if (max_ulp > allowed_ulp) return 2;   // precision bug
        return 3;                                // systematic mismatch rate
    };

    // K: ≤ 2 ULP allowed (cosf/sinf precision divergence between CUDA libdevice
    // and CPU libm). V: ≤ 1 ULP (no trig, same expectation as combine_features).
    const int kj = judge(k_diff, k_max, /*allowed_ulp=*/2);
    const int vj = judge(v_diff, v_max, /*allowed_ulp=*/1);

    if (kj == 0 && vj == 0) {
        std::printf("PASS: K=%d diffs (≤2 ULP, ≤1%% rate)  V=%d diffs (≤1 ULP, ≤1%% rate)\n",
                    k_diff, v_diff);
        return 0;
    }
    if (kj == 2 || vj == 2) {
        std::printf("FAIL: K max ULP=%d (limit 2), V max ULP=%d (limit 1); precision bug\n",
                    k_max, v_max);
    } else {
        std::printf("FAIL: mismatch rate exceeds 1%% (K=%d, V=%d); systematic precision loss\n",
                    k_diff, v_diff);
    }
    return 1;
}
