// Regression test for the delta-net fast/slow path divergence.
//
// Background: `ggml_compute_forward_delta_net_f32` (ggml.c) gates the IQK
// fused fast path on `emit_intermediates == 0`. On Vulkan, pipeline
// selection also differs based on `state_inplace` (which is tied to
// emit_intermediates being 0). These two paths are semantically equivalent
// but produce different floating-point outputs. Mixing paths at batch=1
// (emit=0) vs batch>1 (emit=1) causes the SSM trajectory to drift over
// many cycles — this was the residual 1-2 token divergence in 35B MTP
// speculation.
//
// Fix landed in llama-delta-net.cpp: force emit_intermediates=true
// unconditionally. This test locks in that the two code paths agree —
// so if anyone reverts the fix or introduces a new variant-split, this
// test fires.
//
// Test harness: construct a tiny delta-net graph (no model load), run with
// emit_intermediates=0 and emit_intermediates=1 on the same backend,
// compare the OUTPUT TOKENS portion (= first output_size floats of the
// result) byte-identical. Intermediates are NOT compared (emit=0 doesn't
// produce them).
//
// Usage:
//   test-delta-net-emit-path-invariance
// (no -m needed — backend inferred from environment / backend registry.)

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <vector>

static void fill_uniform(float * data, size_t n, float lo, float hi, uint32_t seed) {
    // Deterministic seed-based fill — avoid rand()/mt19937 so results are
    // reproducible across libstdc++ versions.
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        const float u = (float)s / (float)0xFFFFFFFFu;
        data[i] = lo + (hi - lo) * u;
    }
}

static bool run_case(ggml_backend_t backend, int64_t head_dim, int64_t n_tokens,
                     int64_t H_v, int64_t gqa_ratio, int64_t n_seqs,
                     bool emit_intermediates, std::vector<float> & out) {
    const int64_t H_k = H_v / gqa_ratio;

    struct ggml_init_params gip = { /*mem_size*/ 1 << 22, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    struct ggml_context * ctx = ggml_init(gip);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return false; }

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_tokens, H_k, n_seqs);
    ggml_set_name(q, "q");
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_tokens, H_k, n_seqs);
    ggml_set_name(k, "k");

    // v permute to [head_dim, n_tokens, H_v, n_seqs] pattern (mirrors production).
    ggml_tensor * v_pre = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, H_v, n_tokens, n_seqs);
    ggml_set_name(v_pre, "v_pre");
    ggml_tensor * v = ggml_permute(ctx, v_pre, 0, 2, 1, 3);

    ggml_tensor * g_pre = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H_v, n_tokens, n_seqs);
    ggml_set_name(g_pre, "g_pre");
    ggml_tensor * g = ggml_permute(ctx, g_pre, 2, 0, 3, 1);

    ggml_tensor * beta_pre = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, H_v, 1, n_tokens, n_seqs);
    ggml_set_name(beta_pre, "beta_pre");
    ggml_tensor * beta = ggml_permute(ctx, beta_pre, 2, 0, 1, 3);

    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, head_dim * H_v, 1, n_seqs);
    ggml_set_name(state, "state");

    ggml_tensor * outT = ggml_delta_net_ext(ctx, q, k, v, g, beta, state, emit_intermediates);
    ggml_set_name(outT, "out");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, outT);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "backend buffer alloc failed\n"); ggml_free(ctx); return false; }

    auto fill = [&](ggml_tensor * t, float lo, float hi, uint32_t seed) {
        const size_t n = ggml_nelements(t);
        std::vector<float> h(n);
        fill_uniform(h.data(), n, lo, hi, seed);
        ggml_backend_tensor_set(t, h.data(), 0, n * sizeof(float));
    };
    fill(q,        -0.5f,  0.5f, 0xA);
    fill(k,        -0.5f,  0.5f, 0xB);
    fill(v_pre,    -0.5f,  0.5f, 0xC);
    fill(g_pre,    -2.0f, -0.1f, 0xD);  // keep decay bounded
    fill(beta_pre, -1.0f,  1.0f, 0xE);
    fill(state,    -0.1f,  0.1f, 0xF);

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga, gf)) {
        fprintf(stderr, "graph alloc failed\n");
        ggml_gallocr_free(ga);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph compute failed\n");
        ggml_gallocr_free(ga);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    // Read back OUTPUT TOKENS portion only — first S_v*H_v*n_tokens*n_seqs
    // floats of outT. The layout after that is implementation-dependent
    // (emit=0 writes 1 state block, emit=1 writes n_tokens state blocks).
    const size_t out_tokens_size = (size_t)(head_dim * H_v * n_tokens * n_seqs);
    out.resize(out_tokens_size);
    ggml_backend_tensor_get(outT, out.data(), 0, out_tokens_size * sizeof(float));

    ggml_gallocr_free(ga);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main() {
    ggml_backend_t backend = nullptr;

    const char * env_backend = std::getenv("GGML_BACKEND");
#ifdef GGML_USE_VULKAN
    if (!backend && (!env_backend || std::strncmp(env_backend, "Vulkan", 6) == 0)) {
        backend = ggml_backend_vk_init(0);
    }
#endif
    if (!backend) backend = ggml_backend_cpu_init();
    if (!backend) { fprintf(stderr, "no backend available\n"); return 2; }

    fprintf(stderr, "Backend: %s\n", ggml_backend_name(backend));

    struct test_spec {
        int64_t head_dim, n_tokens, H_v, gqa_ratio, n_seqs;
    };
    const std::vector<test_spec> specs = {
        {64,  1, 8, 2, 1},   // batch=1 token-gen shape
        {64,  2, 8, 2, 1},   // batch=2 verify shape
        {64,  4, 8, 2, 1},   // batch=4
        {128, 1, 8, 2, 1},   // h128 tg
        {128, 2, 8, 2, 1},   // h128 verify
    };

    int fails = 0;
    printf("\n%-4s %-4s %-4s %-4s %-4s %-18s %-18s %s\n",
           "hd", "tok", "Hv", "gqa", "ns", "emit=0 sum", "emit=1 sum", "match");
    for (const auto & s : specs) {
        std::vector<float> o0, o1;
        if (!run_case(backend, s.head_dim, s.n_tokens, s.H_v, s.gqa_ratio, s.n_seqs, /*emit=*/false, o0)) {
            printf("%-4ld %-4ld %-4ld %-4ld %-4ld emit=0 RUN_FAIL\n",
                   (long)s.head_dim, (long)s.n_tokens, (long)s.H_v, (long)s.gqa_ratio, (long)s.n_seqs);
            fails++;
            continue;
        }
        if (!run_case(backend, s.head_dim, s.n_tokens, s.H_v, s.gqa_ratio, s.n_seqs, /*emit=*/true, o1)) {
            printf("%-4ld %-4ld %-4ld %-4ld %-4ld emit=1 RUN_FAIL\n",
                   (long)s.head_dim, (long)s.n_tokens, (long)s.H_v, (long)s.gqa_ratio, (long)s.n_seqs);
            fails++;
            continue;
        }
        if (o0.size() != o1.size()) { fprintf(stderr, "size mismatch\n"); fails++; continue; }

        double s0 = 0.0, s1 = 0.0, max_abs_diff = 0.0;
        bool byte_identical = true;
        for (size_t i = 0; i < o0.size(); i++) {
            s0 += o0[i]; s1 += o1[i];
            const double d = std::fabs((double)o0[i] - (double)o1[i]);
            if (d > max_abs_diff) max_abs_diff = d;
            if (o0[i] != o1[i]) byte_identical = false;
        }
        const bool pass = byte_identical;
        if (!pass) fails++;
        printf("%-4ld %-4ld %-4ld %-4ld %-4ld %-18.6g %-18.6g %s (max|Δ|=%.3g)\n",
               (long)s.head_dim, (long)s.n_tokens, (long)s.H_v, (long)s.gqa_ratio, (long)s.n_seqs,
               s0, s1, pass ? "SAME" : "DIFF", max_abs_diff);
    }

    printf("\nfails = %d / %d\n", fails, (int)specs.size());
    printf("%s\n", fails == 0 ? "PASS" : "FAIL");

    ggml_backend_free(backend);
    return fails == 0 ? 0 : 1;
}
