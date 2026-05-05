// CUDA graph cache: comparator strictness on op_params + tensor dtype.
//
// Drives two graphs that share (n_nodes, op, ne) — same topology key
// AND same shape key under the current FNV-1a hashes — but differ in
// element-tensor dtype:
//
//   graph1: ggml_add(F32, F32) → F32
//   graph2: ggml_add(F16, F16) → F16
//
// Today (pre-Item 8.2): the comparator at
// `ggml_graph_node_has_matching_properties` does NOT include
// `node->type` or `src[i]->type` in the strict-check list. Result:
// graph2 hits the cache entry instantiated for graph1, the captured
// kernel was selected for F32 inputs, and submitting F16 bytes
// produces silent corruption of the output. RED.
//
// Post-Item 8.2: the comparator strict-checks node and src dtypes.
// graph2 sees a mismatch, takes the re-instantiate path, gets the
// F16-correct kernel, and produces matching output. GREEN.
//
// This is the smallest plausible reproducer for the suspected
// PHASE 33 concat.cu:202 GGML_ASSERT(src0->type==src1->type==dst->type)
// failure mode under multi-slot.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int64_t N = 16;

static bool run_f32_graph(ggml_backend_t backend, std::vector<float> & out) {
    static const size_t mem_size = 4 * 1024 * 1024;
    struct ggml_init_params params = { mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, N);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, N);
    ggml_tensor * c = ggml_add(ctx, a, b);
    ggml_set_name(c, "out");
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }
    std::vector<float> af((size_t) 32 * N, 1.0f);
    std::vector<float> bf((size_t) 32 * N, 2.0f);
    ggml_backend_tensor_set(a, af.data(), 0, af.size() * sizeof(float));
    ggml_backend_tensor_set(b, bf.data(), 0, bf.size() * sizeof(float));
    const auto status = ggml_backend_graph_compute(backend, gf);
    bool ok = (status == GGML_STATUS_SUCCESS);
    if (ok) {
        out.assign(af.size(), 0.0f);
        ggml_backend_tensor_get(c, out.data(), 0, out.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

static bool run_f16_graph(ggml_backend_t backend, std::vector<float> & out) {
    static const size_t mem_size = 4 * 1024 * 1024;
    struct ggml_init_params params = { mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, N);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 32, N);
    ggml_tensor * c = ggml_add(ctx, a, b);
    ggml_set_name(c, "out");
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }
    std::vector<ggml_fp16_t> ah((size_t) 32 * N), bh((size_t) 32 * N);
    for (size_t i = 0; i < ah.size(); ++i) {
        ah[i] = ggml_fp32_to_fp16(1.0f);
        bh[i] = ggml_fp32_to_fp16(2.0f);
    }
    ggml_backend_tensor_set(a, ah.data(), 0, ah.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(b, bh.data(), 0, bh.size() * sizeof(ggml_fp16_t));
    const auto status = ggml_backend_graph_compute(backend, gf);
    bool ok = (status == GGML_STATUS_SUCCESS);
    if (ok) {
        std::vector<ggml_fp16_t> outh(ah.size());
        ggml_backend_tensor_get(c, outh.data(), 0, outh.size() * sizeof(ggml_fp16_t));
        out.assign(outh.size(), 0.0f);
        for (size_t i = 0; i < outh.size(); ++i) {
            out[i] = ggml_fp16_to_fp32(outh[i]);
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

int main() {
    setenv("GGML_CUDA_GRAPH_MAX", "256", 1);

    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed; skipping\n");
        return 0;
    }

    std::vector<float> out_f32, out_f16;

    if (!run_f32_graph(backend, out_f32)) {
        fprintf(stderr, "F32 graph failed\n");
        ggml_backend_free(backend);
        return 1;
    }
    if (!run_f16_graph(backend, out_f16)) {
        fprintf(stderr, "F16 graph failed\n");
        ggml_backend_free(backend);
        return 1;
    }

    ggml_backend_free(backend);

    printf("=== test-cuda-graph-comparator-op-params ===\n");
    printf("  ran F32 graph then F16 graph (same op + ne; differ in dtype)\n");

    // Both graphs compute (1+2) elementwise, so the F16 result should
    // round-trip to ~3.0 in float. If the cache miss-routed and the
    // F16 submission ran the F32 kernel against F16 input bytes, the
    // bytes are reinterpreted and produce nonsense values (e.g.,
    // FP16 bit pattern 0x3c00 = 1.0 reinterpreted as the low 16 bits
    // of an FP32 = a denormal, output ≈ 0 or NaN).

    int n_bad = 0;
    for (size_t i = 0; i < out_f16.size(); ++i) {
        const float v = out_f16[i];
        if (!std::isfinite(v) || std::fabs(v - 3.0f) > 1e-2f) {
            if (n_bad < 4) {
                fprintf(stderr, "  F16[%zu] = %.6g (expected ~3.0)\n", i, v);
            }
            n_bad++;
        }
    }
    if (n_bad) {
        printf("RESULT: FAIL — %d F16 outputs deviated from expected ~3.0 (likely cache miss-routing)\n", n_bad);
        return 1;
    }
    printf("RESULT: PASS — F16 output matches expected ~3.0; comparator distinguishes dtypes\n");
    return 0;
}
