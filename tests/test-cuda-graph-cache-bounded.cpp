// Bound on ggml_backend_cuda_context::cuda_graphs growth.
//
// Today, ggml/src/ggml-cuda.cu:4247-4252 unconditionally inserts a new
// std::unique_ptr<ggml_cuda_graph> into ctx.cuda_graphs whenever a new
// cgraph topology key is seen, with no eviction. Real workloads with
// shape-varying decode (multi-slot, MTP, prefill vs decode) drive
// distinct keys for every iteration; the cache grows monotonically,
// each entry holding a cudaGraphExec instance, and eventually the
// device runs out of memory at cudaGraphInstantiate.
//
// This test runs N decode shapes through the backend and asserts the
// graph cache does not exceed MAX. It is RED today; landing an LRU
// (or any bounded eviction) on the cache turns it GREEN.
//
// Build: gated on GGML_CUDA. Drives only ggml-level graphs (no llama
// layers required), so it runs in seconds.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// We force a small cache cap via env, then drive enough distinct
// shapes to exceed it. Without the cap (legacy code) the cache grows
// to N_DISTINCT_SHAPES; with the cap it must stay at MAX_CACHE_ENTRIES.
constexpr size_t MAX_CACHE_ENTRIES = 8;
constexpr int    N_DISTINCT_SHAPES = 32;

static bool run_one_shape(ggml_backend_t backend, int64_t n) {
    static const size_t mem_size = 4 * 1024 * 1024;
    struct ggml_init_params params = { mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, n);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, n);
    ggml_tensor * out = ggml_add(ctx, a, b);
    ggml_set_name(out, "out");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }

    std::vector<float> data((size_t) 32 * n, 1.0f);
    ggml_backend_tensor_set(a, data.data(), 0, data.size() * sizeof(float));
    ggml_backend_tensor_set(b, data.data(), 0, data.size() * sizeof(float));

    const auto status = ggml_backend_graph_compute(backend, gf);
    const bool ok = (status == GGML_STATUS_SUCCESS);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

int main() {
    // Force a small cap so we can demonstrate the bound with cheap test
    // workload. Production default is 128 (see ggml-cuda.cu).
    char cap_env[32];
    snprintf(cap_env, sizeof(cap_env), "%zu", MAX_CACHE_ENTRIES);
    setenv("GGML_CUDA_GRAPH_MAX", cap_env, 1);

    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed; skipping\n");
        return 0;
    }

    // Drive N distinct shapes (each n triggers a new cgraph topology
    // → new cuda_graphs key). A correctly bounded cache should hold at
    // most MAX_CACHE_ENTRIES afterward.
    for (int i = 0; i < N_DISTINCT_SHAPES; ++i) {
        const int64_t n = 8 + i;
        if (!run_one_shape(backend, n)) {
            fprintf(stderr, "shape n=%lld failed\n", (long long) n);
            ggml_backend_free(backend);
            return 1;
        }
    }

    const size_t cache_size = ggml_backend_cuda_graph_cache_size(backend);
    printf("=== test-cuda-graph-cache-bounded ===\n");
    printf("  drove %d distinct shapes\n", N_DISTINCT_SHAPES);
    printf("  cache_size = %zu (max allowed: %zu)\n", cache_size, MAX_CACHE_ENTRIES);

    ggml_backend_free(backend);

    if (cache_size > MAX_CACHE_ENTRIES) {
        printf("RESULT: FAIL (graph cache grows unbounded — see ggml-cuda.cu:4247)\n");
        return 1;
    }
    printf("RESULT: PASS\n");
    return 0;
}
