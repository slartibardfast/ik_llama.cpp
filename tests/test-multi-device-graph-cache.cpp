// test-multi-device-graph-cache.cpp
//
// PHASE_CUDA_NATIVE_DISPATCH C5 — bind the split end-capture / launch-
// exec / destroy-exec API used by the sched-layer outer-graph cache.
//
// The sched caches cudaGraphExec_t handles keyed by topology hash so
// repeat dispatches replay via cudaGraphLaunch without re-capturing or
// re-instantiating. This test exercises the low-level API directly:
//
//   1. Capture once: _begin → enqueue ops → _end_capture(&exec).
//   2. Launch the cached exec N times via _launch_exec — assert bit-
//      identical output across all N replays.
//   3. _destroy_exec — clean teardown.
//
// Specs bound:
//   /home/dconnolly/yarn-agentic/specs/cuda-native-dispatch/multi_device_graph_cache.allium
//   /home/dconnolly/yarn-agentic/specs/cuda-native-dispatch/CUDAGraphCacheConsistency.tla
//
// Returns: 0 = PASS, 1 = FAIL, 77 = SKIP (fewer than 2 CUDA devices).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define FAIL_AT(msg, ...) do { \
    std::fprintf(stderr, "FAIL %s:%d: " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    std::exit(1); \
} while (0)

static constexpr int N_REPLAYS = 50;
static constexpr int64_t N     = 64;

struct one_backend_graph {
    ggml_context * ctx           = nullptr;
    ggml_cgraph  * gf            = nullptr;
    ggml_tensor  * a             = nullptr;
    ggml_tensor  * b             = nullptr;
    ggml_tensor  * out           = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    ~one_backend_graph() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx)    ggml_free(ctx);
    }
};

static void build_one_backend_graph(ggml_backend_t backend, one_backend_graph & g) {
    static const size_t mem_size = 4 * 1024 * 1024;
    ggml_init_params p = { mem_size, nullptr, /*no_alloc=*/true };
    g.ctx = ggml_init(p);
    if (!g.ctx) FAIL_AT("ggml_init failed");
    g.a   = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N, N);
    g.b   = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, N, N);
    g.out = ggml_add(g.ctx, g.a, g.b);
    ggml_set_name(g.out, "out");
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    g.buffer = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    if (!g.buffer) FAIL_AT("ggml_backend_alloc_ctx_tensors failed");
}

static uint64_t fnv1a64(const void * data, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

int main() {
    const int n_cuda = ggml_backend_cuda_get_device_count();
    if (n_cuda < 2) {
        std::fprintf(stderr, "test-multi-device-graph-cache: SKIP (need 2 CUDA devices, have %d)\n", n_cuda);
        return 77;
    }
    std::fprintf(stdout, "test-multi-device-graph-cache: running with %d CUDA device(s)\n", n_cuda);

    ggml_backend_t cuda_a = ggml_backend_cuda_init(0, nullptr);
    ggml_backend_t cuda_b = ggml_backend_cuda_init(1, nullptr);
    if (!cuda_a || !cuda_b) FAIL_AT("ggml_backend_cuda_init failed");

    one_backend_graph ga, gb;
    build_one_backend_graph(cuda_a, ga);
    build_one_backend_graph(cuda_b, gb);

    std::vector<float> data_a((size_t) N * N, 1.0f);
    std::vector<float> data_b((size_t) N * N, 2.0f);

    // Pre-populate inputs once — the captured graph's memcpys will
    // re-read from these source pointers on every replay.
    ggml_backend_tensor_set(ga.a, data_a.data(), 0, data_a.size() * sizeof(float));
    ggml_backend_tensor_set(ga.b, data_b.data(), 0, data_b.size() * sizeof(float));
    ggml_backend_tensor_set(gb.a, data_a.data(), 0, data_a.size() * sizeof(float));
    ggml_backend_tensor_set(gb.b, data_b.data(), 0, data_b.size() * sizeof(float));
    ggml_backend_synchronize(cuda_a);
    ggml_backend_synchronize(cuda_b);

    ggml_cuda_outer_capture_count_reset();

    // ===== Capture phase =====
    ggml_backend_t secondaries[] = { cuda_b };
    if (!ggml_cuda_outer_capture_begin(cuda_a, secondaries, 1)) {
        FAIL_AT("outer_capture_begin returned false");
    }
    if (ggml_backend_graph_compute_async(cuda_a, ga.gf) != GGML_STATUS_SUCCESS) {
        FAIL_AT("graph_compute_async on cuda_a failed");
    }
    if (ggml_backend_graph_compute_async(cuda_b, gb.gf) != GGML_STATUS_SUCCESS) {
        FAIL_AT("graph_compute_async on cuda_b failed");
    }
    ggml_cuda_graph_exec_t exec = nullptr;
    if (!ggml_cuda_outer_capture_end_capture(cuda_a, secondaries, 1, &exec)) {
        FAIL_AT("outer_capture_end_capture returned false");
    }
    if (exec == nullptr) FAIL_AT("end_capture produced null exec");
    std::fprintf(stdout, "  captured exec handle = %p\n", exec);

    // ===== Replay phase =====
    std::vector<float>    out_host(N * N);
    std::vector<uint64_t> hashes(N_REPLAYS);
    for (int iter = 0; iter < N_REPLAYS; ++iter) {
        if (!ggml_cuda_outer_capture_launch_exec(cuda_a, exec)) {
            FAIL_AT("launch_exec returned false at iter %d", iter);
        }
        ggml_backend_synchronize(cuda_a);
        ggml_backend_synchronize(cuda_b);

        ggml_backend_tensor_get(ga.out, out_host.data(), 0, out_host.size() * sizeof(float));
        hashes[iter] = fnv1a64(out_host.data(), out_host.size() * sizeof(float));
    }

    const size_t count = ggml_cuda_outer_capture_count();
    std::fprintf(stdout, "  outer_capture_count = %zu (expected %d)\n", count, N_REPLAYS);
    if (count != (size_t) N_REPLAYS) {
        FAIL_AT("expected count = %d, got %zu", N_REPLAYS, count);
    }

    // Sum must be 1 + 2 = 3.0.
    if (std::abs(out_host[0] - 3.0f) > 1e-5f) {
        FAIL_AT("expected out_host[0] = 3.0, got %f", out_host[0]);
    }
    // All replays byte-identical.
    for (int i = 1; i < N_REPLAYS; ++i) {
        if (hashes[i] != hashes[0]) {
            FAIL_AT("hash mismatch at replay %d: %016lx vs %016lx",
                    i, (unsigned long) hashes[i], (unsigned long) hashes[0]);
        }
    }
    std::fprintf(stdout, "  output bit-identical across %d replays: %016lx\n",
                 N_REPLAYS, (unsigned long) hashes[0]);

    ggml_cuda_outer_capture_destroy_exec(exec);
    ggml_backend_free(cuda_a);
    ggml_backend_free(cuda_b);

    std::fprintf(stdout, "test-multi-device-graph-cache: PASS\n");
    return 0;
}
