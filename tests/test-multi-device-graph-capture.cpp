// test-multi-device-graph-capture.cpp
//
// PHASE_CUDA_NATIVE_DISPATCH C4 — bind outer-capture mechanism.
//
// Drives ggml_cuda_outer_capture_begin / _end_and_launch on a 2-CUDA-
// backend setup. Each iteration:
//   1. Begin outer capture on cuda[0]'s stream (Relaxed mode + cross-
//      stream event-chain pulls cuda[1]'s stream into the capture).
//   2. Run a trivial ggml graph_compute on each backend. Under the
//      C3 in-capture gate these eager-launch kernels onto each
//      backend's stream, where the CUDA driver folds them into the
//      outer captured cudaGraph_t.
//   3. End capture: fan-in events + cudaStreamEndCapture +
//      cudaGraphInstantiate + cudaGraphLaunch + destroy.
//
// Binding:
//   - ggml_cuda_outer_capture_count() advances by exactly N after N
//     successful begin→end_and_launch round-trips.
//   - The output tensor hash is byte-identical across all N runs.
//
// Specs bound:
//   /home/dconnolly/yarn-agentic/specs/cuda-native-dispatch/cross_device_event_chain.allium
//   /home/dconnolly/yarn-agentic/specs/cuda-native-dispatch/CUDANativeDispatch.tla
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

static constexpr int N_ITERS = 20;
static constexpr int64_t N   = 64;

// Build a minimal compute context with one add graph (a + b) routed
// to the supplied backend. Allocates the tensors via the backend's
// default buffer type; returns the context, graph, output tensor,
// and buffer (all of which must outlive the graph_compute call).
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

// Simple 64-bit FNV-1a hash over the output buffer (for byte-identity
// across iters).
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
        std::fprintf(stderr, "test-multi-device-graph-capture: SKIP (need 2 CUDA devices, have %d)\n", n_cuda);
        return 77;
    }

    std::fprintf(stdout, "test-multi-device-graph-capture: running with %d CUDA device(s)\n", n_cuda);

    ggml_backend_t cuda_a = ggml_backend_cuda_init(0, nullptr);
    ggml_backend_t cuda_b = ggml_backend_cuda_init(1, nullptr);
    if (!cuda_a || !cuda_b) FAIL_AT("ggml_backend_cuda_init failed");

    one_backend_graph ga, gb;
    build_one_backend_graph(cuda_a, ga);
    build_one_backend_graph(cuda_b, gb);

    std::vector<float> data_a((size_t) N * N, 1.0f);
    std::vector<float> data_b((size_t) N * N, 2.0f);

    ggml_cuda_outer_capture_count_reset();

    std::vector<float>   out_host(N * N);
    std::vector<uint64_t> hashes(N_ITERS);

    for (int iter = 0; iter < N_ITERS; ++iter) {
        ggml_backend_t secondaries[] = { cuda_b };
        bool ok_begin = ggml_cuda_outer_capture_begin(cuda_a, secondaries, 1);
        if (!ok_begin) FAIL_AT("ggml_cuda_outer_capture_begin returned false at iter %d", iter);
        if (!ggml_cuda_outer_capture_active()) {
            FAIL_AT("outer_capture_active should be true after _begin");
        }

        // Set inputs on each backend (cudaMemcpyAsync — capturable in
        // Relaxed mode).
        ggml_backend_tensor_set(ga.a, data_a.data(), 0, data_a.size() * sizeof(float));
        ggml_backend_tensor_set(ga.b, data_b.data(), 0, data_b.size() * sizeof(float));
        ggml_backend_tensor_set(gb.a, data_a.data(), 0, data_a.size() * sizeof(float));
        ggml_backend_tensor_set(gb.b, data_b.data(), 0, data_b.size() * sizeof(float));

        // C3 gate: under outer capture, graph_compute_async does eager
        // kernel launches → captured into the outer graph. Use _async
        // (NOT the synchronous _compute) because cudaDeviceSynchronize
        // / cudaStreamSynchronize are forbidden during capture; the
        // captured graph carries its own ordering via the cross-stream
        // join + fan-in event-chain.
        if (ggml_backend_graph_compute_async(cuda_a, ga.gf) != GGML_STATUS_SUCCESS) {
            FAIL_AT("graph_compute_async on cuda_a failed at iter %d", iter);
        }
        if (ggml_backend_graph_compute_async(cuda_b, gb.gf) != GGML_STATUS_SUCCESS) {
            FAIL_AT("graph_compute_async on cuda_b failed at iter %d", iter);
        }

        bool ok_end = ggml_cuda_outer_capture_end_and_launch(cuda_a, secondaries, 1);
        if (!ok_end) FAIL_AT("ggml_cuda_outer_capture_end_and_launch returned false at iter %d", iter);
        if (ggml_cuda_outer_capture_active()) {
            FAIL_AT("outer_capture_active should be false after _end_and_launch");
        }

        ggml_backend_synchronize(cuda_a);
        ggml_backend_synchronize(cuda_b);

        // Read cuda_a output and hash it.
        ggml_backend_tensor_get(ga.out, out_host.data(), 0, out_host.size() * sizeof(float));
        hashes[iter] = fnv1a64(out_host.data(), out_host.size() * sizeof(float));
    }

    const size_t count = ggml_cuda_outer_capture_count();
    std::fprintf(stdout, "  outer_capture_count after %d iters = %zu\n", N_ITERS, count);
    if (count != (size_t) N_ITERS) {
        FAIL_AT("expected outer_capture_count == %d, got %zu", N_ITERS, count);
    }

    // Output must be 1.0 + 2.0 = 3.0 elementwise.
    if (std::abs(out_host[0] - 3.0f) > 1e-5f) {
        FAIL_AT("expected out_host[0] = 3.0, got %f", out_host[0]);
    }

    // All hashes byte-identical (deterministic captured replay).
    for (int i = 1; i < N_ITERS; ++i) {
        if (hashes[i] != hashes[0]) {
            FAIL_AT("hash mismatch at iter %d: %016lx vs %016lx",
                    i, (unsigned long) hashes[i], (unsigned long) hashes[0]);
        }
    }
    std::fprintf(stdout, "  output hash byte-identical across %d iters: %016lx\n",
                 N_ITERS, (unsigned long) hashes[0]);

    ggml_backend_free(cuda_a);
    ggml_backend_free(cuda_b);

    std::fprintf(stdout, "test-multi-device-graph-capture: PASS\n");
    return 0;
}
