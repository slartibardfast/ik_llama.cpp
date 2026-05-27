// test-single-threaded-dispatch.cpp
//
// PHASE_CUDA_NATIVE_DISPATCH C1 — bind HostThreadIsExactlyOne.
//
// The new ggml_backend_sched_compute_splits records its caller thread
// id via ggml_backend_sched_dispatch_thread_observe() (internal) and
// exposes the distinct-thread-id count via the public API
// ggml_backend_sched_dispatch_thread_count(). This test asserts that
// across N sched graph computes, exactly one distinct host thread has
// entered compute_splits — i.e. the openmp parallel block + std::barrier
// worker-thread variants are gone.
//
// Strategy: build a 3-backend sched (2 CUDA devices + CPU when 2 GPUs
// are present, else 1 CUDA + CPU). Configure split_mode_graph=true and
// async=true so the multi-backend dispatch path matches the production
// CLIP encoder shape. Run a small graph through sched 100 times.
// Reset the dispatch-thread counter at the start; assert it reads 1
// at the end.
//
// Returns: 0 = PASS, 1 = FAIL, 77 = SKIP (no CUDA devices).
//
// Specs bound:
//   /home/dconnolly/yarn-agentic/specs/cuda-native-dispatch/single_threaded_dispatch.allium
//   /home/dconnolly/yarn-agentic/specs/cuda-native-dispatch/CUDANativeDispatch.tla

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define FAIL_AT(msg, ...) do { \
    std::fprintf(stderr, "FAIL %s:%d: " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    std::exit(1); \
} while (0)

static constexpr int N_ITERS = 100;

int main() {
    const int n_cuda = ggml_backend_cuda_get_device_count();
    if (n_cuda < 1) {
        std::fprintf(stderr, "test-single-threaded-dispatch: SKIP (no CUDA devices)\n");
        return 77;
    }

    std::fprintf(stdout, "test-single-threaded-dispatch: running with %d CUDA device(s)\n", n_cuda);

    // Build backend list: every available CUDA device + CPU.
    std::vector<ggml_backend_t>             backends;
    std::vector<ggml_backend_buffer_type_t> bufts;
    for (int d = 0; d < n_cuda && d < 2; ++d) {
        ggml_backend_t b = ggml_backend_cuda_init(d, nullptr);
        if (!b) FAIL_AT("ggml_backend_cuda_init(%d) returned null", d);
        backends.push_back(b);
        bufts.push_back(ggml_backend_get_default_buffer_type(b));
    }
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) FAIL_AT("ggml_backend_cpu_init returned null");
    backends.push_back(cpu);
    bufts.push_back(ggml_backend_cpu_buffer_type());

    // PHASE_CUDA_NATIVE_DISPATCH C2: bind eager init of the four PD1
    // lazy-create surfaces on every CUDA backend. C1 made these races
    // race-free (single dispatcher thread); C2 makes them unreachable.
    for (auto * be : backends) {
        if (ggml_backend_is_cuda(be) && !ggml_backend_cuda_eager_init_complete(be)) {
            FAIL_AT("CUDA backend has non-pre-allocated lazy fields after init "
                    "(C2 NoLazyInitInDispatchPath violated)");
        }
    }

    // Match the production CLIP encoder sched shape: parallel=true,
    // split_mode_graph=true, async=true. This forces the multi-backend
    // dispatch branch when n_backends > 2.
    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends.data(), bufts.data(), (int) backends.size(),
        /*graph_size=*/ 4096, /*parallel=*/ true);
    if (!sched) FAIL_AT("ggml_backend_sched_new returned null");
    ggml_backend_sched_set_split_mode_graph(sched, true, /*async=*/ true);

    // Reset the dispatch-thread counter so a count of 1 binds exactly
    // to the work this test drives (any earlier dispatches on this
    // process — none expected, but defensive — are forgotten).
    ggml_backend_sched_dispatch_thread_reset();
    if (ggml_backend_sched_dispatch_thread_count() != 0) {
        FAIL_AT("dispatch_thread_count after reset should be 0, got %zu",
                ggml_backend_sched_dispatch_thread_count());
    }

    // Tiny graph: add two tensors. Sched routes it to whichever backend
    // it picks; the dispatch path through compute_splits is what we are
    // binding here, not the kernel choice.
    static const size_t ctx_size = 4 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) FAIL_AT("ggml_init returned null");

    constexpr int64_t N = 64;
    ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N);
    ggml_tensor * b   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N);
    ggml_tensor * sum = ggml_add(ctx, a, b);
    ggml_set_name(sum, "sum");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sum);

    if (!ggml_backend_sched_reserve(sched, gf)) {
        FAIL_AT("ggml_backend_sched_reserve failed");
    }

    // Pre-fill inputs on host. The sched will route the actual storage
    // wherever it chose; tensor_set handles the upload.
    std::vector<float> data((size_t) N * N, 1.0f);

    for (int iter = 0; iter < N_ITERS; ++iter) {
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) {
            FAIL_AT("ggml_backend_sched_alloc_graph failed at iter %d", iter);
        }
        ggml_backend_tensor_set(a, data.data(), 0, data.size() * sizeof(float));
        ggml_backend_tensor_set(b, data.data(), 0, data.size() * sizeof(float));

        const enum ggml_status st = ggml_backend_sched_graph_compute(sched, gf);
        if (st != GGML_STATUS_SUCCESS) {
            FAIL_AT("graph_compute returned status %d at iter %d", (int) st, iter);
        }
    }

    const size_t tid_count = ggml_backend_sched_dispatch_thread_count();
    std::fprintf(stdout, "  dispatch_thread_count after %d iters = %zu\n",
                 N_ITERS, tid_count);
    if (tid_count != 1) {
        FAIL_AT("HostThreadIsExactlyOne violated: expected 1 dispatcher thread, got %zu",
                tid_count);
    }

    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    for (auto * be : backends) ggml_backend_free(be);

    std::fprintf(stdout, "test-single-threaded-dispatch: PASS\n");
    return 0;
}
