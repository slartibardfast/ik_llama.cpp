// Unit test for the per-op fork/join concurrent event API
// (ggml_backend_concurrent_event_*). Exercises:
//   1. event creation against a CUDA backend
//   2. fork() puts per-slot streams into a wait-state on a fan-out event
//   3. caller dispatches per-slot work on the per-slot streams
//   4. join() makes the main stream wait on all per-slot completions
//   5. main-stream D2H sees all per-slot writes (ordering correctness)
//
// Aborts on any failure. Pass = exit 0.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// Private CUDA backend accessor — declared extern here so the test can drive
// per-slot kernel dispatch directly. Defined in ggml/src/ggml-cuda.cu.
extern "C" cudaStream_t ggml_backend_cuda_concurrent_event_stream(ggml_backend_concurrent_event_t event, int slot_idx);

#define CHECK_CUDA(expr) do {                                    \
    cudaError_t _err = (expr);                                   \
    if (_err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(_err));   \
        std::abort();                                            \
    }                                                            \
} while (0)

#define CHECK(cond) do {                                         \
    if (!(cond)) {                                               \
        fprintf(stderr, "Assertion failed at %s:%d: %s\n",       \
                __FILE__, __LINE__, #cond);                      \
        std::abort();                                            \
    }                                                            \
} while (0)

int main() {
    if (ggml_backend_cuda_get_device_count() < 1) {
        fprintf(stderr, "no CUDA devices available, skipping\n");
        return 0;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    CHECK(backend != nullptr);

    // --- API surface checks -----------------------------------------------

    {
        // n_slots=0 must abort upstream; we test only non-degenerate paths.
        // NULL event behavior on free/fork/join is documented as no-op.
        ggml_backend_concurrent_event_free(nullptr);
        ggml_backend_concurrent_event_fork(nullptr);
        ggml_backend_concurrent_event_join(nullptr);
        CHECK(ggml_backend_concurrent_event_n_slots(nullptr) == 0);
    }

    // --- Ordering correctness test ----------------------------------------
    //
    // Allocate one device buffer of N * sizeof(int32_t). Fork. Each slot s
    // writes the pattern (s + 0xC0DE0000) into buf[s] on its per-slot stream.
    // Join. Main stream copies buf -> host. Synchronize. Verify all slots
    // committed the right value before the host copy ran.

    const int n_slots = 4;
    ggml_backend_concurrent_event_t ev = ggml_backend_concurrent_event_new(backend, n_slots);
    CHECK(ev != nullptr);
    CHECK(ggml_backend_concurrent_event_n_slots(ev) == n_slots);

    int32_t * d_buf = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buf, n_slots * sizeof(int32_t)));

    // Sentinel: zero out the device buffer on the default stream first, so a
    // missed write would show as 0 instead of an uninitialized value.
    CHECK_CUDA(cudaMemset(d_buf, 0, n_slots * sizeof(int32_t)));
    CHECK_CUDA(cudaDeviceSynchronize());

    ggml_backend_concurrent_event_fork(ev);

    // Per-slot host-side staging values + H2D copies on per-slot streams.
    int32_t h_vals[n_slots];
    for (int s = 0; s < n_slots; ++s) {
        h_vals[s] = 0xC0DE0000 | s;
    }
    for (int s = 0; s < n_slots; ++s) {
        cudaStream_t st = ggml_backend_cuda_concurrent_event_stream(ev, s);
        CHECK(st != nullptr);
        // Copy a single int32 from host to its slot in the device buffer on the
        // per-slot stream. Must complete before join() returns to main stream.
        CHECK_CUDA(cudaMemcpyAsync(d_buf + s, &h_vals[s], sizeof(int32_t), cudaMemcpyHostToDevice, st));
    }

    ggml_backend_concurrent_event_join(ev);

    // After join, the main stream must observe all per-slot writes. Do the D2H
    // on the default stream (as a stand-in for "main"); a missing barrier would
    // race here.
    int32_t h_out[n_slots];
    std::memset(h_out, 0xFF, sizeof(h_out));
    CHECK_CUDA(cudaMemcpy(h_out, d_buf, sizeof(h_out), cudaMemcpyDeviceToHost));

    for (int s = 0; s < n_slots; ++s) {
        if (h_out[s] != h_vals[s]) {
            fprintf(stderr, "slot %d: expected 0x%08x, got 0x%08x\n", s, (unsigned)h_vals[s], (unsigned)h_out[s]);
            std::abort();
        }
    }

    CHECK_CUDA(cudaFree(d_buf));
    ggml_backend_concurrent_event_free(ev);

    // --- Reuse / lifetime check -------------------------------------------
    //
    // Create + destroy multiple events to exercise stream/event allocator paths.

    for (int i = 0; i < 8; ++i) {
        ggml_backend_concurrent_event_t e2 = ggml_backend_concurrent_event_new(backend, 1 + (i % 4));
        CHECK(e2 != nullptr);
        ggml_backend_concurrent_event_fork(e2);
        ggml_backend_concurrent_event_join(e2);
        ggml_backend_concurrent_event_free(e2);
    }

    ggml_backend_free(backend);

    printf("test-cuda-concurrent-event: PASS\n");
    return 0;
}
