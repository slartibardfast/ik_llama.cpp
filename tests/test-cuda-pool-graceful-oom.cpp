// CUDA pool: graceful OOM refusal — direct pool exercise.
//
// Drives a synthetic OOM through the backend pool via the
// GGML_CUDA_POOL_FORCE_FAIL_NEXT debug hook and asserts the pool
// returns nullptr instead of aborting. Uses the test-only
// ggml_backend_cuda_pool_alloc_test entry point so the binding doesn't
// depend on which op happens to call into the pool from a graph
// compute.
//
//   T1: pool->alloc(N) under the hook returns nullptr; no abort.
//   T2: a normal pool->alloc(N) returns a valid pointer that
//       round-trips through pool->free; regression canary that the
//       hook is opt-in only and doesn't perturb default behaviour.

#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstdlib>

// Direct access to the pool requires a backend context — the public
// API exposes pools indirectly. T1 + T2 exercise the chain via
// ggml_backend_graph_compute on a tiny graph; the synthetic hook
// fires on the first compute_forward node.

// T1: pool->alloc under the synthetic-fail hook returns nullptr without
// aborting. Returns 0 on PASS.
static int run_t1_forced_oom(ggml_backend_t backend) {
    setenv("GGML_CUDA_POOL_FORCE_FAIL_NEXT", "1", 1);
    size_t actual = 0;
    void * p = ggml_backend_cuda_pool_alloc_test(backend, 4096, &actual);
    unsetenv("GGML_CUDA_POOL_FORCE_FAIL_NEXT");

    if (p != nullptr) {
        // Hook didn't fire OR pool ignored it. Free what we got and
        // report failure.
        ggml_backend_cuda_pool_free_test(backend, p, actual);
        return 1;
    }
    return 0;
}

// T2: pool->alloc without the hook returns a valid pointer. Round-trips
// through free without crashing. Returns 0 on PASS.
static int run_t2_normal(ggml_backend_t backend) {
    size_t actual = 0;
    void * p = ggml_backend_cuda_pool_alloc_test(backend, 4096, &actual);
    if (p == nullptr) return 1;
    ggml_backend_cuda_pool_free_test(backend, p, actual);
    return 0;
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed; skipping\n");
        return 0;
    }

    printf("=== test-cuda-pool-graceful-oom ===\n");

    // T2 first — ensure the default path works (otherwise a real bug
    // elsewhere would mask T1 by failing both the same way).
    const int t2 = run_t2_normal(backend);
    printf("  T2 (no hook): %s\n", t2 == 0 ? "PASS — pool alloc returned valid pointer" : "FAIL");
    if (t2 != 0) {
        printf("RESULT: FAIL — default-path pool alloc did not succeed\n");
        ggml_backend_free(backend);
        return 1;
    }

    // T1 — forced OOM via the env hook. Expect nullptr return, not
    // an abort (if it aborted we'd never reach this print).
    const int t1 = run_t1_forced_oom(backend);
    printf("  T1 (hook=1): %s\n", t1 == 0 ? "PASS — pool returned nullptr without aborting"
                                           : "FAIL — hook ignored or pool did not soft-fail");

    ggml_backend_free(backend);

    if (t1 != 0) return 1;
    printf("RESULT: PASS — graceful OOM refusal at the pool layer\n");
    return 0;
}
