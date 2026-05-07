// test-mtp-fused-split-argmax.cpp
//
// Granular unit test for Phase 36 Issue F: ggml_argmax on the result
// of ggml_concat across two CUDA devices returns out-of-vocab
// indices.
//
// Strategy:
//   Case A (sanity): single-device concat + argmax — expected PASS.
//     Two halves of the logit vector live on CUDA0, concat'd into
//     one tensor, argmax over the concat. Confirms argmax + concat
//     plumbing works at the op level.
//
//   Case B (bug): split-mode-graph emulation — schedule the two
//     halves on different CUDA backends (CUDA0 + CUDA1) via
//     ggml_backend_sched, concat'd into one logical tensor, then
//     argmax. If the argmax kernel runs on one device but the
//     concat input lives across two, the read-side may walk past
//     its own slice into uninitialised memory.
//
// Comparing Case A vs Case B isolates whether the regression is
// "argmax + concat" generally (Case A would fail) or specifically
// "argmax + cross-device concat" (only Case B fails). The latter
// is Issue F as observed at runtime; the per-device argmax +
// reduction fix targets exactly that.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cstdio>
#include <cstdint>
#include <vector>

#ifdef GGML_USE_CUDA
// Build a graph that fills two halves with known peaks, concat's
// them, and argmax's. Returns the argmax via *out_argmax.
//
// `place_split_devs` controls how the halves are placed:
//   - false: both halves on CUDA0 (Case A, single-device).
//   - true:  half_lo on CUDA0, half_hi on CUDA1 (Case B, multi-device).
static int run_concat_argmax_test(
        bool       place_split_devs,
        int64_t    n_vocab,
        int64_t    peak_lo,
        int64_t    peak_hi_local,   // index within the high half (offset is added)
        float      peak_lo_val,
        float      peak_hi_val,
        int32_t *  out_argmax) {
    const int64_t half = n_vocab / 2;

    ggml_backend_t backend_0 = ggml_backend_cuda_init(0, nullptr);
    ggml_backend_t backend_1 = place_split_devs ? ggml_backend_cuda_init(1, nullptr) : nullptr;
    if (!backend_0 || (place_split_devs && !backend_1)) {
        fprintf(stderr, "FAIL: cuda_init returned null\n");
        return -1;
    }

    struct ggml_init_params iparams = { 16*1024*1024, nullptr, false };
    struct ggml_context * ctx = ggml_init(iparams);

    ggml_tensor * half_lo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, 1);
    ggml_tensor * half_hi = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, 1);
    ggml_set_name(half_lo, "half_lo");
    ggml_set_name(half_hi, "half_hi");
    ggml_set_input(half_lo);
    ggml_set_input(half_hi);

    ggml_tensor * concat = ggml_concat(ctx, half_lo, half_hi, 0);
    ggml_set_name(concat, "concat");

    ggml_tensor * amax = ggml_argmax(ctx, concat);
    ggml_set_name(amax, "amax");
    ggml_set_output(amax);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, amax);

    // ggml_backend_sched requires the LAST backend to be a CPU
    // backend. Add one to satisfy the assertion.
    ggml_backend_t backend_cpu = ggml_backend_cpu_init();

    // Drive allocation through ggml_backend_sched so we can pin
    // half_lo / half_hi to specific backends.
    ggml_backend_t backends[3] = {
        backend_0,
        backend_1 ? backend_1 : backend_cpu,
        backend_cpu,
    };
    int n_backends = backend_1 ? 3 : 2;
    ggml_backend_sched_t sched = ggml_backend_sched_new(
            backends, /*bufts*/ nullptr, n_backends, /*graph_size*/ 16, /*parallel*/ false);

    if (place_split_devs && backend_1) {
        ggml_backend_sched_set_tensor_backend(sched, half_lo, backend_0);
        ggml_backend_sched_set_tensor_backend(sched, half_hi, backend_1);
    }
    ggml_backend_sched_reserve(sched, gf);
    ggml_backend_sched_alloc_graph(sched, gf);

    std::vector<float> data_lo(half, 0.0f);
    std::vector<float> data_hi(half, 0.0f);
    data_lo[peak_lo]       = peak_lo_val;
    data_hi[peak_hi_local] = peak_hi_val;
    ggml_backend_tensor_set(half_lo, data_lo.data(), 0, half * sizeof(float));
    ggml_backend_tensor_set(half_hi, data_hi.data(), 0, half * sizeof(float));

    ggml_backend_sched_graph_compute(sched, gf);

    int32_t result = -1;
    ggml_backend_tensor_get(amax, &result, 0, sizeof(int32_t));

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    ggml_backend_free(backend_0);
    if (backend_1) ggml_backend_free(backend_1);
    ggml_backend_free(backend_cpu);

    if (out_argmax) *out_argmax = result;
    return 0;
}
#endif

int main() {
#ifndef GGML_USE_CUDA
    fprintf(stderr, "skipped: built without CUDA\n");
    return 77;
#else
    if (ggml_backend_cuda_get_device_count() < 2) {
        fprintf(stderr, "skipped: requires 2+ CUDA devices (have %d)\n",
                ggml_backend_cuda_get_device_count());
        return 77;
    }

    const int64_t n_vocab        = 8192;
    const int64_t peak_lo_idx    = 1234;          // in lo half
    const int64_t peak_hi_local  = 5172;          // in hi half (local index)
    const int64_t peak_hi_global = (n_vocab/2) + peak_hi_local;

    // Place a small peak in lo (5.0) and a TALLER peak in hi (100.0).
    // Global argmax = peak_hi_global. Both cases (single + split)
    // must agree on that for argmax to be correct.

    // Case A: single-device baseline.
    int32_t result_a = -1;
    if (run_concat_argmax_test(/*split_devs*/ false, n_vocab,
            peak_lo_idx, peak_hi_local, 5.0f, 100.0f, &result_a) != 0) {
        fprintf(stderr, "FAIL [single]: harness error\n");
        return 1;
    }
    if (result_a != peak_hi_global) {
        fprintf(stderr, "FAIL [single]: argmax=%d, expected %ld\n",
                result_a, (long) peak_hi_global);
        return 1;
    }
    printf("PASS [single-device concat+argmax]: result=%d, expected=%ld\n",
           result_a, (long) peak_hi_global);

    // Case B: cross-device concat + argmax — this is Issue F.
    int32_t result_b = -1;
    if (run_concat_argmax_test(/*split_devs*/ true, n_vocab,
            peak_lo_idx, peak_hi_local, 5.0f, 100.0f, &result_b) != 0) {
        fprintf(stderr, "FAIL [split]: harness error\n");
        return 1;
    }
    const bool issue_f_live = (result_b != peak_hi_global);
    if (issue_f_live) {
        printf("CONFIRMED Issue F: split argmax=%d, expected=%ld (in-range=%s)\n",
               result_b, (long) peak_hi_global,
               (result_b >= 0 && result_b < n_vocab) ? "yes" : "no");
        return 1;
    }
    printf("Issue F NOT reproduced: split argmax=%d (correct)\n", result_b);
    return 0;
#endif
}
