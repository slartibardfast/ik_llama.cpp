// test-cuda-calibration-framework.cpp
//
// PHASE_CUDA_NATIVE_DISPATCH C0 — bind the calibration framework's
// invariants (calibrated_dispatch_framework.allium + CalibrationFramework.tla).
//
// Scope: framework plumbing only. Per-op equivalence tests come with
// C8/C9/C10/C11. C0 covers:
//   T5: no ops registered → all thresholds = SIZE_MAX
//   T4: per-op env override wins over probing
//   T6: synthetic op registration → calibration finds expected bucket
//   T7: deterministic probe → two fresh contexts produce same threshold
//   T8: cache write/read round-trip preserves thresholds
//   T9: GGML_CALIBRATION_FORCE_RECALIBRATE bypasses cache hit
//
// Returns: 0 = PASS, 1 = FAIL, 77 = SKIP (no CUDA devices).

#include "ggml-cuda.h"
#include "ggml-cuda-calibration.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <atomic>

namespace {

#define FAIL_AT(msg, ...) do { \
    std::fprintf(stderr, "FAIL %s:%d: " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    std::exit(1); \
} while (0)

#define EXPECT_EQ(a, b, msg) do { \
    const size_t _a = (size_t)(a); \
    const size_t _b = (size_t)(b); \
    if (_a != _b) { \
        FAIL_AT(msg " — expected %zu, got %zu", _b, _a); \
    } \
} while (0)

// =============================================================================
// Synthetic probe. Deterministic: alt-strategy is faster than default
// for payloads >= g_synthetic_crossover_bytes. Calibration framework
// should pick the smallest bucket >= g_synthetic_crossover_bytes.
// =============================================================================

static std::atomic<int>  g_probe_call_count{0};
static size_t            g_synthetic_crossover_bytes = 100ULL * 1024 * 1024;

ggml_cuda_probe_result synthetic_probe(
        ggml_backend_cuda_context * /*ctx*/,
        bool                        use_alt,
        size_t                      payload_bytes,
        int                         /*n_iters*/) {
    g_probe_call_count.fetch_add(1, std::memory_order_relaxed);
    ggml_cuda_probe_result r{};
    // Step-function crossover at g_synthetic_crossover_bytes.
    //   p <  crossover : alt is 2.0× slower than default
    //   p >= crossover : alt is 0.5× as slow (i.e., 2× faster)
    // Default latency: 1.0 ms baseline + 0.1 ms/MB bandwidth term.
    const double def_ms = 1.0 + 0.1 * (double) payload_bytes / (1024.0 * 1024.0);
    const double scale  = (payload_bytes >= g_synthetic_crossover_bytes) ? 0.5 : 2.0;
    if (use_alt) {
        r.p50_ms = def_ms * scale;
        r.p95_ms = def_ms * scale * 1.05;
    } else {
        r.p50_ms = def_ms;
        r.p95_ms = def_ms * 1.02;
    }
    return r;
}

// Spin up a CUDA backend. Calibration runs as part of init.
struct backend_handle {
    ggml_backend_t backend = nullptr;
    backend_handle() {
        backend = ggml_backend_cuda_init(0, nullptr);
        if (!backend) FAIL_AT("ggml_backend_cuda_init(0) returned null");
    }
    ~backend_handle() { if (backend) ggml_backend_free(backend); }
};

void clear_env() {
    unsetenv("GGML_CALIBRATION_DISABLE");
    unsetenv("GGML_CALIBRATION_FORCE_RECALIBRATE");
    unsetenv("GGML_CAL_REDUCE_CROSS_DEVICE_THRESHOLD_BYTES");
    unsetenv("GGML_CAL_MATMUL_STREAM_SPLIT_THRESHOLD_BYTES");
    unsetenv("GGML_CAL_PEER_COPY_THRESHOLD_BYTES");
    unsetenv("GGML_CAL_GRAPH_CAPTURE_THRESHOLD_BYTES");
    unsetenv("XDG_CACHE_HOME");
}

// =============================================================================
// T5: no ops registered → every threshold is SIZE_MAX
// =============================================================================
void test_no_ops_registered() {
    clear_env();
    setenv("GGML_CALIBRATION_DISABLE", "1", 1);
    ggml_cuda_calibration_reset_registry_for_tests();

    backend_handle h;
    for (int op = 0; op < GGML_CAL_OP_COUNT_; ++op) {
        const size_t thr = ggml_cuda_calibration_threshold_for_backend(
            h.backend, (ggml_cuda_calibrated_op) op);
        EXPECT_EQ(thr, SIZE_MAX,
                  "no ops registered: every threshold must be SIZE_MAX");
    }
    std::fprintf(stdout, "  T5 PASS: no ops registered → all SIZE_MAX\n");
}

// =============================================================================
// T4: per-op env override wins over probing
// =============================================================================
void test_env_override_per_op() {
    clear_env();
    setenv("GGML_CALIBRATION_DISABLE", "1", 1);
    setenv("GGML_CAL_REDUCE_CROSS_DEVICE_THRESHOLD_BYTES", "12345", 1);

    ggml_cuda_calibration_reset_registry_for_tests();
    ggml_cuda_calibration_register_op(GGML_CAL_OP_REDUCE_CROSS_DEVICE,
                                      "REDUCE_CROSS_DEVICE",
                                      synthetic_probe);

    backend_handle h;
    const size_t thr = ggml_cuda_calibration_threshold_for_backend(
        h.backend, GGML_CAL_OP_REDUCE_CROSS_DEVICE);
    EXPECT_EQ(thr, (size_t) 12345,
              "env override must produce the literal byte threshold");
    std::fprintf(stdout, "  T4 PASS: env override threshold = 12345\n");
}

// =============================================================================
// T6: synthetic op → calibration finds the expected bucket
// =============================================================================
void test_synthetic_probe_finds_bucket() {
    clear_env();
    setenv("GGML_CALIBRATION_DISABLE", "1", 1);

    g_synthetic_crossover_bytes = 1ULL * 1024 * 1024;  // 1 MB
    ggml_cuda_calibration_reset_registry_for_tests();
    ggml_cuda_calibration_register_op(GGML_CAL_OP_REDUCE_CROSS_DEVICE,
                                      "REDUCE_CROSS_DEVICE",
                                      synthetic_probe);

    backend_handle h;
    const size_t thr = ggml_cuda_calibration_threshold_for_backend(
        h.backend, GGML_CAL_OP_REDUCE_CROSS_DEVICE);
    // Synthetic crossover at 1 MB: smallest bucket where alt.p95 <
    // def.p50 is the 1 MB bucket.
    EXPECT_EQ(thr, ggml_cal_buckets[1],
              "synthetic-probe crossover should quantize to 1 MB bucket");
    std::fprintf(stdout, "  T6 PASS: synthetic probe found 1 MB bucket\n");
}

// =============================================================================
// T7: deterministic probe → two fresh contexts produce same threshold
// =============================================================================
void test_deterministic_probe() {
    clear_env();
    setenv("GGML_CALIBRATION_DISABLE", "1", 1);
    g_synthetic_crossover_bytes = 100ULL * 1024 * 1024;
    ggml_cuda_calibration_reset_registry_for_tests();
    ggml_cuda_calibration_register_op(GGML_CAL_OP_REDUCE_CROSS_DEVICE,
                                      "REDUCE_CROSS_DEVICE",
                                      synthetic_probe);

    backend_handle h1;
    const size_t a = ggml_cuda_calibration_threshold_for_backend(
        h1.backend, GGML_CAL_OP_REDUCE_CROSS_DEVICE);

    backend_handle h2;
    const size_t b = ggml_cuda_calibration_threshold_for_backend(
        h2.backend, GGML_CAL_OP_REDUCE_CROSS_DEVICE);

    EXPECT_EQ(a, b, "deterministic probe must yield same threshold");
    std::fprintf(stdout, "  T7 PASS: deterministic probe (threshold=%zu)\n", a);
}

// =============================================================================
// T8 + T9: cache write/read round-trip; FORCE_RECALIBRATE bypasses
// =============================================================================
void test_cache_roundtrip() {
    clear_env();
    g_synthetic_crossover_bytes = 100ULL * 1024 * 1024;
    ggml_cuda_calibration_reset_registry_for_tests();
    ggml_cuda_calibration_register_op(GGML_CAL_OP_REDUCE_CROSS_DEVICE,
                                      "REDUCE_CROSS_DEVICE",
                                      synthetic_probe);

    // Temp cache dir so we don't pollute the user cache.
    const char * tmp = "/tmp/ggml-cal-test";
    // Clean any prior cache files.
    std::string clean = std::string("rm -rf ") + tmp;
    (void) std::system(clean.c_str());
    setenv("XDG_CACHE_HOME", tmp, 1);

    size_t first_threshold;
    {
        backend_handle h;
        first_threshold = ggml_cuda_calibration_threshold_for_backend(
            h.backend, GGML_CAL_OP_REDUCE_CROSS_DEVICE);
        EXPECT_EQ(first_threshold, ggml_cal_buckets[3],
                  "first calibrate should pick 100 MB bucket");
        if (ggml_cuda_calibration_was_loaded_from_cache(h.backend)) {
            FAIL_AT("first calibrate must be cache miss (probed), got cache hit");
        }
    }

    {
        backend_handle h;
        const size_t t = ggml_cuda_calibration_threshold_for_backend(
            h.backend, GGML_CAL_OP_REDUCE_CROSS_DEVICE);
        EXPECT_EQ(t, first_threshold,
                  "cache load should reproduce probed threshold");
        if (!ggml_cuda_calibration_was_loaded_from_cache(h.backend)) {
            FAIL_AT("second calibrate must be cache hit, got cache miss");
        }
    }
    std::fprintf(stdout, "  T8 PASS: cache round-trip preserves threshold\n");

    setenv("GGML_CALIBRATION_FORCE_RECALIBRATE", "1", 1);
    {
        backend_handle h;
        if (ggml_cuda_calibration_was_loaded_from_cache(h.backend)) {
            FAIL_AT("FORCE_RECALIBRATE=1 should not use cache");
        }
    }
    std::fprintf(stdout, "  T9 PASS: FORCE_RECALIBRATE bypasses cache hit\n");
    unsetenv("GGML_CALIBRATION_FORCE_RECALIBRATE");
}

}  // namespace

int main() {
    if (ggml_backend_cuda_get_device_count() < 1) {
        std::fprintf(stderr, "test-cuda-calibration-framework: SKIP (no CUDA devices)\n");
        return 77;
    }

    std::fprintf(stdout, "test-cuda-calibration-framework: running\n");

    test_no_ops_registered();
    test_env_override_per_op();
    test_synthetic_probe_finds_bucket();
    test_deterministic_probe();
    test_cache_roundtrip();

    std::fprintf(stdout, "test-cuda-calibration-framework: PASS\n");
    return 0;
}
