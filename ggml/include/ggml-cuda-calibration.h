// ggml-cuda/calibration.h
//
// PHASE_CUDA_NATIVE_DISPATCH commit C0 — Calibrated dispatch framework.
//
// At context init, ggml_cuda_calibrate() probes each registered op at a
// fixed set of payload sizes and records the smallest size where the
// alt-strategy's p95 latency is less than the default-strategy's p50
// latency (the "conservative crossover"). The threshold is quantized to
// one of {0, 1MB, 10MB, 100MB, 1GB, SIZE_MAX} and stored per-context.
//
// Per-dispatch lookup via ggml_cuda_threshold_for() routes between the
// default and alt strategies based on payload size. Production behavior
// on xeon: REDUCE_CROSS_DEVICE quantizes to the 1 GB bucket (libmgpu's
// per-CLIP-layer reduce ~2-3 MB → memcpy path always); MATMUL_STREAM_
// SPLIT / PEER_COPY → SIZE_MAX; GRAPH_CAPTURE → 0.
//
// The threshold table is cached at
//   $XDG_CACHE_HOME/ggml/cuda-calibration-{gpu_uuid_hash}.json
// (fallback $HOME/.cache, then /tmp/ggml-cache, then in-memory only).
// Cache key = sha256(gpu_uuids ++ cuda_version ++ ggml_commit). Mismatch
// triggers re-calibration.
//
// Env knobs:
//   GGML_CALIBRATION_DISABLE=1          — skip cache I/O, run in-memory
//   GGML_CALIBRATION_FORCE_RECALIBRATE=1 — ignore cache, re-probe
//   GGML_CAL_<OP_NAME>_THRESHOLD_BYTES=N — per-op manual override (e.g.,
//     GGML_CAL_REDUCE_CROSS_DEVICE_THRESHOLD_BYTES=100000000)
//
// Binding spec:
//   /home/dconnolly/yarn-agentic/specs/cuda-native-dispatch/
//     calibrated_dispatch_framework.allium
//     CalibrationFramework.tla
//
// C0 ships the framework + the test that binds it. C8/C9/C10/C11 register
// their respective ops via ggml_cuda_calibration_register_op().

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

// Forward declaration: avoid pulling in CUDA headers here so this header
// is includeable from plain C++ (e.g., the test in tests/spec/).
struct ggml_backend_cuda_context;

#ifdef __cplusplus
extern "C" {
#endif

// Registered calibrated ops. To add a new op:
//   1) Add an entry here (and bump _COUNT_).
//   2) Add its display name to ggml_cal_op_name() in calibration.cu.
//   3) Call ggml_cuda_calibration_register_op(...) from the op's
//      registration site (typically a static initializer in the op's .cu).
//   4) Add an entry to the persisted cache JSON schema (compat).
enum ggml_cuda_calibrated_op {
    GGML_CAL_OP_REDUCE_CROSS_DEVICE  = 0,
    GGML_CAL_OP_MATMUL_STREAM_SPLIT  = 1,
    GGML_CAL_OP_PEER_COPY            = 2,
    GGML_CAL_OP_GRAPH_CAPTURE        = 3,
    GGML_CAL_OP_COUNT_               = 4,
};

// Probe size buckets. LOCKED per PHASE_CUDA_NATIVE_DISPATCH.md §3.0
// rule 4: {0, 1MB, 10MB, 100MB, 1GB, SIZE_MAX}. Index 0 and N-1 are
// sentinels; probes run at indices 1..N-2.
//
// Defined in calibration.cu; declared here for use by op probe
// implementations and tests.
extern const size_t ggml_cal_buckets[6];
extern const int    ggml_cal_n_buckets;  // = 6

#ifdef __cplusplus
}  // extern "C"
#endif

// Probe result for one (op, strategy, payload_size, n_iters) sample.
struct ggml_cuda_probe_result {
    double p50_ms;
    double p95_ms;
};

// Per-op probe function. Implementation runs n_iters iterations of the
// requested strategy on the given payload_bytes and returns (p50, p95).
// The first ~half of iterations are treated as warm-up internally by
// the framework; the impl should run n_iters total and the framework
// extracts percentiles from the last half.
typedef ggml_cuda_probe_result (*ggml_cuda_probe_fn)(
    ggml_backend_cuda_context * ctx,
    bool                        use_alt,         // false = default, true = alt strategy
    size_t                      payload_bytes,
    int                         n_iters);

// Per-context threshold table. Lives as a field on
// ggml_backend_cuda_context (see common.cuh).
struct ggml_cuda_calibration_table {
    size_t      thresholds[GGML_CAL_OP_COUNT_];  // SIZE_MAX = never use alt
    bool        loaded_from_cache = false;
    bool        calibrated        = false;       // true once ggml_cuda_calibrate has run
    std::string cache_key;                        // for diagnostics
};

#ifdef __cplusplus
extern "C" {
#endif

// Register a probe function for an op. Called by C8-C11 from their op's
// static-initializer or backend-init site. Safe to call multiple times
// (re-registration overwrites). Calling with probe=nullptr unregisters.
//
// The registration is GLOBAL (not per-context), since the same op uses
// the same probe regardless of which CUDA context is being calibrated.
void ggml_cuda_calibration_register_op(
    enum ggml_cuda_calibrated_op op,
    const char *                 name,    // e.g., "REDUCE_CROSS_DEVICE"
    ggml_cuda_probe_fn           probe);

// Run calibration on the given context. Called once during context
// initialization (in ggml_backend_cuda_init after p2p access setup).
// Loads cache if hit, otherwise probes every registered op. Always
// initializes ctx->calibration_table to a valid state (even for ops
// without a registered probe — those get SIZE_MAX).
//
// Idempotent: if ctx->calibration_table.calibrated == true, returns
// immediately without re-running.
void ggml_cuda_calibrate(ggml_backend_cuda_context * ctx);

// Per-dispatch lookup. Returns the byte threshold for op_id on this
// context, or SIZE_MAX if the op is not registered or no crossover
// was found during calibration.
//
// Performance: this is hot-path code at each calibrated op site. The
// impl reads from the per-context table; no locking, no atomics.
size_t ggml_cuda_threshold_for(
    ggml_backend_cuda_context * ctx,
    enum ggml_cuda_calibrated_op op);

// Convenience: at op dispatch site, returns whether to use alt strategy
// for a given payload. Equivalent to
//   payload_bytes >= ggml_cuda_threshold_for(ctx, op)
// but inline-friendly.
//
// Caller pattern at a calibrated op site:
//   if (ggml_cuda_use_alt_strategy(ctx, GGML_CAL_OP_REDUCE_CROSS_DEVICE, bytes)) {
//       do_ncclAllReduce(...);   // alt path
//   } else {
//       do_memcpy_peer_and_add(...);  // default path
//   }
static inline bool ggml_cuda_use_alt_strategy(
    struct ggml_backend_cuda_context * ctx,
    enum ggml_cuda_calibrated_op       op,
    size_t                             payload_bytes) {
    return payload_bytes >= ggml_cuda_threshold_for(ctx, op);
}

// For tests and diagnostics: get the display name of an op.
const char * ggml_cal_op_name(enum ggml_cuda_calibrated_op op);

// For tests: clear the global probe registry. After this, no ops are
// registered until ggml_cuda_calibration_register_op() is called again.
// Use only in tests; production code never unregisters.
void ggml_cuda_calibration_reset_registry_for_tests(void);

// For tests: take a ggml_backend_t (CUDA backend) and return whether
// its calibration table was loaded from cache (vs probed fresh). The
// backend's context struct ggml_backend_cuda_context is internal; this
// helper exists so tests don't need to peek into the internal struct.
// Returns false if backend is null or not a CUDA backend.
struct ggml_backend;
bool ggml_cuda_calibration_was_loaded_from_cache(struct ggml_backend * backend);

// For tests: get the threshold for an op on a given backend.
// Equivalent to ggml_cuda_threshold_for() but accepts the public
// ggml_backend_t type.
size_t ggml_cuda_calibration_threshold_for_backend(
    struct ggml_backend *        backend,
    enum ggml_cuda_calibrated_op op);

#ifdef __cplusplus
}  // extern "C"
#endif
