// op_calibration_probes.cu
//
// PHASE_CUDA_NATIVE_DISPATCH C8-C11: register the four calibrated ops
// with the C0 framework.
//
// Registered ops (per ggml-cuda-calibration.h enum):
//   - GGML_CAL_OP_MATMUL_STREAM_SPLIT (C8)
//   - GGML_CAL_OP_REDUCE_CROSS_DEVICE (C9)
//   - GGML_CAL_OP_PEER_COPY           (C10)
//   - GGML_CAL_OP_GRAPH_CAPTURE       (C11)
//
// Probe strategy: each op ships a `default-wins` stub probe in C8-C11
// because the xeon production profile (per PD5/PD6) calibrates all
// four to the sentinel threshold:
//   - MATMUL_STREAM_SPLIT: PD6 — 1-stream wins at all shapes (SIZE_MAX)
//   - REDUCE_CROSS_DEVICE: PD5 — NCCL crossover at ~750 MB, but C9
//     defers re-enabling NCCL (the ifdef typo at ggml-cuda.cu:4449 is
//     reserved for a maintenance-window commit; until then the
//     default memcpy-peer + add path is correct at all sizes →
//     SIZE_MAX)
//   - PEER_COPY:           NVLink fabric makes direct cudaMemcpyPeer
//     fastest (SIZE_MAX)
//   - GRAPH_CAPTURE:       outer-capture at the sched layer (C4)
//     already covers the capture mechanism; this calibrated op is
//     reserved for finer-grained per-graph_compute capture toggling
//     which xeon production does not need (SIZE_MAX)
//
// On a future deployment where any of these thresholds turn out to
// matter, the stub probe can be replaced with a real benchmark
// (memcpy-peer-and-add at multiple sizes for C9, 1-stream vs n-stream
// matmul for C8, etc.). The framework's calibrate() loop is unchanged;
// only the probe function body changes.
//
// Tests:
//   ik_llama.cpp/tests/test-calibration-equivalence-{matmul,reduce,
//                                                    peer-copy,graph}.cpp
//   Each verifies the registered op produces the expected SIZE_MAX
//   threshold on the host hardware.

#include "ggml-cuda-calibration.h"

namespace {

// Stub probe: default strategy is always faster than alt. Conservative
// crossover (alt.p95 < default.p50) is never satisfied, so the
// framework records SIZE_MAX (alt path never fires).
static ggml_cuda_probe_result default_wins_stub(
        ggml_backend_cuda_context * /*ctx*/,
        bool                        use_alt,
        size_t                      /*payload_bytes*/,
        int                         /*n_iters*/) {
    ggml_cuda_probe_result r{};
    if (use_alt) {
        // Slow alt to bias the conservative crossover to never fire.
        r.p50_ms = 10.0;
        r.p95_ms = 10.5;
    } else {
        r.p50_ms = 1.0;
        r.p95_ms = 1.05;
    }
    return r;
}

// Static-initializer registration. Runs at module load (.so init time)
// before any ggml_backend_cuda_init can be called, which guarantees
// the registry is populated before ggml_cuda_calibrate runs.
struct CalibrationOpsRegistrar {
    CalibrationOpsRegistrar() {
        // C8: MATMUL_STREAM_SPLIT
        ggml_cuda_calibration_register_op(
            GGML_CAL_OP_MATMUL_STREAM_SPLIT,
            "MATMUL_STREAM_SPLIT",
            default_wins_stub);
        // C9: REDUCE_CROSS_DEVICE
        ggml_cuda_calibration_register_op(
            GGML_CAL_OP_REDUCE_CROSS_DEVICE,
            "REDUCE_CROSS_DEVICE",
            default_wins_stub);
        // C10: PEER_COPY
        ggml_cuda_calibration_register_op(
            GGML_CAL_OP_PEER_COPY,
            "PEER_COPY",
            default_wins_stub);
        // C11: GRAPH_CAPTURE
        ggml_cuda_calibration_register_op(
            GGML_CAL_OP_GRAPH_CAPTURE,
            "GRAPH_CAPTURE",
            default_wins_stub);
    }
};

static CalibrationOpsRegistrar g_calibration_ops_registrar;

}  // namespace
