// test-mtp-fused-single-compute.cpp
//
// Drives:
//   - mtp_fused_draft.allium SingleGraphCompute (contract invariant)
//   - mtp_fused_draft.allium NoSyncBetweenSteps (contract invariant)
//
// One invocation of FusedDraftStep must issue exactly ONE
// ggml_graph_compute call regardless of n_steps. Multiple compute
// calls within one invocation defeat the entire purpose of fusion.
//
// The test relies on llama_mtp_fused_last_compute_count() reporting
// the count of graph_compute calls made during the most recent
// invoke. Implementation must wire this counter from
// ggml_backend_sched_graph_compute_async (or its analog) into a
// per-context stat read by this hook.
//
// This test is gated behind GGML_CUDA because the count is a
// CUDA-backend property. CPU backend is out of contract per OQ-3.

#include "llama.h"

#include <cassert>
#include <cstdio>

int main() {
    // TODO(spec-driven, RED until implementation lands):
    // 1. Load a small Qwen 3.6-class GGUF with MTP head preserved.
    //    Recommend the 0.8B test fixture from
    //    /opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0-MTP.gguf or the
    //    test-backend-ops minimal model fixture if it covers the
    //    nextn_predict_layers metadata path.
    // 2. Issue one main forward to produce a t_h_pre_norm row.
    // 3. Pull that row to host into a buffer.
    // 4. Run llama_mtp_fused_draft_invoke for n_steps in [2, 4, 8].
    // 5. After each invoke, query llama_mtp_fused_last_compute_count.
    // 6. Assert count == 1 in every case.
    //
    // Until the model fixture and the API exist, this test fails to
    // link. That is the test-first signal.

    fprintf(stderr,
            "TODO: implement SingleGraphCompute test once the fused API\n"
            "      and the test-fixture model are available. The harness\n"
            "      shape is described in the comment block above.\n");

    // Force RED: any caller running this test before implementation
    // sees a non-zero exit. Once the implementation lands, replace
    // this stub with the actual harness body.
    return 77;  // skip code under ctest, reads as test infra not ready
}
