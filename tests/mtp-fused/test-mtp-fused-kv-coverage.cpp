// test-mtp-fused-kv-coverage.cpp
//
// Drives:
//   - mtp_fused_draft.allium KvWritesAreExactlyNSteps (system-wide)
//   - mtp_fused_draft.allium KvWrittenAtPositionsPThroughPPlusN
//     (contract invariant)
//
// One fused round at base position p with n_steps=N must write
// exactly N MTP-layer KV slots, all at consecutive positions
// [p, p+N). No KV writes leak to other ranges.
//
// Implementation must expose an inspection API on the MTP-layer
// KV cache that this test can read:
//
//   LLAMA_API int32_t llama_mtp_kv_pos_max(
//         struct llama_context * ctx,
//         llama_seq_id           seq_id);
//
//   LLAMA_API bool llama_mtp_kv_has_pos(
//         struct llama_context * ctx,
//         llama_seq_id           seq_id,
//         llama_pos              pos);

#include "llama.h"

#include <cassert>
#include <cstdio>

int main() {
    // TODO(spec-driven, RED until implementation lands):
    //
    // 1. Load test fixture model. Reset MTP-layer KV to empty
    //    (seq_rm seq_id=0, pos in [-1, -1)).
    // 2. Capture pre-state: pos_max should be -1.
    // 3. Run one main forward to populate seed_hidden at position p=0.
    // 4. Invoke llama_mtp_fused_draft_invoke with n_steps = N.
    // 5. Assert llama_mtp_kv_pos_max == p + N - 1 (= N - 1 for p=0).
    // 6. For pos in [p, p+N): assert llama_mtp_kv_has_pos == true.
    // 7. For pos in [p+N, p+N+10]: assert llama_mtp_kv_has_pos == false
    //    (no write leakage past the round's range).
    // 8. Repeat at non-zero p to verify the round's base position is
    //    respected (e.g. simulate prefilled state by running a few
    //    main forwards before the fused round).

    fprintf(stderr,
            "TODO: implement KV-coverage test once the fused API and\n"
            "      MTP-KV inspection helpers are available.\n");

    return 77;
}
