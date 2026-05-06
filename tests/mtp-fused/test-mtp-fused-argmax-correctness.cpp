// test-mtp-fused-argmax-correctness.cpp
//
// Drives:
//   - mtp_fused_draft.allium ArgmaxCorrectnessIntrinsic (system-wide)
//   - mtp_fused_draft.allium ArgmaxWinnerWhenArgmaxOnly (contract)
//   - mtp_fused_draft.allium NTokensProducedInOrder (contract)
//
// For each step k of a fused round under argmax-only sampling, the
// emitted token_id must equal argmax over the step's logits, with
// ties broken by lowest token id. The test recomputes the argmax
// host-side from the cached logit tensor and compares.
//
// This is a property-style test: parameterise n_steps in
// [2, 8] and assert the property at every step in every config.

#include "llama.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <vector>

int main() {
    // TODO(spec-driven, RED until implementation lands):
    // The implementation must expose, alongside llama_mtp_fused_result,
    // an inspection API that returns the per-step logits tensor (or a
    // host-side copy of it) so this test can recompute argmax and
    // compare. Suggested signature in include/llama.h:
    //
    //   LLAMA_API bool llama_mtp_fused_get_step_logits(
    //         struct llama_context * ctx,
    //         int32_t                step_index,
    //         float                * out_logits,    // n_vocab floats
    //         int32_t                n_vocab);
    //
    // Once the inspection API exists:
    //   1. Load the test fixture model (same as test-mtp-fused-single-compute).
    //   2. Run one main forward, capture t_h_pre_norm row.
    //   3. Run llama_mtp_fused_draft_invoke at n_steps = 4.
    //   4. For each step k in [0, 3]:
    //        a. Read out.tokens[k] (the emitted draft token).
    //        b. Pull step-k logits via llama_mtp_fused_get_step_logits.
    //        c. Compute argmax host-side, breaking ties by lowest id.
    //        d. Assert host_argmax == out.tokens[k].
    //   5. Repeat for n_steps in {2, 3, 5, 6, 7, 8}. Each must satisfy
    //      the property independently.
    //   6. Verify out.n_steps == requested n_steps (NTokensProducedInOrder).

    fprintf(stderr,
            "TODO: implement ArgmaxCorrectnessIntrinsic test once the\n"
            "      fused API + step-logits inspection API land. Harness\n"
            "      shape described above.\n");

    return 77;  // ctest skip; harness not ready
}
