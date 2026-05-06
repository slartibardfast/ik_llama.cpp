// test-mtp-fused-prob-populated.cpp
//
// Drives:
//   - mtp_fused_draft.allium ProbabilityFieldPopulated (contract)
//
// Every emitted DraftToken must carry a populated prob field equal
// to the softmax probability of the chosen token under the step's
// logits. Required for downstream consumers (ratio-based accept
// tests).
//
// The implementation must NOT leave probs uninitialised. This test
// catches the failure mode where the impl emits tokens via argmax
// but forgets to compute and propagate the probability.

#include "llama.h"

#include <cassert>
#include <cmath>
#include <cstdio>

int main() {
    // TODO(spec-driven, RED until implementation lands):
    //
    // 1. Run llama_mtp_fused_draft_invoke(n_steps=4) on the test
    //    fixture model.
    // 2. For each step k in [0, 4):
    //      a. Verify out.probs[k] is finite (not NaN, not inf).
    //      b. Verify 0.0 < out.probs[k] <= 1.0. (argmax winner cannot
    //         have probability 0; can have probability 1 in a
    //         peaked distribution.)
    //      c. Optionally: compare against host-recomputed softmax of
    //         the step's logits at the chosen token id. Within FP
    //         tolerance (e.g. abs error < 1e-4) should match.
    //
    // 3. Run with n_steps in {2, 3, 5, 6, 7, 8}. The property must
    //    hold at every step in every config.

    fprintf(stderr,
            "TODO: implement prob-populated test once the fused API\n"
            "      and the test fixture are available.\n");

    return 77;
}
