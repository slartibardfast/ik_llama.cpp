// test-accept-determinism.cpp
//
// Drives:
//   - mtp_verify_accept.allium DeterminismUnderFixedLogits
//
// Property: two invocations of llama_mtp_accept_verify with the same
// VerifyUbatch and the same VerifyLogits and the same AcceptMode
// produce the same AcceptDecision (n_accepted, bonus_token, bonus_pos
// all byte-identical). The accept algorithm is a pure function of
// its inputs.
//
// Test strategy:
//   1. Build a deterministic logits buffer + draft_tokens.
//   2. Invoke A; capture decision_a.
//   3. Invoke B with byte-identical inputs; capture decision_b.
//   4. memcmp(&decision_a, &decision_b, sizeof(...)) == 0.
//   5. Repeat with several (logits, draft_tokens) shapes including
//      n_accepted = 0, n_accepted = n_drafts, and intermediate.

#include "llama.h"

#include <cstdio>

int main() {
    // TODO(spec-driven, RED until implementation lands).
    fprintf(stderr,
            "TODO: implement DeterminismUnderFixedLogits test once\n"
            "      llama_mtp_accept_verify is callable with stubbed logits.\n");
    return 77;
}
