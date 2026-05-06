// test-accept-bonus-correctness.cpp
//
// Drives:
//   - mtp_verify_accept.allium BonusIsArgmaxAtFirstUnacceptedRow
//
// Property: bonus_token = argmax(logits at row n_accepted), with
// ties broken by lowest token id. Two cases:
//   (a) n_accepted < n_drafts: row n_accepted is the rejection site;
//       its argmax is what the model "wants" instead of the rejected
//       draft.
//   (b) n_accepted = n_drafts: every draft matched; row n_drafts is
//       the LAST logit row, predicting the next-next-token after
//       the last accepted draft.
//
// Test strategy:
//   1. Build a logits matrix with chosen argmax per row.
//   2. Build draft_tokens such that exactly k drafts match argmax;
//      assert decision.bonus_token == argmax(row k).
//   3. Construct a tie at row k (two tokens with identical max
//      probability); assert bonus_token == lower id.
//   4. Sweep k from 0 to n_drafts inclusive (both cases).

#include "llama.h"

#include <cstdio>

int main() {
    // TODO(spec-driven, RED until implementation lands).
    fprintf(stderr,
            "TODO: implement BonusIsArgmaxAtFirstUnacceptedRow test once\n"
            "      llama_mtp_accept_verify is callable with stubbed logits.\n");
    return 77;
}
