// test-accept-progress.cpp
//
// Drives:
//   - mtp_verify_accept.allium VerifyAlwaysMakesProgress
//
// Property: each verify round advances the committed position by at
// least 1 (the bonus always commits) and at most n_drafts + 1.
// Speculative decoding cannot stall.
//
// Test strategy:
//   1. Synthesise a worst-case VerifyLogits where the very first row
//      contradicts draft_tokens[0]. Assert decision.n_accepted == 0
//      and decision.n_accepted + 1 >= 1.
//   2. Synthesise a best-case where every row matches.
//      Assert decision.n_accepted == n_drafts and
//      decision.n_accepted + 1 == n_drafts + 1.
//   3. Sweep with pseudo-random logits, assert
//      0 <= decision.n_accepted <= n_drafts.

#include "llama.h"

#include <cstdio>

int main() {
    // TODO(spec-driven, RED until implementation lands).
    fprintf(stderr,
            "TODO: implement VerifyAlwaysMakesProgress test once\n"
            "      llama_mtp_accept_verify is callable with stubbed logits.\n");
    return 77;
}
