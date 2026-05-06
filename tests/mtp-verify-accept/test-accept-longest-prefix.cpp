// test-accept-longest-prefix.cpp
//
// Drives:
//   - mtp_verify_accept.allium LongestPrefixMatchUnderArgmax
//   - mtp_verify_accept.allium NAcceptedWithinBound
//
// Property: for any (VerifyUbatch ub, VerifyLogits logits) under
// AcceptMode.argmax_match, n_accepted is the largest k in [0, n_drafts]
// such that for every i in [0, k): draft_tokens[i] = argmax(logits at row i).
// Ties broken by lowest token id.
//
// Test strategy (host-side, no GPU required):
//   1. Synthesise logits as a small (n_positions, n_vocab) float array
//      with chosen argmax indices.
//   2. Build draft_tokens with a controlled match prefix length k.
//   3. Run llama_mtp_accept_verify; assert decision.n_accepted == k.
//   4. Assert 0 <= n_accepted <= n_drafts.
//   5. Sweep k across [0, n_drafts]; include the all-accepted case.
//
// The test does not need an llama_context with a real model — it
// needs the implementation to expose either (a) a host-side helper
// that runs the comparison given a logits buffer + draft_tokens, or
// (b) a fixture ctx where the verify forward output can be stubbed.
// (b) is the more general path; (a) is a useful seam for unit tests.
//
// If the implementation chooses (a), this test calls it directly. If
// only (b) is available, this test must coordinate with a fixture
// model and is upgraded to a server-level integration test.

#include "llama.h"

#include <cstdio>

int main() {
    // TODO(spec-driven, RED until implementation lands):
    // Implement once one of:
    //   - llama_mtp_accept_verify_from_logits(...)  [host-helper seam]
    //   - llama_mtp_accept_verify(...) on fixture ctx with stubbed logits
    // is available.
    fprintf(stderr,
            "TODO: implement LongestPrefixMatchUnderArgmax test once the\n"
            "      llama_mtp_accept_verify implementation lands.\n");
    return 77;
}
