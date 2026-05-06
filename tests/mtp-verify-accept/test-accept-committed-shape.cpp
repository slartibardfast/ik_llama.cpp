// test-accept-committed-shape.cpp
//
// Drives:
//   - mtp_verify_accept.allium TotalCommittedIsNAcceptedPlusOne
//   - mtp_verify_accept.allium CommittedSequenceShape
//
// Properties (system-wide):
//   1. Each successful AcceptVerify commits exactly n_accepted + 1
//      tokens to the sequence.
//   2. The committed sequence is the accepted prefix of draft_tokens
//      followed by bonus_token. The seed_token was committed in the
//      prior round and is not re-counted.
//
// Test strategy:
//   1. Stage a controlled invocation with known n_accepted.
//   2. Inspect the committed-tokens sink (debug endpoint or test
//      fixture's recorded commit log) and assert:
//        len(committed) == n_accepted + 1
//        committed[0..n_accepted] == draft_tokens[0..n_accepted]
//        committed[n_accepted] == bonus_token
//   3. Repeat for n_accepted = 0, n_accepted = n_drafts.

#include "llama.h"

#include <cstdio>

int main() {
    // TODO(spec-driven, RED until implementation lands).
    //
    // This test needs a commit log seam. The implementation can
    // either:
    //   - expose a debug-build-only callback set via
    //     llama_set_commit_callback(ctx, fn) that is invoked once
    //     per committed token, OR
    //   - record a per-context commit history accessible via
    //     llama_get_recent_commits(ctx, n).
    // The test implementer should pick the lighter seam; if neither
    // is acceptable, this test escalates to the server-level
    // integration tier (test-accept-chop-coordination.sh pattern).
    fprintf(stderr,
            "TODO: implement TotalCommittedIsNAcceptedPlusOne +\n"
            "      CommittedSequenceShape tests once a commit-log seam\n"
            "      is exposed by llama_mtp_accept_verify.\n");
    return 77;
}
