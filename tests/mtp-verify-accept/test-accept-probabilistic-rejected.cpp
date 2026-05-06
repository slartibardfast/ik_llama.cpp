// test-accept-probabilistic-rejected.cpp
//
// Drives:
//   - mtp_verify_accept.allium ProbabilisticModeOutOfScope
//
// Property: AcceptMode.probabilistic is rejected at the contract
// boundary. The implementation must return a non-zero error code
// and produce no AcceptDecision until mtp_fused_draft.allium scope
// expands to cover top_k > 1 sampling. (See OQ-1.)
//
// Test strategy:
//   1. Invoke llama_mtp_accept_verify with mode =
//      LLAMA_ACCEPT_MODE_PROBABILISTIC.
//   2. Assert return code != 0.
//   3. Assert *out is unchanged from its sentinel pre-call value.

#include "llama.h"

#include <cstdio>
#include <cstring>

int main() {
    // TODO(spec-driven, RED until implementation lands).
    //
    // Sentinel pattern for the unchanged-out check:
    //   llama_accept_decision d;
    //   memset(&d, 0xAA, sizeof d);
    //   int rc = llama_mtp_accept_verify(ctx, drafts, 3, 0,
    //               LLAMA_ACCEPT_MODE_PROBABILISTIC, &d);
    //   assert(rc != 0);
    //   for (size_t i = 0; i < sizeof d; ++i) assert(((unsigned char *)&d)[i] == 0xAA);
    fprintf(stderr,
            "TODO: implement ProbabilisticModeOutOfScope test once\n"
            "      llama_mtp_accept_verify is callable.\n");
    return 77;
}
