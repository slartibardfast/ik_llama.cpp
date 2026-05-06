// test-accept-logits-shape.cpp
//
// Drives:
//   - mtp_verify_accept.allium LogitsShapeMatchesUbatch
//
// The verify forward emits one logit row per input position. For a
// VerifyUbatch with n_drafts drafts (total length n_drafts + 1
// including the seed), llama_get_logits_ith(ctx, k) for k in
// [0, n_drafts] must each return a non-null pointer to a row of
// n_vocab floats. The accept-decision relies on this to reach row
// index n_drafts when every draft is accepted (the all-accepted
// bonus case).
//
// Test strategy:
//   1. Decode a verify ubatch of length N+1 with logits_all=true.
//   2. For k in [0, N], confirm llama_get_logits_ith(ctx, k) != nullptr.
//   3. For k = N+1 (out-of-range), confirm nullptr.

#include "llama.h"

#include <cstdio>

int main() {
    // TODO(spec-driven, RED until implementation lands):
    // 1. Build a fixture context with a small dummy model.
    // 2. Submit a verify ubatch of length n_drafts + 1 with
    //    llama_batch_init/llama_decode and ensure logits_all=true.
    // 3. Iterate the per-row logits pointer and assert non-null
    //    for valid k, null for invalid k.
    fprintf(stderr,
            "TODO: implement LogitsShapeMatchesUbatch test once the\n"
            "      llama_mtp_accept_verify implementation lands.\n");
    return 77;
}
