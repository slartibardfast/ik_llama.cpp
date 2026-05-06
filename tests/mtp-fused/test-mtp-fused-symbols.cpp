// test-mtp-fused-symbols.cpp
//
// Compile-only RED test for mtp_fused_draft.allium contract presence.
//
// Drives:
//   - contract-signature.FusedDraftStep.invoke
//   - existence of LLAMA_MTP_OP_DRAFT_GEN_FUSED variant
//   - existence of llama_mtp_fused_draft_invoke() API
//   - existence of llama_mtp_fused_result struct with required fields
//
// This test does not exercise behaviour. It fails to compile until the
// implementation adds the named symbols. Once it compiles, the runtime
// body still exits non-zero (pointing at the missing implementation
// link), so the test stays RED until both the API and the runtime are
// in place.
//
// Test-first contract: this is the FIRST file that becomes GREEN when
// the implementation lands. It documents the minimal API surface the
// rest of the suite depends on.

#include "llama.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

int main() {
    // 1. Enum variant LLAMA_MTP_OP_DRAFT_GEN_FUSED must exist.
    //    See mtp_fused_draft.allium MtpOpType enum, ideal-state
    //    membership { none | draft_gen_fused }.
    static_assert(LLAMA_MTP_OP_DRAFT_GEN_FUSED != LLAMA_MTP_OP_NONE,
                  "LLAMA_MTP_OP_DRAFT_GEN_FUSED must be a distinct enum variant");

    // 2. Struct llama_mtp_fused_result must expose n_steps, tokens, probs.
    //    See mtp_fused_draft.allium DraftToken { step_index, token_id, prob }.
    llama_mtp_fused_result r{};
    (void) r.n_steps;
    (void) r.tokens[0];
    (void) r.probs[0];

    // 3. API entry point llama_mtp_fused_draft_invoke must be linkable.
    //    See mtp_fused_draft.allium FusedDraftStep.invoke signature.
    //    This is a link-only check; we never actually call it with a
    //    real ctx in this test.
    typedef int32_t (*invoke_fn_t)(
            struct llama_context *,
            llama_token,
            const float *,
            int32_t,
            struct llama_mtp_fused_result *);
    invoke_fn_t fn = &llama_mtp_fused_draft_invoke;
    if (fn == nullptr) {
        fprintf(stderr,
                "FAIL: llama_mtp_fused_draft_invoke null after &-take\n");
        return 1;
    }

    // 4. Observability hook llama_mtp_fused_last_compute_count must exist.
    //    Used by test-mtp-fused-single-compute to assert SingleGraphCompute.
    typedef int32_t (*count_fn_t)(struct llama_context *);
    count_fn_t cf = &llama_mtp_fused_last_compute_count;
    if (cf == nullptr) {
        fprintf(stderr,
                "FAIL: llama_mtp_fused_last_compute_count null after &-take\n");
        return 1;
    }

    printf("=== test-mtp-fused-symbols ===\n");
    printf("  LLAMA_MTP_OP_DRAFT_GEN_FUSED present\n");
    printf("  llama_mtp_fused_result struct present\n");
    printf("  llama_mtp_fused_draft_invoke linkable\n");
    printf("  llama_mtp_fused_last_compute_count linkable\n");
    printf("  GREEN — symbol surface exists\n");
    return 0;
}
