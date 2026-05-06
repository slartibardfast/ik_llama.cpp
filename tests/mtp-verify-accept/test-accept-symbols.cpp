// test-accept-symbols.cpp
//
// Compile-only RED test for mtp_verify_accept.allium contract presence.
//
// Drives:
//   - contract-signature.AcceptVerify.invoke
//   - existence of enum llama_accept_mode { argmax_match | probabilistic }
//   - existence of struct llama_accept_decision { n_accepted, bonus_token, bonus_pos }
//   - existence of llama_mtp_accept_verify() API
//
// This test does not exercise behaviour. It fails to compile until the
// implementation adds the named symbols. Once it compiles + links the
// runtime body returns 0 (the symbol surface is the GREEN signal here);
// the per-invariant tests in this bundle then drive behavioural slices.

#include "llama.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

int main() {
    // 1. Enum llama_accept_mode must exist with both variants distinct.
    //    See mtp_verify_accept.allium AcceptMode enum.
    static_assert(LLAMA_ACCEPT_MODE_ARGMAX_MATCH != LLAMA_ACCEPT_MODE_PROBABILISTIC,
                  "argmax_match and probabilistic must be distinct variants");

    // 2. Struct llama_accept_decision must expose the three result fields.
    //    See mtp_verify_accept.allium AcceptDecision external entity.
    llama_accept_decision d{};
    (void) d.n_accepted;
    (void) d.bonus_token;
    (void) d.bonus_pos;

    // 3. API entry point llama_mtp_accept_verify must be linkable.
    //    See mtp_verify_accept.allium AcceptVerify.invoke signature:
    //      (ub: VerifyUbatch, logits: VerifyLogits, mode: AcceptMode) -> AcceptDecision
    //    The C-shape passes draft_tokens + n_drafts + pos_seed (the
    //    VerifyUbatch fields the host needs) and reads logits inline
    //    via llama_get_logits_ith / llama_get_draft_argmax inside ctx.
    typedef int32_t (*invoke_fn_t)(
            struct llama_context        *,
            const llama_token           *,
            int32_t,
            int32_t,
            enum llama_accept_mode,
            struct llama_accept_decision *);
    invoke_fn_t fn = &llama_mtp_accept_verify;
    if (fn == nullptr) {
        fprintf(stderr,
                "FAIL: llama_mtp_accept_verify null after &-take\n");
        return 1;
    }

    printf("=== test-accept-symbols ===\n");
    printf("  llama_accept_mode enum present\n");
    printf("  llama_accept_decision struct present\n");
    printf("  llama_mtp_accept_verify linkable\n");
    printf("  GREEN — symbol surface exists\n");
    return 0;
}
