// test-mtp-fused-step-count-bound.cpp
//
// Drives:
//   - mtp_fused_draft.allium StepCountWithinBound (contract invariant)
//   - mtp_fused_draft.allium StepCountConformsToBound (system-wide)
//
// The fused draft path applies only for n_steps in
// [config.min_fused_steps, config.max_fused_steps] (default 2..8).
// Outside this range, the implementation must reject the call with a
// non-zero return code, leaving no side effects.
//
// RED-first: today no llama_mtp_fused_draft_invoke exists; link error.
// After the API lands but before bound-checking is wired, the in-bound
// case may pass while the out-of-bound case fails to error — keeping
// this test RED. Bound-checking is the slice that turns it GREEN.

#include "llama.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// These bounds match config.min_fused_steps / config.max_fused_steps in
// mtp_fused_draft.allium. The implementation must publish the same
// bounds (e.g. as LLAMA_MTP_FUSED_MIN_STEPS / LLAMA_MTP_FUSED_MAX_STEPS
// in include/llama.h) so this test stays in sync with the spec.
constexpr int32_t SPEC_MIN_FUSED_STEPS = 2;
constexpr int32_t SPEC_MAX_FUSED_STEPS = 8;

int main() {
    // We don't load a real model in this test. The bound check must
    // be enforced before the call reaches the model — i.e. it's a
    // pure parameter validation.
    //
    // Pass a dummy ctx. The implementation must reject n_steps out of
    // bounds without dereferencing ctx. If it dereferences first, the
    // test crashes and is correctly diagnosed as RED.

    struct llama_context * dummy_ctx = nullptr;
    float dummy_hidden[8192]{};   // generous; n_embd <= 8192 covers Qwen 3.6
    llama_token seed = 42;
    llama_mtp_fused_result out{};

    // Below-min: n_steps = 1 must reject (fused path requires at least 2).
    int32_t rc_below = llama_mtp_fused_draft_invoke(
            dummy_ctx, seed, dummy_hidden, /*n_steps=*/ 1, &out);
    if (rc_below == 0) {
        fprintf(stderr,
                "FAIL: n_steps=1 below min (%d) but invoke returned 0\n",
                SPEC_MIN_FUSED_STEPS);
        return 1;
    }

    // Above-max: n_steps = max+1 must reject.
    int32_t rc_above = llama_mtp_fused_draft_invoke(
            dummy_ctx, seed, dummy_hidden,
            /*n_steps=*/ SPEC_MAX_FUSED_STEPS + 1, &out);
    if (rc_above == 0) {
        fprintf(stderr,
                "FAIL: n_steps=%d above max but invoke returned 0\n",
                SPEC_MAX_FUSED_STEPS + 1);
        return 1;
    }

    // Negative: n_steps = -1 must reject.
    int32_t rc_neg = llama_mtp_fused_draft_invoke(
            dummy_ctx, seed, dummy_hidden, /*n_steps=*/ -1, &out);
    if (rc_neg == 0) {
        fprintf(stderr, "FAIL: n_steps=-1 but invoke returned 0\n");
        return 1;
    }

    // Zero: n_steps = 0 must reject.
    int32_t rc_zero = llama_mtp_fused_draft_invoke(
            dummy_ctx, seed, dummy_hidden, /*n_steps=*/ 0, &out);
    if (rc_zero == 0) {
        fprintf(stderr, "FAIL: n_steps=0 but invoke returned 0\n");
        return 1;
    }

    // out must have been left untouched on rejection. Spec wording:
    // "Outside this range the contract does not apply." We choose the
    // strict reading: no partial output on rejection.
    if (out.n_steps != 0) {
        fprintf(stderr,
                "FAIL: rejected call left out.n_steps=%d (expected 0)\n",
                out.n_steps);
        return 1;
    }

    printf("=== test-mtp-fused-step-count-bound ===\n");
    printf("  n_steps=1 rejected (below min %d): rc=%d\n",
           SPEC_MIN_FUSED_STEPS, rc_below);
    printf("  n_steps=%d rejected (above max %d): rc=%d\n",
           SPEC_MAX_FUSED_STEPS + 1, SPEC_MAX_FUSED_STEPS, rc_above);
    printf("  n_steps=-1 rejected: rc=%d\n", rc_neg);
    printf("  n_steps=0 rejected: rc=%d\n", rc_zero);
    printf("  GREEN — bound enforcement holds\n");
    return 0;
}
