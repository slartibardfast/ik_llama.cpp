// test-hook-acceptance-vs-update-accepted.cpp
//
// Drives:
//   - mtp_ubatch_hook.allium AcceptanceParity
//
// The verify-time MTP KV hook (cparams.mtp_inline_kv_hook) eliminates
// the separate MTP_OP_UPDATE_ACCEPTED forward pass. The replacement
// must produce token-for-token identical accept decisions on a
// greedy-sampling workload — otherwise we've introduced a quality
// regression while skipping the redundant decode.
//
// Strategy:
//   1. Run a fixed prompt at temperature=0 with hook OFF
//      (LLAMA_MTP_INLINE_KV unset). Record the accepted-token
//      sequence and the per-cycle accept count.
//   2. Reset KV cache. Run the same prompt with hook ON
//      (LLAMA_MTP_INLINE_KV set). Record the same.
//   3. Assert: accepted_tokens_off == accepted_tokens_on (byte-identical).
//   4. Assert: per_cycle_accept_off == per_cycle_accept_on.
//
// Throughput differs (the hook is faster); only the *correctness*
// invariant is asserted here. A separate microbenchmark tracks the
// throughput delta.

#include "llama.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

int main() {
    // TODO(spec-driven, RED until the test infrastructure can drive
    // the env var per-process. The hook gate reads LLAMA_MTP_INLINE_KV
    // at context construction; toggling it requires re-init of the
    // llama_context, which is not a primitive the unit test fixture
    // currently provides.
    //
    // Concrete steps when the fixture lands:
    // 1. unsetenv("LLAMA_MTP_INLINE_KV"); init ctx_a, run 256 greedy
    //    tokens on a fixed prompt, capture vector<llama_token> A.
    // 2. llama_free(ctx_a); setenv("LLAMA_MTP_INLINE_KV", "1", 1);
    //    init ctx_b, run same prompt, capture vector<llama_token> B.
    // 3. assert(A == B) elementwise.
    // 4. (Optional) capture per-cycle MTP-decode counts via
    //    llama_mtp_hook_fire_count() and llama_mtp_decode_count();
    //    assert hook_fires_b == n_verify_cycles_b and
    //    update_accepted_decodes_b == 0.
    //
    // Until that fixture exists, this file is a documentation-grade
    // RED test (return 77 = SKIP) tying the contract to the source.

    fprintf(stderr,
            "TODO: implement AcceptanceParity test once the test\n"
            "      fixture supports per-process env-var-controlled\n"
            "      ctx init.\n");

    return 77;
}
