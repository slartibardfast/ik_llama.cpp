// test-hook-fires-once.cpp
//
// Drives:
//   - mtp_ubatch_hook.allium FiresOncePerUbatch (contract invariant)
//   - mtp_ubatch_hook.allium SingleMtpDecodePerInvocation (contract)
//
// Per ubatch processed by the main model with MTP registered, the
// hook fires exactly once and dispatches exactly one MTP-block
// decode. This test counts both via observability hooks and asserts
// the relationship.

#include "llama.h"

#include <cassert>
#include <cstdio>

int main() {
    // TODO(spec-driven, RED until implementation lands):
    //
    // 1. Load test fixture model with MTP enabled (cparams.mtp = true).
    // 2. Reset counters: hook_fire_count_before = llama_mtp_hook_fire_count(ctx);
    //                    decode_count_before    = llama_mtp_decode_count(ctx);
    // 3. Run llama_decode on a single-ubatch input (n_tokens <= n_ubatch).
    // 4. After llama_decode returns:
    //      hook_fires = llama_mtp_hook_fire_count(ctx) - hook_fire_count_before;
    //      mtp_decodes = llama_mtp_decode_count(ctx) - decode_count_before;
    //    Assert hook_fires == 1.
    //    Assert mtp_decodes == 1 (one MTP-block batched decode per
    //                              hook fire).
    //
    // 5. Run llama_decode on a multi-ubatch input (n_tokens > n_ubatch).
    //    Compute n_expected_ubatches = ceil(n_tokens / n_ubatch).
    //    Assert hook_fires == n_expected_ubatches.
    //    Assert mtp_decodes == n_expected_ubatches.
    //
    // 6. Sanity: with cparams.mtp = false (or no MTP-context registered),
    //    llama_decode must NOT fire the hook even though the main
    //    forward still runs. hook_fires must be 0.

    fprintf(stderr,
            "TODO: implement fires-once test once the hook + observability\n"
            "      counters land.\n");

    return 77;
}
