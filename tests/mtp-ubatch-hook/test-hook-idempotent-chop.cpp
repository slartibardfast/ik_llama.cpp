// test-hook-idempotent-chop.cpp
//
// Drives:
//   - mtp_ubatch_hook.allium IdempotentUnderRepeatedChop (contract on
//     KvRangeChop)
//
// Calling the post-accept chop (llama_kv_cache_seq_rm) twice with the
// same arguments must yield the same KV state as calling it once.
// Guards against duplicated rejection signals corrupting state.

#include "llama.h"

#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
    // TODO(spec-driven, RED until implementation lands):
    //
    // 1. Load test fixture model, populate MTP-KV via a fused round
    //    at p=0, n_steps=4. Now MTP-KV covers positions [0..3].
    // 2. Snapshot mtp_kv_pos_max_a = llama_mtp_kv_pos_max(ctx, 0).
    //    Must equal 3.
    // 3. Call llama_kv_cache_seq_rm(ctx_mtp, seq_id=0, p0=2, p1=-1).
    // 4. Snapshot mtp_kv_pos_max_b = llama_mtp_kv_pos_max(ctx, 0).
    //    Must equal 1.
    // 5. Call llama_kv_cache_seq_rm(ctx_mtp, seq_id=0, p0=2, p1=-1)
    //    AGAIN with identical arguments.
    // 6. Snapshot mtp_kv_pos_max_c = llama_mtp_kv_pos_max(ctx, 0).
    //    Must still equal 1.
    // 7. For pos in [0, 1]: llama_mtp_kv_has_pos must still be true.
    // 8. For pos in [2, 3]: llama_mtp_kv_has_pos must still be false.
    //
    // Failure mode: if the chop accidentally underflows pos_max on the
    // second call, or if the seq_rm walks past pos_max-1 and corrupts
    // an unrelated slot, this test fires.

    fprintf(stderr,
            "TODO: implement idempotent-chop test once the fused API\n"
            "      and MTP-KV inspection helpers land.\n");

    return 77;
}
