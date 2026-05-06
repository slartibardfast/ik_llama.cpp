// test-hook-no-secondary-decode.cpp
//
// Drives:
//   - mtp_ubatch_hook.allium NoSecondaryDecode (contract invariant on
//     KvRangeChop)
//
// Critical anti-regression test. The whole point of the hook
// architecture is that the post-accept path is a memory-state
// operation only — llama_kv_cache_seq_rm — with NO additional MTP
// forward. If the implementation regresses to a post-accept MTP
// decode (the old MTP_OP_UPDATE_ACCEPTED path), this test fires.
//
// Test strategy:
//   1. Run a verify cycle on a 4-draft batch.
//   2. Count MTP decodes during the entire cycle.
//   3. Expect exactly 1 (the in-hook MTP decode; the chop is NOT a
//      decode).
//   Compare to the pre-hook baseline (would have been 2: hook decode
//   + post-accept update_accepted decode).

#include "llama.h"

#include <cassert>
#include <cstdio>

int main() {
    // TODO(spec-driven, RED until implementation lands):
    //
    // 1. Load test fixture model with MTP enabled.
    // 2. Pre-fill so verify can run with reasonable accepts.
    // 3. Snapshot mtp_decodes_before = llama_mtp_decode_count(ctx).
    // 4. Build a 4-token verify batch (1 last-accepted + 3 drafts).
    //    Submit via llama_decode.
    // 5. Read verify logits, decide accept count (e.g. simulate
    //    n_accepted=2).
    // 6. Call llama_kv_cache_seq_rm(ctx_mtp, seq_id, p+2, -1)
    //    to chop rejected suffix.
    // 7. Snapshot mtp_decodes_after = llama_mtp_decode_count(ctx).
    // 8. Assert mtp_decodes_after - mtp_decodes_before == 1.
    //    (One hook-driven MTP decode during the verify forward; zero
    //    further decodes for the chop.)
    //
    // Failure mode: if the implementation falls back to issuing a
    // separate llama_decode(ctx_mtp, accepted_batch) post-accept,
    // the count is 2 and this test catches it.

    fprintf(stderr,
            "TODO: implement no-secondary-decode test once the hook\n"
            "      and llama_mtp_decode_count counter land.\n");

    return 77;
}
