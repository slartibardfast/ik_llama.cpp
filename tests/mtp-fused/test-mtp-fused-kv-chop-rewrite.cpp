// test-mtp-fused-kv-chop-rewrite.cpp
//
// Drives:
//   - mtp_fused_draft.allium KvChopRewriteIsIdempotent (system-wide)
//
// Round 1 writes MTP-layer KV at p..p+N-1.
// Verify accepts k drafts; chop removes [p+k, p+N-1].
// Round 2 writes p+k..p+k+N-1.
//
// The resulting MTP-KV state must equal the reference state where
// round 1 only ever wrote p..p+k-1 in the first place, then round 2
// wrote p+k..p+k+N-1. Backend implementations may not introduce
// stale-data artefacts in the chopped region.
//
// Test strategy:
//   path A: full round 1 + chop + round 2 (production path)
//   path B: round 1 truncated (only first k drafts written) + round 2
//   compare KV state byte-for-byte

#include "llama.h"

#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    // TODO(spec-driven, RED until implementation lands):
    //
    // The implementation needs an MTP-KV inspection API that returns
    // the raw bytes of a slot for byte-equality comparison:
    //
    //   LLAMA_API bool llama_mtp_kv_get_slot(
    //         struct llama_context * ctx,
    //         llama_seq_id           seq_id,
    //         llama_pos              pos,
    //         void                 * out_k,    // K-row bytes
    //         void                 * out_v,    // V-row bytes
    //         size_t                 stride);
    //
    // Path A:
    //   1. Reset MTP KV. Capture seed_hidden at p=0.
    //   2. Run llama_mtp_fused_draft_invoke(n_steps=4) at p=0.
    //   3. Simulate verify result with k_accepted=2.
    //   4. Call llama_kv_cache_seq_rm(ctx_mtp, seq_id=0, p0=2, p1=-1).
    //   5. Run main forward to advance to p=2.
    //   6. Capture seed_hidden_2 at p=2.
    //   7. Run llama_mtp_fused_draft_invoke(n_steps=4) at p=2.
    //   8. Snapshot MTP KV bytes for positions [0..5] -> bytes_a.
    //
    // Path B:
    //   1. Reset MTP KV. Capture seed_hidden at p=0.
    //   2. Run llama_mtp_fused_draft_invoke(n_steps=2) at p=0
    //      (write only the prefix that would have been accepted).
    //   3. Run main forward to advance to p=2.
    //   4. Capture seed_hidden_2 at p=2 (must be byte-identical to
    //      path A's seed_hidden_2 — depends on identical main-forward
    //      determinism).
    //   5. Run llama_mtp_fused_draft_invoke(n_steps=4) at p=2.
    //   6. Snapshot MTP KV bytes for positions [0..5] -> bytes_b.
    //
    // Assert: memcmp(bytes_a, bytes_b, total_bytes) == 0.
    //
    // Failure mode: if the chop operation merely flags positions as
    // "not present" without zeroing the underlying buffer, a future
    // round-2 write that overlaps could produce different attention
    // mask treatment, leaving a stale-data artefact. This is what
    // KvChopRewriteIsIdempotent guards against.

    fprintf(stderr,
            "TODO: implement KV chop+rewrite test once the fused API\n"
            "      and llama_mtp_kv_get_slot inspection helper exist.\n");

    return 77;
}
