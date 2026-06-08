// test-paged-resync-reclaim.cpp
//
// PHASE_HYBRID_STATE_RESTORE regression test (2026-06-07).
//
// Models the per-seq paged reclaim that llama_kv_cache_seq_rm's resync loop
// performs after a [p0, max] prefix trim: free_seq(seq) followed by
// write_tokens(seq, n_surviving). This is the operation that was NEVER RUN for
// n_stream==1 (the --parallel 1 production config) because the new_head_local
// freed-tracker driving the resync was gated under `if (n_stream > 1)`. With
// the gate, write_tokens grew the reservation on every decode and free_seq was
// never called, so the pool monotonically filled until n_free==0 and every
// subsequent restore's find_slot failed -> persistent HTTP 413 at depth on a
// non-full cache.
//
// This test asserts the conservation + identity-preservation contract the fix
// relies on, at n_seqs==1 (single stream):
//   1. After grow (write_tokens) + resync (free_seq + write_tokens(n_keep)),
//      n_free recovers to exactly (total - ceil(n_keep / block)).
//   2. Repeated grow/trim cycles do NOT monotonically drain the pool (the
//      production-bug signature: n_free never recovering).
//   3. A [p0, max] prefix trim preserves block identity for the surviving
//      prefix (the low blocks come back, by LIFO).
//
// Binds the allocator-level contract; the seq_rm GATING fix itself is bound
// end-to-end by scripts/repro-pool-drain.sh (n_free recovers, no 413).

#include "../../src/llama-paged-kv-allocator.h"

#include <cstdio>
#include <vector>

int main() {
    constexpr int32_t POOL = 64;     // blocks
    constexpr int32_t N_SEQS = 1;    // single stream — the production config
    const int32_t BLK = []{ llama_paged_kv_allocator t; t.init(POOL, N_SEQS); return t.block_size_tokens(); }();

    auto need = [&](int32_t toks){ return toks == 0 ? 0 : (toks - 1) / BLK + 1; };

    llama_paged_kv_allocator a;
    a.init(POOL, N_SEQS);

    // --- 1: grow then resync-trim, n_free must recover exactly ---
    const int32_t grow_toks = 40 * BLK;     // reserve 40 blocks
    if (!a.write_tokens(0, grow_toks)) { std::printf("FAIL: initial grow rejected\n"); return 1; }
    if (a.n_free() != POOL - 40) { std::printf("FAIL: post-grow n_free=%d want %d\n", a.n_free(), POOL - 40); return 1; }

    // Record the surviving-prefix block ids before the resync.
    const int32_t keep_toks = 10 * BLK;     // [p0, max] trim leaves 10 blocks
    std::vector<int32_t> prefix_before;
    for (int i = 0; i < 10; ++i) prefix_before.push_back(a.block_id_at(0, i));

    // The resync operation (what seq_rm does, now also for n_stream==1):
    a.free_seq(0);
    if (a.n_free() != POOL) { std::printf("FAIL: free_seq did not return all blocks (n_free=%d)\n", a.n_free()); return 1; }
    if (!a.write_tokens(0, keep_toks)) { std::printf("FAIL: re-reserve rejected\n"); return 1; }
    if (a.n_free() != POOL - need(keep_toks)) {
        std::printf("FAIL: post-resync n_free=%d want %d\n", a.n_free(), POOL - need(keep_toks));
        return 1;
    }

    // 3: identity preservation for the surviving prefix.
    for (int i = 0; i < 10; ++i) {
        if (a.block_id_at(0, i) != prefix_before[i]) {
            std::printf("FAIL: prefix block %d changed identity (%d -> %d)\n",
                        i, prefix_before[i], a.block_id_at(0, i));
            return 1;
        }
    }

    // --- 2: many grow/trim cycles must NOT monotonically drain ---
    // Without the seq_rm resync this is exactly what bit production: each cycle
    // would consume blocks that are never returned. Here we model the fixed
    // lifecycle and assert the pool is fully reclaimable every cycle.
    for (int cycle = 0; cycle < 200; ++cycle) {
        a.free_seq(0);
        if (a.n_free() != POOL) {
            std::printf("FAIL: cycle %d free_seq left n_free=%d (pool drained — the production bug)\n",
                        cycle, a.n_free());
            return 1;
        }
        // Grow to near-full, as a deep conversation would.
        if (!a.write_tokens(0, (POOL - 1) * BLK)) {
            std::printf("FAIL: cycle %d grow to near-full rejected\n", cycle);
            return 1;
        }
    }

    std::printf("PASS: paged resync reclaim — n_free recovers, identity preserved, no monotonic drain over 200 cycles\n");
    return 0;
}
