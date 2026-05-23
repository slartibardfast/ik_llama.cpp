// test-pool-bounds-respected.cpp
//
// T5.9.A.4 binding test for paged_kv_pool_sizing.allium
// ::PoolBoundsRespected. Drives the allocator past its capacity
// and asserts every successful alloc returns bid ∈ [0, pool_capacity),
// and that capacity exhaustion is signalled cleanly (OOB sentinel).
//
// Distinct from test-paged-allocator-determinism (which covers
// deterministic re-runs + LIFO + OOM-as-sentinel) in that this test
// asserts a SCALING invariant — bid is bounded by pool_capacity
// across many alloc/free cycles, not just at first exhaustion.
//
// Binds:
//   paged_kv_pool_sizing.allium::PoolBoundsRespected
//     ::SuccessfulAllocInBounds
//     ::BlockTableEntriesInBounds
//   PagedKVAllocator.tla::PoolBoundsRespected
//
// GREEN on HEAD (allocator already bounds; this test exists as
// defence-in-depth catching a T5.9.B regression that would relax
// the bound off-by-one).

#include "../../src/llama-paged-kv-allocator.h"

#include <cstdio>
#include <vector>

int main() {
    constexpr int32_t POOL_CAPACITY = 8;
    constexpr int32_t N_SEQS        = 4;

    llama_paged_kv_allocator a;
    a.init(POOL_CAPACITY, N_SEQS);

    // Phase 1: drive alloc to exhaustion. Every successful alloc
    // returns bid ∈ [0, POOL_CAPACITY); subsequent allocs return
    // OOB_SENTINEL.
    int n_success = 0;
    int n_oob     = 0;
    for (int i = 0; i < POOL_CAPACITY * 3; ++i) {
        const int32_t seq = i % N_SEQS;
        const int32_t bid = a.alloc_block(seq);
        if (bid == llama_paged_kv_allocator::OOB_SENTINEL) {
            ++n_oob;
        } else {
            ++n_success;
            if (bid < 0 || bid >= POOL_CAPACITY) {
                std::printf(
                    "FAIL: SuccessfulAllocInBounds violated — alloc %d "
                    "returned bid=%d outside [0, %d)\n",
                    i, bid, POOL_CAPACITY);
                return 1;
            }
        }
    }
    if (n_success != POOL_CAPACITY) {
        std::printf(
            "FAIL: expected exactly %d successful allocs, got %d (oob=%d)\n",
            POOL_CAPACITY, n_success, n_oob);
        return 1;
    }
    if (n_oob != POOL_CAPACITY * 2) {
        std::printf(
            "FAIL: expected %d OOB sentinels after exhaustion, got %d\n",
            POOL_CAPACITY * 2, n_oob);
        return 1;
    }

    // Phase 2: free all + re-alloc cycle. Every bid in any seq's
    // block_table stays in [0, POOL_CAPACITY) throughout. Drive
    // 100 free+alloc cycles to catch a regression that grows the
    // bid space over time.
    for (int cycle = 0; cycle < 100; ++cycle) {
        for (int32_t s = 0; s < N_SEQS; ++s) {
            a.free_seq(s);
        }
        for (int32_t s = 0; s < N_SEQS; ++s) {
            // Each seq writes 2 blocks worth of tokens.
            const bool ok = a.write_tokens(
                s, a.block_size_tokens() * 2);
            if (!ok) {
                std::printf(
                    "FAIL: cycle %d seq %d write_tokens failed unexpectedly\n",
                    cycle, s);
                return 1;
            }
            // Check every bid in this seq's table is in bounds.
            const auto & bt = a.block_table(s);
            for (size_t i = 0; i < bt.size(); ++i) {
                if (bt[i] < 0 || bt[i] >= POOL_CAPACITY) {
                    std::printf(
                        "FAIL: BlockTableEntriesInBounds violated — "
                        "cycle %d seq %d bt[%zu]=%d outside [0, %d)\n",
                        cycle, s, i, bt[i], POOL_CAPACITY);
                    return 1;
                }
            }
        }
    }

    // Phase 3: verify total_blocks() reports POOL_CAPACITY unchanged
    // after the cycling.
    if (a.total_blocks() != POOL_CAPACITY) {
        std::printf(
            "FAIL: total_blocks() drifted — got %d expected %d\n",
            a.total_blocks(), POOL_CAPACITY);
        return 1;
    }

    std::printf(
        "PASS: PoolBoundsRespected verified — %d successful allocs "
        "stayed within [0, %d), 100 free/realloc cycles produced no "
        "out-of-bounds bid.\n",
        n_success, POOL_CAPACITY);
    return 0;
}
