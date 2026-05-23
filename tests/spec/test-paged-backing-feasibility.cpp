// test-paged-backing-feasibility.cpp
//
// T5.9.A.4 binding test for paged_kv_pool_sizing.allium
// ::PoolBoundsRespected under a feasibility-shaped workload — a
// smaller proxy for the GP5.9.feasibility gate's ctx 8M NP=8
// pattern. Drives the allocator with concurrent alloc/free across
// many seqs in a pool that's deliberately smaller than the
// nominal-max under the workload's logical demand, and asserts:
//
//   - allocator never crashes or asserts
//   - in-flight block count never exceeds pool_capacity
//   - all in-flight bids stay in [0, pool_capacity)
//   - exhaustion is signalled cleanly via OOB sentinel
//   - free-and-realloc reaches the pool capacity in steady state
//
// Distinct from test-pool-bounds-respected (which is a single
// alloc-to-exhaustion sweep + free/realloc cycles) in that this
// test models a realistic high-ctx pattern: many seqs running
// concurrently, with periodic prompt completions freeing blocks.
//
// Binds:
//   paged_kv_pool_sizing.allium::PoolBoundsRespected
//   paged_block_allocator.allium::HighCtxFeasibility
//     ::ActualFootprintBoundedByWrites
//     ::OOMBehaviourIsFirstClass
//
// GREEN on HEAD (allocator already handles this pattern; test
// exists as defence-in-depth against T5.9.B regression that
// might lose track of in-flight count under buffer shrinkage).

#include "../../src/llama-paged-kv-allocator.h"

#include <cstdio>
#include <vector>

int main() {
    // Smaller proxy for ctx 8M NP=8 / 64 = 1M blocks.
    // 256 blocks × 64 tokens = 16K logical capacity.
    constexpr int32_t POOL_CAPACITY = 256;
    constexpr int32_t N_SEQS        = 8;
    constexpr int32_t N_OPS         = 1000;

    llama_paged_kv_allocator a;
    a.init(POOL_CAPACITY, N_SEQS);

    // Track in-flight block count and OOB events. The workload runs
    // in three phases to actually exercise both saturation AND the
    // free/realloc cycling pattern under saturation.
    int n_alloc_success = 0;
    int n_alloc_oob     = 0;
    int n_free_ops      = 0;
    int peak_in_flight  = 0;

    auto verify_state = [&](const char * where, int op) -> int {
        int32_t in_flight = 0;
        for (int32_t s = 0; s < N_SEQS; ++s) {
            in_flight += a.n_blocks_owned_by(s);
            const auto & bt = a.block_table(s);
            for (size_t i = 0; i < bt.size(); ++i) {
                if (bt[i] < 0 || bt[i] >= POOL_CAPACITY) {
                    std::printf(
                        "FAIL: %s op %d seq %d bt[%zu]=%d outside [0, %d)\n",
                        where, op, s, i, bt[i], POOL_CAPACITY);
                    return 1;
                }
            }
        }
        if (in_flight > POOL_CAPACITY) {
            std::printf(
                "FAIL: %s op %d in-flight=%d exceeds POOL_CAPACITY=%d\n",
                where, op, in_flight, POOL_CAPACITY);
            return 1;
        }
        if (a.n_free() + in_flight != POOL_CAPACITY) {
            std::printf(
                "FAIL: %s op %d n_free=%d + in_flight=%d != POOL_CAPACITY=%d\n",
                where, op, a.n_free(), in_flight, POOL_CAPACITY);
            return 1;
        }
        if (in_flight > peak_in_flight) peak_in_flight = in_flight;
        return 0;
    };

    // Phase 1: fill pool to saturation + overfill to drive OOB.
    // Round-robin across seqs; allocate more than POOL_CAPACITY blocks
    // so at least one OOB sentinel must appear.
    for (int op = 0; op < POOL_CAPACITY + 64; ++op) {
        const int32_t seq = op % N_SEQS;
        const int32_t bid = a.alloc_block(seq);
        if (bid == llama_paged_kv_allocator::OOB_SENTINEL) {
            ++n_alloc_oob;
        } else {
            ++n_alloc_success;
            if (bid < 0 || bid >= POOL_CAPACITY) {
                std::printf(
                    "FAIL: phase1 op %d alloc returned bid=%d outside "
                    "[0, %d)\n",
                    op, bid, POOL_CAPACITY);
                return 1;
            }
        }
        if (verify_state("phase1", op) != 0) return 1;
    }

    // Phase 2: free-and-realloc cycles under saturation. Each cycle
    // frees one seq, then a different seq attempts to fill up the
    // freed slots. This pattern mirrors a server tick where one slot
    // completes and another arrives.
    for (int cycle = 0; cycle < 100; ++cycle) {
        const int32_t seq_to_free = cycle % N_SEQS;
        a.free_seq(seq_to_free);
        ++n_free_ops;
        if (verify_state("phase2-free", cycle) != 0) return 1;

        const int32_t seq_to_fill = (cycle + 4) % N_SEQS;
        // Fill until OOB.
        while (true) {
            const int32_t bid = a.alloc_block(seq_to_fill);
            if (bid == llama_paged_kv_allocator::OOB_SENTINEL) {
                ++n_alloc_oob;
                break;
            }
            ++n_alloc_success;
            if (bid < 0 || bid >= POOL_CAPACITY) {
                std::printf(
                    "FAIL: phase2 cycle %d alloc bid=%d outside [0, %d)\n",
                    cycle, bid, POOL_CAPACITY);
                return 1;
            }
        }
        if (verify_state("phase2-fill", cycle) != 0) return 1;
    }

    // Phase 3: drain to empty, verifying every block returns to free.
    for (int32_t s = 0; s < N_SEQS; ++s) {
        a.free_seq(s);
        ++n_free_ops;
        if (verify_state("phase3", s) != 0) return 1;
    }
    if (a.n_free() != POOL_CAPACITY) {
        std::printf(
            "FAIL: post-drain n_free=%d != POOL_CAPACITY=%d\n",
            a.n_free(), POOL_CAPACITY);
        return 1;
    }
    (void)N_OPS;

    // OOMBehaviourIsFirstClass: the workload should have HIT the
    // exhaustion path at least once. If n_alloc_oob == 0, the
    // workload didn't exercise the OOM signal — that means the
    // test isn't actually a feasibility proxy.
    if (n_alloc_oob == 0) {
        std::printf(
            "FAIL: workload never exhausted the pool — n_alloc_oob=0. "
            "Test isn't exercising OOMBehaviourIsFirstClass.\n");
        return 1;
    }
    // peak_in_flight should reach POOL_CAPACITY at some point.
    if (peak_in_flight != POOL_CAPACITY) {
        std::printf(
            "FAIL: peak_in_flight=%d never reached POOL_CAPACITY=%d. "
            "Workload didn't actually saturate the pool.\n",
            peak_in_flight, POOL_CAPACITY);
        return 1;
    }

    std::printf(
        "PASS: feasibility proxy verified — %d alloc success, "
        "%d alloc OOB, %d frees, peak in-flight = %d (= POOL_CAPACITY=%d).\n",
        n_alloc_success, n_alloc_oob, n_free_ops, peak_in_flight,
        POOL_CAPACITY);
    return 0;
}
