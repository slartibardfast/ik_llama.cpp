// test-paged-allocator-determinism.cpp
//
// NPC-binding determinism test for the paged KV block allocator.
// Runs the same alloc/free trace 3x in a fresh allocator instance;
// asserts identical block_id sequences across the runs.
//
// Catches non-deterministic allocator state (hash-map iteration
// order, etc.) that would propagate to GP5.NPC failure under
// verify-production-determinism.sh.
//
// Binds:
//   paged_block_allocator.allium::DeterministicAtFixedSequence
//     ::SameHistorySameOutcome
//     ::LIFOFreeListOrder
//   PagedKVAllocator.tla::AllocLazy + invariant chain
//
// STUB property test using a reference impl matching the production
// allocator's expected algorithm. The reference impl is the same as
// in test-kv-block-allocator.cpp (LIFO std::deque-backed free list).
//
// Transitions to T5.1-binding when the production
// llama_paged_kv_alloc_block API is wired in: this test then drives
// the production allocator instead of the reference.

#include <cassert>
#include <cstdio>
#include <vector>
#include <deque>

struct RefAllocator {
    int n_blocks;
    std::vector<int> block_pool_owner;
    std::vector<std::vector<int>> block_tables;
    std::deque<int> free_list;
    RefAllocator(int n_blocks_, int n_seqs) : n_blocks(n_blocks_) {
        block_pool_owner.assign(n_blocks_, -1);
        block_tables.assign(n_seqs, {});
        for (int b = n_blocks_ - 1; b >= 0; --b) free_list.push_front(b);
    }
    int alloc_block(int seq) {
        if (free_list.empty()) return -1;
        int b = free_list.front();
        free_list.pop_front();
        block_pool_owner[b] = seq;
        block_tables[seq].push_back(b);
        return b;
    }
    void free_seq(int seq) {
        auto& tbl = block_tables[seq];
        for (auto it = tbl.rbegin(); it != tbl.rend(); ++it) {
            free_list.push_front(*it);
            block_pool_owner[*it] = -1;
        }
        tbl.clear();
    }
};

// Op trace: positive = alloc(seq), negative = free(-seq - 1) (so -1 means free seq 0).
static std::vector<int> RunTrace(const std::vector<int>& ops, int n_blocks, int n_seqs) {
    RefAllocator a(n_blocks, n_seqs);
    std::vector<int> results;
    for (int op : ops) {
        if (op >= 0) {
            results.push_back(a.alloc_block(op));
        } else {
            a.free_seq(-op - 1);
            results.push_back(-1);  // Free returns void; record a marker.
        }
    }
    return results;
}

int main() {
    // Trace: alloc seq0, alloc seq1, alloc seq0, free seq0, alloc seq2, alloc seq0
    std::vector<int> trace = {0, 1, 0, -1, 2, 0};
    std::vector<int> r1 = RunTrace(trace, 8, 3);
    std::vector<int> r2 = RunTrace(trace, 8, 3);
    std::vector<int> r3 = RunTrace(trace, 8, 3);
    if (r1 != r2 || r2 != r3) {
        printf("FAIL: allocator determinism violated.\n");
        for (size_t i = 0; i < r1.size(); ++i) {
            printf("  step %zu: r1=%d r2=%d r3=%d\n", i, r1[i], r2[i], r3[i]);
        }
        return 1;
    }

    // Pool-exhaustion trace — verify OOM signal is deterministic too.
    std::vector<int> oom_trace;
    for (int i = 0; i < 10; ++i) oom_trace.push_back(0);  // 10 allocs to seq 0 from pool of 8
    std::vector<int> o1 = RunTrace(oom_trace, 8, 1);
    std::vector<int> o2 = RunTrace(oom_trace, 8, 1);
    std::vector<int> o3 = RunTrace(oom_trace, 8, 1);
    if (o1 != o2 || o2 != o3) {
        printf("FAIL: OOM signal not deterministic.\n");
        return 1;
    }
    // First 8 should be 0..7 (LIFO from initial free_list [7,6,...,0] - wait,
    // we push front from n_blocks-1 down to 0, so pop front gives 0 first).
    for (int i = 0; i < 8; ++i) {
        if (o1[i] != i) {
            printf("FAIL: LIFOFreeListOrder violated: o1[%d]=%d expected %d\n", i, o1[i], i);
            return 1;
        }
    }
    // Allocs 9, 10 should return -1 (OOM sentinel).
    if (o1[8] != -1 || o1[9] != -1) {
        printf("FAIL: OOM sentinel not returned at exhaustion.\n");
        return 1;
    }

#ifndef LLAMA_PAGED_KV_LANDED
#define LLAMA_PAGED_KV_LANDED 0
#endif
    if (!LLAMA_PAGED_KV_LANDED) {
        printf("FAIL: T5.1 paged allocator not yet wired in.\n");
        printf("      Reference impl determinism + LIFO + OOM checks pass;\n");
        printf("      production allocator API not yet bound to this test.\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}
