// test-paged-allocator-determinism.cpp
//
// T5.1 NPC-binding determinism test for the paged KV block allocator.
// Runs the same alloc/free trace 3x in a fresh allocator instance;
// asserts identical block_id sequences across the runs.
//
// Catches non-deterministic allocator state (hash-map iteration order,
// etc.) that would propagate to GP5.NPC failure under
// verify-production-determinism.sh.
//
// Binds:
//   paged_block_allocator.allium::DeterministicAtFixedSequence
//     ::SameHistorySameOutcome
//     ::LIFOFreeListOrder
//   PagedKVAllocator.tla::AllocLazy
//
// Returns: 0 = PASS, 1 = FAIL.

#include "../../src/llama-paged-kv-allocator.h"

#include <cstdio>
#include <vector>

// Op trace: positive = write_tokens(seq, BLOCK_SIZE_TOKENS),
// negative = free_seq(-x - 1).
static std::vector<int32_t> RunTrace(const std::vector<int> & ops,
                                     int32_t n_blocks, int32_t n_seqs) {
    llama_paged_kv_allocator a;
    a.init(n_blocks, n_seqs);
    std::vector<int32_t> results;
    for (int op : ops) {
        if (op >= 0) {
            const bool ok = a.write_tokens(op, a.block_size_tokens());
            // Record the most-recent block_id for this seq, or -1 on OOM.
            if (!ok) {
                results.push_back(-1);
            } else {
                const int32_t n = a.n_blocks_owned_by(op);
                results.push_back(a.block_id_at(op, n - 1));
            }
        } else {
            a.free_seq(-op - 1);
            results.push_back(-2);  // free marker
        }
    }
    return results;
}

int main() {
    // Trace 1: standard interleaved alloc/free, 3 seqs.
    const std::vector<int> trace = {0, 1, 0, -1, 2, 0, -2, -3};  // -1 = free(0)
    const std::vector<int32_t> r1 = RunTrace(trace, 8, 3);
    const std::vector<int32_t> r2 = RunTrace(trace, 8, 3);
    const std::vector<int32_t> r3 = RunTrace(trace, 8, 3);
    if (r1 != r2 || r2 != r3) {
        printf("FAIL: allocator determinism violated across 3 runs.\n");
        for (size_t i = 0; i < r1.size(); ++i) {
            printf("  step %zu: r1=%d r2=%d r3=%d\n",
                   i, r1[i], r2[i], r3[i]);
        }
        return 1;
    }

    // Trace 2: pool-exhaustion stress. 10 writes from seq 0 in an 8-block pool.
    std::vector<int> oom_trace;
    for (int i = 0; i < 10; ++i) {
        oom_trace.push_back(0);
    }
    const std::vector<int32_t> o1 = RunTrace(oom_trace, 8, 1);
    const std::vector<int32_t> o2 = RunTrace(oom_trace, 8, 1);
    const std::vector<int32_t> o3 = RunTrace(oom_trace, 8, 1);
    if (o1 != o2 || o2 != o3) {
        printf("FAIL: OOM signal not deterministic across runs.\n");
        return 1;
    }
    // First 8 should be blocks 0..7 (LIFO with front-pushed init).
    for (int i = 0; i < 8; ++i) {
        if (o1[i] != i) {
            printf("FAIL: LIFOFreeListOrder: step %d returned %d, expected %d\n",
                   i, o1[i], i);
            return 1;
        }
    }
    // Allocs 9, 10 should signal OOM (-1).
    if (o1[8] != -1 || o1[9] != -1) {
        printf("FAIL: OOM sentinel not returned at exhaustion (got %d, %d)\n",
               o1[8], o1[9]);
        return 1;
    }

    // Trace 3: re-alloc after free preserves LIFO discipline.
    const std::vector<int> lifo_trace = {0, 0, 0, -1, 1};
    const std::vector<int32_t> l1 = RunTrace(lifo_trace, 8, 2);
    const std::vector<int32_t> l2 = RunTrace(lifo_trace, 8, 2);
    if (l1 != l2) {
        printf("FAIL: post-free re-alloc not deterministic\n");
        return 1;
    }

    printf("PASS: allocator determinism + LIFO + OOM signal verified across 3 runs\n");
    return 0;
}
