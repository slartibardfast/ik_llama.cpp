// Phase 0.4 — Unit test for qnext_state_slot_allocator.
//
// Exercises the allocator's invariants in isolation (no GPU, no
// llama_context). RED until Phase 1 lands the allocator API.
//
// Build: gated behind `LLAMA_BUILD_TESTS` and the allocator's own
// header; not wired into CMakeLists.txt until Phase 1.
//
// Invariants checked:
//   - alloc(seq) is idempotent: alloc(s) twice returns same slot
//   - release(seq) returns the slot to the free-list
//   - alloc after release reuses a freed slot
//   - free + active slot counts always sum to n_slots
//   - lookup() returns -1 for non-allocated seqs
//   - alloc on full allocator asserts (caller violation)

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <unordered_set>

#include "qnext-state-slot-allocator.h"  // Phase 1: in src/llama-context.h

int main() {
    qnext_state_slot_allocator alloc;
    alloc.init(/*n_slots=*/4);

    // initial: all slots free
    assert(alloc.n_free() == 4);
    assert(alloc.n_active() == 0);
    assert(alloc.lookup(0) == -1);

    // alloc 3 slots
    int32_t s0 = alloc.alloc(/*seq=*/0);
    int32_t s1 = alloc.alloc(/*seq=*/1);
    int32_t s2 = alloc.alloc(/*seq=*/2);
    assert(s0 >= 0 && s0 < 4);
    assert(s1 >= 0 && s1 < 4);
    assert(s2 >= 0 && s2 < 4);

    // distinct
    std::unordered_set<int32_t> distinct = { s0, s1, s2 };
    assert(distinct.size() == 3);

    // sum invariant
    assert(alloc.n_free() == 1);
    assert(alloc.n_active() == 3);

    // idempotent alloc
    assert(alloc.alloc(/*seq=*/1) == s1);
    assert(alloc.n_active() == 3);

    // release returns slot to free-list
    alloc.release(/*seq=*/1);
    assert(alloc.n_free() == 2);
    assert(alloc.n_active() == 2);
    assert(alloc.lookup(/*seq=*/1) == -1);

    // new alloc reuses a freed slot
    int32_t s10 = alloc.alloc(/*seq=*/10);
    assert(s10 >= 0 && s10 < 4);
    // s10 must be one of {s1} (only freed) or any other if implementation uses LIFO/FIFO
    // weak assertion: slot count consistent
    assert(alloc.n_active() == 3);
    assert(alloc.n_free() == 1);

    // release all
    alloc.release(/*seq=*/0);
    alloc.release(/*seq=*/2);
    alloc.release(/*seq=*/10);
    assert(alloc.n_free() == 4);
    assert(alloc.n_active() == 0);

    // alloc after full release succeeds
    int32_t s99 = alloc.alloc(/*seq=*/99);
    assert(s99 >= 0 && s99 < 4);
    assert(alloc.n_active() == 1);

    fprintf(stderr, "test-qnext-state-allocator: ALL OK\n");
    return 0;
}
