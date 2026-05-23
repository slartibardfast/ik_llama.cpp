// test-kv-block-allocator.cpp
//
// T5.1 binding test for the paged KV block allocator.
// Drives the production llama_paged_kv_allocator class
// (src/llama-paged-kv-allocator.{h,cpp}) and asserts the contracts
// from /home/llm/yarn-agentic/specs/kv-cache/paged_block_allocator.allium:
//
//   BlockUniquelyOwned     — no block_id is in two seqs' tables.
//   FreeListDisjoint       — free + tables partition pool ownership.
//   AllocLazy              — table size = ceil(written / block_size).
//   DeterministicAtFixedSequence — same op history => same outcome.
//   IdentityMappingAtSingleSeq   — single-seq allocs in ascending order.
//   AllocBlockBehavior     — alloc returns OOB on pool-full.
//   FreeSeqBehavior        — free returns to LIFO; subsequent allocs
//                            preserve LIFOFreeListOrder.
//
// Returns: 0 = PASS, 1 = FAIL.

#include "../../src/llama-paged-kv-allocator.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

static bool check_block_uniquely_owned(const llama_paged_kv_allocator & a) {
    for (int32_t s1 = 0; s1 < a.n_seqs(); ++s1) {
        for (int32_t s2 = s1 + 1; s2 < a.n_seqs(); ++s2) {
            const int32_t n1 = a.n_blocks_owned_by(s1);
            const int32_t n2 = a.n_blocks_owned_by(s2);
            for (int32_t i = 0; i < n1; ++i) {
                for (int32_t j = 0; j < n2; ++j) {
                    if (a.block_id_at(s1, i) == a.block_id_at(s2, j)) {
                        printf("FAIL: BlockUniquelyOwned: block %d in seqs %d and %d\n",
                               a.block_id_at(s1, i), s1, s2);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

static bool check_alloc_lazy(const llama_paged_kv_allocator & a) {
    const int32_t blk = a.block_size_tokens();
    for (int32_t s = 0; s < a.n_seqs(); ++s) {
        const int32_t w = a.written_tokens_of(s);
        const int32_t needed = (w == 0) ? 0 : ((w - 1) / blk) + 1;
        const int32_t have   = a.n_blocks_owned_by(s);
        if (have != needed) {
            printf("FAIL: AllocLazy: seq %d has %d blocks for %d writes (needed %d)\n",
                   s, have, w, needed);
            return false;
        }
    }
    return true;
}

static bool t_basic_alloc_free() {
    llama_paged_kv_allocator a;
    a.init(8, 3);
    if (!a.write_tokens(0, 128)) return false;  // 2 blocks
    if (!a.write_tokens(1, 64))  return false;  // 1 block
    if (!a.write_tokens(2, 192)) return false;  // 3 blocks
    if (!check_block_uniquely_owned(a)) return false;
    if (!check_alloc_lazy(a)) return false;
    if (a.n_blocks_owned_by(0) != 2) return false;
    if (a.n_blocks_owned_by(1) != 1) return false;
    if (a.n_blocks_owned_by(2) != 3) return false;
    if (a.n_free() != 2) return false;

    a.free_seq(1);
    if (!check_block_uniquely_owned(a)) return false;
    if (!check_alloc_lazy(a)) return false;
    if (a.n_blocks_owned_by(1) != 0) return false;
    if (a.n_free() != 3) return false;
    return true;
}

static bool t_identity_single_seq() {
    // Sole writer => block_table[0] = [0, 1, 2, ...] (ascending).
    llama_paged_kv_allocator a;
    a.init(8, 3);
    a.write_tokens(0, 192);  // 3 blocks
    for (int32_t i = 0; i < a.n_blocks_owned_by(0); ++i) {
        if (a.block_id_at(0, i) != i) {
            printf("FAIL: IdentityMappingAtSingleSeq: block_table[0][%d] = %d, expected %d\n",
                   i, a.block_id_at(0, i), i);
            return false;
        }
    }
    return true;
}

static bool t_oom_signal() {
    llama_paged_kv_allocator a;
    a.init(4, 2);
    if (!a.write_tokens(0, 128)) return false;  // 2 blocks
    if (!a.write_tokens(1, 128)) return false;  // 2 blocks; pool full
    if (a.write_tokens(0, 64)) {
        printf("FAIL: OOM signal not raised when pool exhausted\n");
        return false;
    }
    // Transactional rollback: state unchanged from before the failed write.
    if (a.n_blocks_owned_by(0) != 2) {
        printf("FAIL: transactional rollback violated: seq 0 has %d blocks (expected 2)\n",
               a.n_blocks_owned_by(0));
        return false;
    }
    if (a.written_tokens_of(0) != 128) {
        printf("FAIL: rollback: seq 0 written_tokens=%d (expected 128)\n",
               a.written_tokens_of(0));
        return false;
    }
    if (!check_block_uniquely_owned(a)) return false;
    return true;
}

static bool t_alloc_lazy_no_writes() {
    llama_paged_kv_allocator a;
    a.init(8, 3);
    for (int32_t s = 0; s < 3; ++s) {
        if (a.n_blocks_owned_by(s) != 0) {
            printf("FAIL: AllocLazy::NoPreallocation: seq %d has %d blocks with 0 writes\n",
                   s, a.n_blocks_owned_by(s));
            return false;
        }
    }
    if (a.n_free() != 8) return false;
    return true;
}

// DeterministicAtFixedSequence — same op history => same outcome.
static bool t_determinism_across_runs() {
    // Op trace: positive = write_tokens(seq, 64), negative = free_seq(-x-1).
    const std::vector<int> trace = {0, 1, 0, -1, 2, 0, -2, -3, 1, 0, 2};
    std::vector<std::vector<int32_t>> snapshots(3);
    for (int run = 0; run < 3; ++run) {
        llama_paged_kv_allocator a;
        a.init(8, 3);
        for (int op : trace) {
            if (op >= 0) {
                a.write_tokens(op, 64);  // 1 block each
            } else {
                a.free_seq(-op - 1);
            }
        }
        // Snapshot: flatten all tables into one sequence in seq order.
        for (int32_t s = 0; s < a.n_seqs(); ++s) {
            snapshots[run].push_back(-1000 - s);  // delimiter
            for (int32_t i = 0; i < a.n_blocks_owned_by(s); ++i) {
                snapshots[run].push_back(a.block_id_at(s, i));
            }
        }
    }
    if (snapshots[0] != snapshots[1] || snapshots[1] != snapshots[2]) {
        printf("FAIL: DeterministicAtFixedSequence — outputs diverge across runs\n");
        return false;
    }
    return true;
}

// LIFOFreeListOrder — most-recently-freed block is the next allocated.
static bool t_lifo_free_list_order() {
    llama_paged_kv_allocator a;
    a.init(8, 2);
    a.write_tokens(0, 64);  // alloc block 0
    a.write_tokens(0, 64);  // alloc block 1
    a.write_tokens(0, 64);  // alloc block 2
    // Free seq 0 => free_list gets blocks pushed back in reverse:
    // pushed in order 2, 1, 0. Free list front is 0 again.
    a.free_seq(0);
    a.write_tokens(1, 64);  // should alloc block 0 (LIFO top)
    if (a.block_id_at(1, 0) != 0) {
        printf("FAIL: LIFOFreeListOrder: post-free re-alloc returned %d, expected 0\n",
               a.block_id_at(1, 0));
        return false;
    }
    return true;
}

static bool t_pool_full_then_free_then_realloc() {
    llama_paged_kv_allocator a;
    a.init(4, 2);
    a.write_tokens(0, 256);  // 4 blocks; pool full
    if (a.n_free() != 0) return false;
    if (a.write_tokens(1, 64)) {
        printf("FAIL: expected OOM on pool-full alloc\n");
        return false;
    }
    a.free_seq(0);
    if (a.n_free() != 4) return false;
    if (!a.write_tokens(1, 64)) {
        printf("FAIL: alloc after free should succeed\n");
        return false;
    }
    return true;
}

int main(int /*argc*/, char ** /*argv*/) {
    bool ok = true;
    ok &= t_basic_alloc_free();
    ok &= t_identity_single_seq();
    ok &= t_oom_signal();
    ok &= t_alloc_lazy_no_writes();
    ok &= t_determinism_across_runs();
    ok &= t_lifo_free_list_order();
    ok &= t_pool_full_then_free_then_realloc();
    if (!ok) {
        printf("FAIL: one or more allocator invariant checks failed\n");
        return 1;
    }
    printf("PASS: all paged KV allocator invariants hold\n");
    return 0;
}
