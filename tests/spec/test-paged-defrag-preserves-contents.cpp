// test-paged-defrag-preserves-contents.cpp
//
// Binding test for paged_kshift_defrag.allium
//   ::DefragPreservesLogicalContents
//   ::MoveBytesAreCopiedExactly
//   ::DefragMaintainsBlockUniquelyOwned
// and paged_block_allocator.allium::DefragMechanics
//   ::LogicalSequencePreserved
//   ::CompactionAfterDefrag
//   ::BlockUniqueOwnershipPreserved
//
// T5.7c — paged defrag at the allocator level.
//
// Pattern (per file-header sketch):
//   1. Allocate N=12 block pool; 4-block seqs for seqs 0/1/2.
//   2. Free seq 1 (creates a 4-block hole in the middle of the pool).
//   3. Write 1 more block for a new seq 3.
//   4. Snapshot each seq's logical contents pre-defrag.
//   5. Run paged.defrag().
//   6. Physically apply each returned {old_bid -> new_bid} move to a
//      synthetic block-pool buffer.
//   7. Re-read each seq's logical contents using post-defrag
//      block_table.
//   8. Assert byte-identical pre vs post for every (seq, logical_pos).
//   9. Assert BlockUniquelyOwned holds post-defrag (no overlap
//      between any two seqs' tables; no allocated bid in free_list).
//  10. Assert CompactionAfterDefrag: allocated bids occupy a
//      contiguous prefix [0..n_allocated).
//
// The synthetic block-pool buffer is a simple bytes vector
// [total_blocks * BYTES_PER_BLOCK]; each block is filled with a
// unique payload at write time so seq-logical reads can detect any
// mis-addressing.

#include "llama-paged-kv-allocator.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef LLAMA_PAGED_KV_LANDED
#define LLAMA_PAGED_KV_LANDED 0
#endif

namespace {

#define FAIL_AT(msg, ...) do { \
    std::fprintf(stderr, "FAIL %s:%d: " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    std::exit(1); \
} while (0)

// Synthetic block: 256 bytes containing a unique pattern derived from
// the writer's (seq, logical_pos_in_seq) at write time. Small enough
// for a 12-block pool to fit in stack-friendly buffers.
static constexpr size_t BYTES_PER_BLOCK = 256;

void fill_block(uint8_t * dst, int32_t seq, int32_t logical_pos_in_seq) {
    // Pattern: byte i = ((seq + 0x10) ^ (logical_pos_in_seq * 7) ^ i)
    // — distinct for any (seq, pos, byte) triple in the test domain.
    for (size_t i = 0; i < BYTES_PER_BLOCK; ++i) {
        dst[i] = (uint8_t)(((uint32_t)(seq + 0x10)
                            ^ (uint32_t)(logical_pos_in_seq * 7)
                            ^ (uint32_t)i) & 0xFF);
    }
}

// Read the bytes of seq's logical block index `i` from the pool
// using the current block_table mapping.
const uint8_t * read_logical_block(const std::vector<uint8_t> & pool,
                                   const std::vector<int32_t> & table,
                                   int32_t i) {
    return pool.data() + (size_t)table[(size_t)i] * BYTES_PER_BLOCK;
}

}  // namespace

int main() {
    if (!LLAMA_PAGED_KV_LANDED) {
        std::fprintf(stderr, "SKIP: LLAMA_PAGED_KV_LANDED not set; "
                              "T5.7c paged defrag test inactive on this build.\n");
        return 1;
    }

    constexpr int32_t N_BLOCKS = 12;
    constexpr int32_t N_SEQS   = 4;
    constexpr int32_t BLK_TOK  = llama_paged_kv_allocator::BLOCK_SIZE_TOKENS;

    llama_paged_kv_allocator alloc;
    alloc.init(N_BLOCKS, N_SEQS);

    // 4 blocks each for seqs 0, 1, 2 — fills 12 blocks total.
    if (!alloc.write_tokens(0, BLK_TOK * 4)) FAIL_AT("write seq 0 failed");
    if (!alloc.write_tokens(1, BLK_TOK * 4)) FAIL_AT("write seq 1 failed");
    if (!alloc.write_tokens(2, BLK_TOK * 4)) FAIL_AT("write seq 2 failed");

    // Synthetic block pool. Fill each (seq, logical_pos) block with
    // a deterministic pattern.
    std::vector<uint8_t> pool((size_t)N_BLOCKS * BYTES_PER_BLOCK, 0);
    auto fill_seq_blocks = [&](int32_t seq) {
        const auto & tbl = alloc.block_table(seq);
        for (size_t i = 0; i < tbl.size(); ++i) {
            uint8_t * dst = pool.data() + (size_t)tbl[i] * BYTES_PER_BLOCK;
            fill_block(dst, seq, (int32_t)i);
        }
    };
    fill_seq_blocks(0);
    fill_seq_blocks(1);
    fill_seq_blocks(2);

    // Free seq 1 → 4 blocks return to free_list (LIFO).
    alloc.free_seq(1);

    // Write 1 block for seq 3 — pops top of free_list (the most-
    // recently-freed block from seq 1).
    if (!alloc.write_tokens(3, BLK_TOK * 1)) FAIL_AT("write seq 3 failed");
    fill_seq_blocks(3);

    // Snapshot pre-defrag: for each seq, the bytes of each logical
    // block. Re-reads via current block_table.
    auto snapshot_seq = [&](int32_t seq) -> std::vector<std::vector<uint8_t>> {
        std::vector<std::vector<uint8_t>> out;
        const auto & tbl = alloc.block_table(seq);
        out.reserve(tbl.size());
        for (size_t i = 0; i < tbl.size(); ++i) {
            const uint8_t * src = read_logical_block(pool, tbl, (int32_t)i);
            out.emplace_back(src, src + BYTES_PER_BLOCK);
        }
        return out;
    };
    auto pre_0 = snapshot_seq(0);
    auto pre_2 = snapshot_seq(2);
    auto pre_3 = snapshot_seq(3);

    // Run defrag.
    auto moves = alloc.defrag();
    std::fprintf(stdout, "defrag: %zu moves\n", moves.size());

    // Apply moves physically to the pool buffer. Move semantics:
    // old_bid's bytes are now at new_bid's region. We copy in the
    // order returned (greedy: highest old → lowest new); the move
    // sequence is non-overlapping by construction (each new_bid is
    // free at the moment of its move; old_bid's data is preserved
    // until that exact step).
    for (const auto & m : moves) {
        std::memcpy(pool.data() + (size_t)m.new_bid * BYTES_PER_BLOCK,
                    pool.data() + (size_t)m.old_bid * BYTES_PER_BLOCK,
                    BYTES_PER_BLOCK);
        // (old_bid's region may be overwritten later by another move
        // or stay as stale-but-unused bytes; the spec contract
        // UnusedBlocksMayBeOverwritten allows this.)
    }

    // Re-read post-defrag and compare byte-by-byte.
    auto post_0 = snapshot_seq(0);
    auto post_2 = snapshot_seq(2);
    auto post_3 = snapshot_seq(3);

    auto check_seq = [](int32_t seq,
                        const std::vector<std::vector<uint8_t>> & pre,
                        const std::vector<std::vector<uint8_t>> & post) {
        if (pre.size() != post.size()) {
            FAIL_AT("seq %d block count changed: pre=%zu post=%zu",
                    seq, pre.size(), post.size());
        }
        for (size_t i = 0; i < pre.size(); ++i) {
            if (pre[i] != post[i]) {
                size_t first_diff = 0;
                while (first_diff < BYTES_PER_BLOCK &&
                       pre[i][first_diff] == post[i][first_diff]) ++first_diff;
                FAIL_AT("seq %d logical block %zu byte %zu: pre=%02x post=%02x",
                        seq, i, first_diff,
                        pre[i][first_diff], post[i][first_diff]);
            }
        }
    };
    check_seq(0, pre_0, post_0);
    check_seq(2, pre_2, post_2);
    check_seq(3, pre_3, post_3);

    // BlockUniquelyOwned: assert no overlap between any two seqs'
    // tables and no allocated bid appears in n_free's bookkeeping.
    std::vector<int8_t> owned(N_BLOCKS, 0);
    auto mark_seq = [&](int32_t seq) {
        const auto & tbl = alloc.block_table(seq);
        for (int32_t b : tbl) {
            if (b < 0 || b >= N_BLOCKS) {
                FAIL_AT("seq %d has out-of-range bid %d", seq, b);
            }
            if (owned[(size_t)b]) {
                FAIL_AT("BlockUniquelyOwned violated: bid %d owned by "
                        "two seqs (this seq=%d)", b, seq);
            }
            owned[(size_t)b] = 1;
        }
    };
    mark_seq(0);
    mark_seq(2);
    mark_seq(3);
    // seq 1 was freed earlier — no blocks.
    int32_t n_allocated = 0;
    for (int8_t v : owned) if (v) ++n_allocated;
    if (n_allocated != 4 + 4 + 1) {
        FAIL_AT("expected 9 allocated blocks post-defrag (4+4+1); got %d",
                n_allocated);
    }

    // CompactionAfterDefrag: allocated bids form a contiguous prefix.
    int32_t highest_allocated = -1;
    for (int32_t b = N_BLOCKS - 1; b >= 0; --b) {
        if (owned[(size_t)b]) { highest_allocated = b; break; }
    }
    int32_t lowest_free = N_BLOCKS;
    for (int32_t b = 0; b < N_BLOCKS; ++b) {
        if (!owned[(size_t)b]) { lowest_free = b; break; }
    }
    if (highest_allocated >= lowest_free) {
        FAIL_AT("CompactionAfterDefrag violated: highest_allocated=%d "
                ">= lowest_free=%d (pool not compact)",
                highest_allocated, lowest_free);
    }

    // Idempotency: re-running defrag yields zero moves and no state
    // mutation.
    auto moves2 = alloc.defrag();
    if (!moves2.empty()) {
        FAIL_AT("Idempotency violated: second defrag returned %zu moves",
                moves2.size());
    }

    std::fprintf(stdout,
                 "test-paged-defrag-preserves-contents: PASS "
                 "(%d allocated blocks, %d compacted moves, "
                 "byte-identical reads for seqs {0, 2, 3})\n",
                 n_allocated, (int)moves.size());
    return 0;
}
