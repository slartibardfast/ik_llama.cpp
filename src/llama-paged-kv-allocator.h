// llama-paged-kv-allocator.h
//
// T5.1 — block allocator + block table for the paged KV layout.
//
// Per /home/llm/yarn-agentic/specs/kv-cache/paged_block_allocator.allium
// and /home/llm/yarn-agentic/specs/kv-cache/PagedKVAllocator.tla.
//
// At T5.1 this class lands as a standalone, dormant component of
// llama_kv_cache. It is NOT yet consumed by find_slot or the K/V
// WRITE/READ paths. find_slot integration is T5.2/T5.3/T5.4 (Bundle A).
//
// The class implements:
//   - BlockUniquelyOwned     — by construction (free_list ∩ tables == ∅)
//   - FreeListDisjoint       — by construction
//   - AllocLazy              — table size = ceil(written / block_size)
//   - DeterministicAtFixedSequence — LIFO std::deque free_list
//   - IdentityMappingAtSingleSeq   — single-seq writes pop blocks in
//                                     ascending order (block 0 first)
//
// Composes with:
//   - paged_write_path.allium  (T5.2 will consume block_table_of(seq))
//   - paged_read_path.allium   (T5.5 will consume block_table_of(seq))
//   - paged_kshift_defrag.allium (T5.7 will consume defrag()).
//
// Per the user's "complete sincerity" reframe: this allocator is
// designed to make high-ctx workloads ALLOCATABLE that contiguous
// cannot — AllocLazy bounds VRAM footprint by actual writes, NOT by
// per-stream slab × n_stream.

#ifndef LLAMA_PAGED_KV_ALLOCATOR_H
#define LLAMA_PAGED_KV_ALLOCATOR_H

#include <cstdint>
#include <deque>
#include <vector>

class llama_paged_kv_allocator {
public:
    // Block size in tokens. Locked per PHASE_NSTREAM_KV_PERF.md
    // §"Mechanism" (working-set L1 fit, 576B per ILP group).
    // OpenQ-T5-A allows drop to 128 if kernel indirection > 5%.
    // Not configurable in T5.1 per feedback_no_workarounds.
    static constexpr int32_t BLOCK_SIZE_TOKENS = 64;

    // Sentinel returned from alloc_block when the pool is full.
    static constexpr int32_t OOB_SENTINEL = -1;

    llama_paged_kv_allocator() = default;
    ~llama_paged_kv_allocator() = default;

    // Initialise / reset to N blocks across M sequences. Idempotent —
    // re-init clears all tables and refills the free list.
    void init(int32_t n_blocks, int32_t n_seqs);

    // Allocate one block to seq. Returns the block_id popped from the
    // top of the LIFO free_list, or OOB_SENTINEL if the pool is empty.
    int32_t alloc_block(int32_t seq);

    // Release all blocks for seq back to the free_list in reverse-
    // allocation order (preserves LIFO discipline).
    // Resets seq's written_tokens count to 0.
    // Idempotent: free_seq on an empty seq is a no-op.
    void free_seq(int32_t seq);

    // Record that seq wrote n more tokens. Triggers lazy alloc of as
    // many blocks as needed to cover the new total. Returns false on
    // OOM (pool exhausted before all needed blocks were allocated);
    // in that case the seq's state is left unchanged from before the
    // call (transactional: the partial allocs are rolled back).
    bool write_tokens(int32_t seq, int32_t n_tokens);

    // Query: number of blocks currently owned by seq.
    int32_t n_blocks_owned_by(int32_t seq) const;

    // Query: the i-th physical block_id owned by seq (i in [0, n_blocks_owned_by(seq))).
    // Returns OOB_SENTINEL on out-of-range.
    int32_t block_id_at(int32_t seq, int32_t i) const;

    // Query: number of free blocks remaining in the pool.
    int32_t n_free() const;

    // Query: total blocks in the pool (the init value).
    int32_t total_blocks() const { return total_blocks_; }

    // Query: block size in tokens (constant per BLOCK_SIZE_TOKENS).
    int32_t block_size_tokens() const { return BLOCK_SIZE_TOKENS; }

    // Query: written tokens count for seq (per AllocLazy).
    int32_t written_tokens_of(int32_t seq) const;

    // Query: const reference to the block table for seq. Used by the
    // T5.2 WRITE-path populator and T5.5 READ-path build-context
    // (when those phases land).
    const std::vector<int32_t> & block_table(int32_t seq) const;

    // Query: number of sequences this allocator was initialised for.
    int32_t n_seqs() const { return (int32_t)tables_.size(); }

    // T5.7c — paged defrag. Compacts the block pool by moving blocks
    // from high IDs into lower-ID free slots, so allocated blocks
    // occupy a contiguous prefix [0..n_allocated). Returns the list
    // of {old_bid -> new_bid} moves the caller must apply physically
    // (block-sized byte copy in the K/V cache buffer).
    //
    // Updates internal state in lockstep with the returned moves:
    //   - pool_owner_  rewires moved blocks
    //   - tables_[s]   replaces old_bid entries with new_bid (preserving
    //                  per-seq logical order)
    //   - free_list_   rebuilt LIFO with the post-defrag free set
    //
    // Idempotent: re-running defrag on an already-compact pool returns
    // an empty move list and is a no-op on state.
    //
    // Binds:
    //   paged_block_allocator.allium::DefragMechanics
    //   paged_kshift_defrag.allium::DefragPreservesLogicalContents
    //     ::MoveBytesAreCopiedExactly (caller's responsibility)
    //     ::DefragMaintainsBlockUniquelyOwned (this method preserves)
    struct defrag_move { int32_t old_bid; int32_t new_bid; };
    std::vector<defrag_move> defrag();

private:
    int32_t total_blocks_ = 0;
    std::vector<int32_t> pool_owner_;          // size total_blocks_; -1 = FREE
    std::vector<std::vector<int32_t>> tables_; // [seq] -> ordered block_ids
    std::deque<int32_t> free_list_;            // LIFO stack (front = top)
    std::vector<int32_t> written_tokens_;
};

#endif  // LLAMA_PAGED_KV_ALLOCATOR_H
