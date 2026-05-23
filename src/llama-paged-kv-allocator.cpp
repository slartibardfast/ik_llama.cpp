// llama-paged-kv-allocator.cpp
//
// T5.1 — implementation of the paged KV block allocator.
//
// See llama-paged-kv-allocator.h for the binding spec references.

#include "llama-paged-kv-allocator.h"
#include "llama-paged-kv-trace.h"

#include <algorithm>
#include <cassert>

static const std::vector<int32_t> empty_block_table;

void llama_paged_kv_allocator::init(int32_t n_blocks, int32_t n_seqs) {
    total_blocks_ = n_blocks;
    pool_owner_.assign((size_t)n_blocks, -1);
    tables_.assign((size_t)n_seqs, {});
    written_tokens_.assign((size_t)n_seqs, 0);
    free_list_.clear();
    // Build the LIFO free_list with block 0 on top so the FIRST alloc
    // returns block 0 (IdentityMappingAtSingleSeq).
    for (int32_t b = n_blocks - 1; b >= 0; --b) {
        free_list_.push_front(b);
    }
}

int32_t llama_paged_kv_allocator::alloc_block(int32_t seq) {
    assert(seq >= 0 && seq < (int32_t)tables_.size() &&
           "llama_paged_kv_allocator::alloc_block: seq out of range");
    if (free_list_.empty()) {
        return OOB_SENTINEL;
    }
    const int32_t b = free_list_.front();
    free_list_.pop_front();
    pool_owner_[(size_t)b] = seq;
    tables_[(size_t)seq].push_back(b);
    // T5.3 trace hook — no-op unless LLAMA_T5_TRACE=1.
    // tick=0: allocator does not own a decode-tick counter (the caller
    // does); event_id from the producer gives ordering.
    llama_paged_kv_trace_event(0, seq, b, LLAMA_PAGED_KV_TRACE_ALLOC, -1);
    return b;
}

void llama_paged_kv_allocator::free_seq(int32_t seq) {
    assert(seq >= 0 && seq < (int32_t)tables_.size() &&
           "llama_paged_kv_allocator::free_seq: seq out of range");
    auto & tbl = tables_[(size_t)seq];
    // Push back to free_list in REVERSE-allocation order so the next
    // alloc gets the most-recently-freed block (LIFO preservation per
    // DeterministicAtFixedSequence::LIFOFreeListOrder).
    for (auto it = tbl.rbegin(); it != tbl.rend(); ++it) {
        free_list_.push_front(*it);
        pool_owner_[(size_t)*it] = -1;
        // T5.3 trace hook.
        llama_paged_kv_trace_event(0, seq, *it, LLAMA_PAGED_KV_TRACE_FREE, -1);
    }
    tbl.clear();
    written_tokens_[(size_t)seq] = 0;
}

bool llama_paged_kv_allocator::write_tokens(int32_t seq, int32_t n_tokens) {
    assert(seq >= 0 && seq < (int32_t)tables_.size() &&
           "llama_paged_kv_allocator::write_tokens: seq out of range");
    if (n_tokens <= 0) {
        return true;
    }
    const int32_t new_total = written_tokens_[(size_t)seq] + n_tokens;
    const int32_t needed = (new_total == 0)
        ? 0
        : ((new_total - 1) / BLOCK_SIZE_TOKENS) + 1;
    const int32_t have = (int32_t)tables_[(size_t)seq].size();
    const int32_t deficit = needed - have;
    if (deficit <= 0) {
        written_tokens_[(size_t)seq] = new_total;
        return true;
    }
    if ((int32_t)free_list_.size() < deficit) {
        // Pool exhausted. Roll back: no partial allocs. The seq's
        // tables_/written_tokens_ are unchanged.
        return false;
    }
    // All deficit allocs will succeed. Commit.
    for (int32_t i = 0; i < deficit; ++i) {
        const int32_t b = alloc_block(seq);
        assert(b != OOB_SENTINEL && "free_list size pre-check failed");
        (void)b;
    }
    written_tokens_[(size_t)seq] = new_total;
    return true;
}

int32_t llama_paged_kv_allocator::n_blocks_owned_by(int32_t seq) const {
    if (seq < 0 || seq >= (int32_t)tables_.size()) {
        return 0;
    }
    return (int32_t)tables_[(size_t)seq].size();
}

int32_t llama_paged_kv_allocator::block_id_at(int32_t seq, int32_t i) const {
    if (seq < 0 || seq >= (int32_t)tables_.size()) {
        return OOB_SENTINEL;
    }
    const auto & tbl = tables_[(size_t)seq];
    if (i < 0 || i >= (int32_t)tbl.size()) {
        return OOB_SENTINEL;
    }
    return tbl[(size_t)i];
}

int32_t llama_paged_kv_allocator::n_free() const {
    return (int32_t)free_list_.size();
}

int32_t llama_paged_kv_allocator::written_tokens_of(int32_t seq) const {
    if (seq < 0 || seq >= (int32_t)written_tokens_.size()) {
        return 0;
    }
    return written_tokens_[(size_t)seq];
}

const std::vector<int32_t> & llama_paged_kv_allocator::block_table(int32_t seq) const {
    if (seq < 0 || seq >= (int32_t)tables_.size()) {
        return empty_block_table;
    }
    return tables_[(size_t)seq];
}

std::vector<llama_paged_kv_allocator::defrag_move> llama_paged_kv_allocator::defrag() {
    // Pool compaction: move allocated blocks at high IDs into low-ID
    // free slots so allocated blocks occupy a contiguous prefix
    // [0..n_allocated). Greedy pairing of highest-allocated with
    // lowest-free; the post-defrag pool has zero gaps below the
    // highest allocated block.
    //
    // Per the spec contract DefragPreservesLogicalContents the caller
    // physically applies each returned move via a block-sized byte
    // copy in the K/V cache buffer. This method only mutates the
    // allocator's bookkeeping; it does not touch device memory.
    std::vector<defrag_move> moves;

    std::vector<int32_t> allocated_bids;   // sorted descending
    std::vector<int32_t> free_bids;        // sorted ascending
    allocated_bids.reserve((size_t)total_blocks_);
    free_bids.reserve((size_t)total_blocks_);
    for (int32_t b = 0; b < total_blocks_; ++b) {
        if (pool_owner_[(size_t)b] >= 0) {
            allocated_bids.push_back(b);
        } else {
            free_bids.push_back(b);
        }
    }
    std::reverse(allocated_bids.begin(), allocated_bids.end());

    size_t ai = 0, fi = 0;
    while (ai < allocated_bids.size() && fi < free_bids.size() &&
           allocated_bids[ai] > free_bids[fi]) {
        const int32_t old_bid = allocated_bids[ai];
        const int32_t new_bid = free_bids[fi];
        const int32_t owner   = pool_owner_[(size_t)old_bid];

        // Rewire ownership: free old_bid, claim new_bid for owner.
        pool_owner_[(size_t)new_bid] = owner;
        pool_owner_[(size_t)old_bid] = -1;

        // Update owner's table: replace old_bid with new_bid at its
        // logical position (single occurrence per seq by
        // BlockUniquelyOwned).
        auto & tbl = tables_[(size_t)owner];
        for (auto & b : tbl) {
            if (b == old_bid) { b = new_bid; break; }
        }

        moves.push_back({old_bid, new_bid});

        // Trace hook — fires only when LLAMA_T5_TRACE=1.
        llama_paged_kv_trace_event(0, owner, new_bid,
                                   LLAMA_PAGED_KV_TRACE_DEFRAG_MOVE, old_bid);

        ++ai; ++fi;
    }

    // Rebuild free_list from pool_owner_ in LIFO ascending-IDs-on-top
    // order (matches init's invariant: lowest free id pops first).
    free_list_.clear();
    for (int32_t b = total_blocks_ - 1; b >= 0; --b) {
        if (pool_owner_[(size_t)b] == -1) {
            free_list_.push_front(b);
        }
    }

    return moves;
}
