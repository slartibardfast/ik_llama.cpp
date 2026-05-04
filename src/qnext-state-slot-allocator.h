#pragma once

// Maps llama_seq_id -> fixed slot index in the qwen3next linear-attn
// state buffer. The state buffer (lctx.kv_self.s_l[il]) is allocated
// with shape [state_dim, n_slots] at context init; this allocator
// hands out per-seq slot indices into that fixed pool.
//
// Why this exists:
//   The qwen3next linear-attn (Gated DeltaNet) path keeps a per-seq
//   recurrent state. Without an explicit slot-of-seq mapping, every
//   token would write to slot 0 (see src/llama.cpp:4355's old
//   data[j] = 0), causing concurrent slots to corrupt each other's
//   recurrent state. This allocator gives each active seq its own
//   slot index.
//
// Invariants:
//   - alloc(s) is idempotent: calling twice returns the same slot.
//   - release(s) returns the slot to the free list.
//   - n_active() + n_free() == n_slots always.
//   - alloc when free list is empty returns -1 (caller bug; should
//     not happen if n_seq_max sized n_slots correctly).
//
// Thread-safety:
//   Not internally synchronized. The caller (llama_context) serialises
//   access; ik_llama.cpp's update_slots() is single-threaded per
//   server tick.

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "llama.h"   // for llama_seq_id

struct qnext_state_slot_allocator {
    int32_t n_slots = 0;
    std::vector<int32_t> free_list;                       // LIFO of available slot indices
    std::unordered_map<llama_seq_id, int32_t> slot_of;    // seq_id -> slot

    void init(int32_t n_slots_) {
        n_slots = n_slots_;
        free_list.clear();
        free_list.reserve((size_t) n_slots_);
        for (int32_t i = n_slots_ - 1; i >= 0; --i) {
            free_list.push_back(i);
        }
        slot_of.clear();
    }

    // Idempotent: returns existing slot if seq is already mapped;
    // otherwise pops a free slot, maps seq -> slot, returns it.
    // Returns -1 if the free list is empty (n_seq_max was too small).
    int32_t alloc(llama_seq_id seq) {
        auto it = slot_of.find(seq);
        if (it != slot_of.end()) {
            return it->second;
        }
        if (free_list.empty()) {
            return -1;
        }
        int32_t slot = free_list.back();
        free_list.pop_back();
        slot_of[seq] = slot;
        return slot;
    }

    // Returns slot to the free list. No-op if seq isn't mapped.
    void release(llama_seq_id seq) {
        auto it = slot_of.find(seq);
        if (it == slot_of.end()) return;
        free_list.push_back(it->second);
        slot_of.erase(it);
    }

    // Returns the slot index for seq, or -1 if not allocated.
    int32_t lookup(llama_seq_id seq) const {
        auto it = slot_of.find(seq);
        return it == slot_of.end() ? -1 : it->second;
    }

    int32_t n_active() const { return (int32_t) slot_of.size(); }
    int32_t n_free()   const { return (int32_t) free_list.size(); }
};
