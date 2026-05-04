#pragma once

// Classifies a llama_batch's per-token seq_id pattern for the
// qwen3next linear-attn dispatcher. Three patterns are recognised:
//
//   QNEXT_SEQ_SINGLE             every token's seq_id[0] is the same
//   QNEXT_SEQ_CONTIGUOUS_BLOCKS  each unique seq_id forms exactly one
//                                contiguous run within the batch
//                                (e.g. [A,A,B,B] or [0,0,0,1,2,2,2])
//   QNEXT_SEQ_INTERLEAVED        otherwise (e.g. [A,B,A,B])
//
// SINGLE and CONTIGUOUS_BLOCKS are both compatible with the kernel's
// single-seq fast path — SINGLE as one whole-batch dispatch, blocks
// as one dispatch per block. INTERLEAVED requires the slow per-token
// fallback. See PHASE32-MTP-MULTISLOT-THEORY.md for the wider plan.
//
// The analyzer is host-side, allocation-free aside from blocks_out's
// internal vector growth. Caller may .reserve() blocks_out to the
// expected n_seqs to avoid the growth.

#include "llama.h"   // for llama_batch, llama_seq_id

#include <cstdint>
#include <vector>

enum qnext_seq_pattern_t {
    QNEXT_SEQ_SINGLE = 0,
    QNEXT_SEQ_CONTIGUOUS_BLOCKS = 1,
    QNEXT_SEQ_INTERLEAVED = 2,
};

struct qnext_seq_block {
    llama_seq_id seq_id;
    int32_t      start;
    int32_t      len;
};

inline qnext_seq_pattern_t qnext_analyze_seq_pattern(
        const llama_batch & batch,
        std::vector<qnext_seq_block> & blocks_out) {
    blocks_out.clear();
    const int32_t n_tokens = batch.n_tokens;
    if (n_tokens <= 0) {
        return QNEXT_SEQ_SINGLE;  // vacuous
    }

    auto sid_at = [&](int32_t i) -> llama_seq_id {
        if (batch.n_seq_id == nullptr || batch.seq_id == nullptr) return 0;
        if (batch.n_seq_id[i] <= 0 || batch.seq_id[i] == nullptr) return 0;
        return batch.seq_id[i][0];
    };

    // Pass 1: collect contiguous runs.
    blocks_out.reserve(8);
    llama_seq_id cur = sid_at(0);
    int32_t run_start = 0;
    for (int32_t i = 1; i < n_tokens; ++i) {
        const llama_seq_id sid = sid_at(i);
        if (sid != cur) {
            blocks_out.push_back({cur, run_start, i - run_start});
            cur = sid;
            run_start = i;
        }
    }
    blocks_out.push_back({cur, run_start, n_tokens - run_start});

    if (blocks_out.size() == 1) {
        return QNEXT_SEQ_SINGLE;
    }

    // Pass 2: check uniqueness — every seq_id must appear in only one run.
    // If a seq_id appears in two runs, the pattern is INTERLEAVED.
    // We expect the number of unique seqs to be small (<= n_seq_max),
    // so a linear scan is fine.
    for (size_t i = 0; i < blocks_out.size(); ++i) {
        for (size_t j = i + 1; j < blocks_out.size(); ++j) {
            if (blocks_out[i].seq_id == blocks_out[j].seq_id) {
                blocks_out.clear();
                return QNEXT_SEQ_INTERLEAVED;
            }
        }
    }
    return QNEXT_SEQ_CONTIGUOUS_BLOCKS;
}
