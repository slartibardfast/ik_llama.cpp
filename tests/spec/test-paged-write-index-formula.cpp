// test-paged-write-index-formula.cpp
//
// Property test for paged WRITE index formula contracts, derived from
// /home/llm/yarn-agentic/specs/kv-cache/paged_write_path.allium.
//
// Binds:
//   WriteIndexInBounds — every inp_kv_idxs entry < pool_size_rows;
//                        every targeted block is allocated.
//   WriteIndexUnique   — no two batch tokens scatter to the same
//                        pool row.
//   PagedKVWriteEquivToLegacyAtIdentity (formula collapse part) —
//     at single-seq identity mapping, inp_kv_idxs collapses to
//     the legacy flat-index pattern.
//
// STUB property test on the formula:
//   inp_kv_idxs[t * n_head_kv + h]
//     = block_table[seq(t)][pos(t) / block_size] * block_size
//       * n_head_kv + (pos(t) % block_size) * n_head_kv + h
//
// Transitions T5.2-binding when src/llama-paged-write-path.cpp lands
// and the production index populator is wired in.

#include <cassert>
#include <cstdio>
#include <vector>

static constexpr int BLOCK_SIZE = 64;
static constexpr int N_HEAD_KV = 8;

struct BatchTok {
    int seq;
    int pos_in_seq;
};

// Reference formula — to be replaced by production populator in T5.2.
static int paged_write_index(
    const std::vector<std::vector<int>>& block_table,
    const BatchTok& t, int h) {
    int blk_idx = t.pos_in_seq / BLOCK_SIZE;
    int tok_in_blk = t.pos_in_seq % BLOCK_SIZE;
    int block_id = block_table[t.seq][blk_idx];
    return block_id * BLOCK_SIZE * N_HEAD_KV + tok_in_blk * N_HEAD_KV + h;
}

static bool t_index_in_bounds() {
    std::vector<std::vector<int>> bt = {{0, 1}, {2, 3}};  // seq 0 owns blocks [0,1]; seq 1 owns [2,3]
    int pool_size_rows = 4 * BLOCK_SIZE * N_HEAD_KV;  // 4 blocks * 64 tok * 8 heads
    std::vector<BatchTok> batch = {{0, 0}, {0, 63}, {0, 64}, {0, 127}, {1, 0}, {1, 100}};
    for (const auto& t : batch) {
        for (int h = 0; h < N_HEAD_KV; ++h) {
            int idx = paged_write_index(bt, t, h);
            if (idx < 0 || idx >= pool_size_rows) {
                printf("FAIL: WriteIndexInBounds: idx=%d pool_size=%d for tok seq=%d pos=%d h=%d\n",
                       idx, pool_size_rows, t.seq, t.pos_in_seq, h);
                return false;
            }
        }
    }
    return true;
}

static bool t_index_unique() {
    std::vector<std::vector<int>> bt = {{0, 1}, {2, 3}};
    std::vector<BatchTok> batch = {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}};
    std::vector<int> seen;
    for (const auto& t : batch) {
        for (int h = 0; h < N_HEAD_KV; ++h) {
            int idx = paged_write_index(bt, t, h);
            for (int s : seen) {
                if (s == idx) {
                    printf("FAIL: WriteIndexUnique: duplicate idx=%d for tok seq=%d pos=%d h=%d\n",
                           idx, t.seq, t.pos_in_seq, h);
                    return false;
                }
            }
            seen.push_back(idx);
        }
    }
    return true;
}

static bool t_collapses_at_identity_single_seq() {
    // Single seq, identity block_table = [0, 1, 2, ...].
    std::vector<std::vector<int>> bt(1);
    for (int i = 0; i < 4; ++i) bt[0].push_back(i);
    for (int pos = 0; pos < 256; ++pos) {
        for (int h = 0; h < N_HEAD_KV; ++h) {
            BatchTok t{0, pos};
            int paged = paged_write_index(bt, t, h);
            // Legacy flat-index: pos * n_head_kv + h (no per-stream offset)
            int legacy = pos * N_HEAD_KV + h;
            if (paged != legacy) {
                printf("FAIL: collapse at identity broken: pos=%d h=%d paged=%d legacy=%d\n",
                       pos, h, paged, legacy);
                return false;
            }
        }
    }
    return true;
}

#ifndef LLAMA_PAGED_KV_LANDED
#define LLAMA_PAGED_KV_LANDED 0
#endif

int main() {
    bool ok = true;
    ok &= t_index_in_bounds();
    ok &= t_index_unique();
    ok &= t_collapses_at_identity_single_seq();
    if (!ok) {
        printf("FAIL: formula property checks failed\n");
        return 1;
    }
    if (!LLAMA_PAGED_KV_LANDED) {
        printf("FAIL: T5.2 paged WRITE populator not yet landed.\n");
        printf("      Formula property checks pass on the reference impl;\n");
        printf("      production populator not yet wired.\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}
