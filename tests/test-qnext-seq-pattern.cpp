// T1 — analyzer unit test for qnext_analyze_seq_pattern.
//
// RED before Phase A: the header doesn't exist yet, so this file
// doesn't compile. CMake glue is added in Phase A.
//
// GREEN after Phase A: every assertion passes.

#include "qnext-seq-pattern.h"
#include "llama.h"

#include <cassert>
#include <cstdio>
#include <vector>

namespace {

struct fake_batch {
    std::vector<int32_t>                  n_seq_id_storage;
    std::vector<std::vector<llama_seq_id>> seq_ids_storage;
    std::vector<llama_seq_id *>           seq_id_ptrs;
    llama_batch                            batch{};

    void build(const std::vector<llama_seq_id> & ids) {
        n_seq_id_storage.assign(ids.size(), 1);
        seq_ids_storage.clear();
        seq_id_ptrs.clear();
        for (auto sid : ids) {
            seq_ids_storage.push_back({sid});
        }
        for (auto & v : seq_ids_storage) {
            seq_id_ptrs.push_back(v.data());
        }
        batch = {};
        batch.n_tokens = (int32_t) ids.size();
        batch.n_seq_id = n_seq_id_storage.data();
        batch.seq_id   = seq_id_ptrs.data();
    }
};

void expect_single(const std::vector<llama_seq_id> & ids) {
    fake_batch fb; fb.build(ids);
    std::vector<qnext_seq_block> blocks;
    auto pat = qnext_analyze_seq_pattern(fb.batch, blocks);
    assert(pat == QNEXT_SEQ_SINGLE);
    (void) blocks;
}

void expect_contiguous(const std::vector<llama_seq_id> & ids,
                       const std::vector<qnext_seq_block> & expected) {
    fake_batch fb; fb.build(ids);
    std::vector<qnext_seq_block> blocks;
    auto pat = qnext_analyze_seq_pattern(fb.batch, blocks);
    assert(pat == QNEXT_SEQ_CONTIGUOUS_BLOCKS);
    assert(blocks.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(blocks[i].seq_id == expected[i].seq_id);
        assert(blocks[i].start  == expected[i].start);
        assert(blocks[i].len    == expected[i].len);
    }
}

void expect_interleaved(const std::vector<llama_seq_id> & ids) {
    fake_batch fb; fb.build(ids);
    std::vector<qnext_seq_block> blocks;
    auto pat = qnext_analyze_seq_pattern(fb.batch, blocks);
    assert(pat == QNEXT_SEQ_INTERLEAVED);
    (void) blocks;
}

} // namespace

int main() {
    // SINGLE
    expect_single({0});
    expect_single({0, 0, 0, 0});
    expect_single({3, 3, 3});

    // CONTIGUOUS_BLOCKS
    expect_contiguous({0, 0, 1, 1},        {{0,0,2}, {1,2,2}});
    expect_contiguous({0, 1, 1},           {{0,0,1}, {1,1,2}});
    expect_contiguous({0, 0, 0, 1, 2, 2, 2}, {{0,0,3}, {1,3,1}, {2,4,3}});
    expect_contiguous({2, 2, 0, 0},        {{2,0,2}, {0,2,2}});

    // INTERLEAVED — same seq reappears after a different one
    expect_interleaved({0, 1, 0, 1});
    expect_interleaved({0, 1, 0});
    expect_interleaved({0, 0, 1, 0});

    // Edge cases
    {
        fake_batch fb; fb.build({});  // n_tokens=0
        std::vector<qnext_seq_block> blocks;
        auto pat = qnext_analyze_seq_pattern(fb.batch, blocks);
        assert(pat == QNEXT_SEQ_SINGLE);
        (void) blocks;
    }

    printf("T1 GREEN: qnext_analyze_seq_pattern passed all cases\n");
    return 0;
}
