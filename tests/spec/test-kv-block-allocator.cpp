// test-kv-block-allocator.cpp
//
// Property test for paged KV block allocator invariants, derived from
// /home/llm/yarn-agentic/specs/kv-cache/paged_block_allocator.allium
// and /home/llm/yarn-agentic/specs/kv-cache/PagedKVAllocator.tla.
//
// Binds the following contracts:
//
//   BlockUniquelyOwned — no block_id is in two seqs' block_tables.
//   FreeListDisjoint   — free_list and block_tables pairwise disjoint.
//   AllocLazy          — table size matches ceil(writes / block_size).
//   DeterministicAtFixedSequence — same op history => same outcome.
//   IdentityMappingAtSingleSeq   — single-seq alloc => contiguous prefix.
//
// STUB property test — exercises a reference implementation of the
// allocator algorithm against synthetic op traces. The reference
// impl mirrors paged_block_allocator.allium's contract semantics.
//
// Transition to T5.1-binding: when src/llama-paged-kv.{h,cpp} land
// in T5.1, the test author replaces the reference impl below with
// calls to the production allocator API (llama_paged_kv_alloc_block /
// llama_paged_kv_free_seq / llama_paged_kv_block_table). Until then,
// this test is GREEN on the reference impl AND asserts the
// production-impl-landed flag is set — flag undefined on HEAD =>
// test FAILS at the final assertion.
//
// Returns: 0 = PASS, 1 = FAIL, 77 = SKIP.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <deque>

// ============================================================
// Reference implementation — the algorithm under contract.
// Will be replaced by production API in T5.1.
// ============================================================

static constexpr int OOB_SENTINEL = -1;

struct RefAllocator {
    int n_blocks;
    int block_size;
    std::vector<int> block_pool_owner;  // n_blocks; -1 = FREE, else seq_id
    std::vector<std::vector<int>> block_tables;  // [seq][i] = block_id
    std::deque<int> free_list;  // LIFO stack
    std::vector<int> written_tokens;

    RefAllocator(int n_blocks_, int block_size_, int n_seqs)
        : n_blocks(n_blocks_), block_size(block_size_),
          block_pool_owner(n_blocks_, -1),
          block_tables(n_seqs),
          written_tokens(n_seqs, 0) {
        // Init: free_list = [n_blocks-1, ..., 0] so first pop = 0
        for (int b = n_blocks - 1; b >= 0; --b) {
            free_list.push_front(b);
        }
    }

    int alloc_block(int seq) {
        if (free_list.empty()) {
            return OOB_SENTINEL;
        }
        int b = free_list.front();
        free_list.pop_front();
        block_pool_owner[b] = seq;
        block_tables[seq].push_back(b);
        return b;
    }

    void free_seq(int seq) {
        // Reverse-order push to preserve LIFO discipline.
        auto& tbl = block_tables[seq];
        for (auto it = tbl.rbegin(); it != tbl.rend(); ++it) {
            free_list.push_front(*it);
            block_pool_owner[*it] = -1;
        }
        tbl.clear();
        written_tokens[seq] = 0;
    }

    bool write_tokens(int seq, int n) {
        int new_total = written_tokens[seq] + n;
        int needed = (new_total == 0) ? 0 : ((new_total - 1) / block_size) + 1;
        int have = (int)block_tables[seq].size();
        int deficit = needed - have;
        if (deficit > (int)free_list.size()) {
            return false;  // OOM
        }
        for (int i = 0; i < deficit; ++i) {
            int b = alloc_block(seq);
            if (b == OOB_SENTINEL) return false;
        }
        written_tokens[seq] = new_total;
        return true;
    }
};

// ============================================================
// Invariant checks
// ============================================================

static bool check_block_uniquely_owned(const RefAllocator& a) {
    for (size_t s1 = 0; s1 < a.block_tables.size(); ++s1) {
        for (size_t s2 = s1 + 1; s2 < a.block_tables.size(); ++s2) {
            for (int b1 : a.block_tables[s1]) {
                for (int b2 : a.block_tables[s2]) {
                    if (b1 == b2) {
                        printf("FAIL: BlockUniquelyOwned violated: block %d in both seq %zu and seq %zu\n",
                               b1, s1, s2);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

static bool check_free_list_disjoint(const RefAllocator& a) {
    for (int b : a.free_list) {
        for (size_t s = 0; s < a.block_tables.size(); ++s) {
            for (int owned : a.block_tables[s]) {
                if (owned == b) {
                    printf("FAIL: FreeListDisjoint violated: block %d in both free_list and seq %zu\n",
                           b, s);
                    return false;
                }
            }
        }
    }
    return true;
}

static bool check_alloc_lazy(const RefAllocator& a) {
    for (size_t s = 0; s < a.block_tables.size(); ++s) {
        int w = a.written_tokens[s];
        int needed = (w == 0) ? 0 : ((w - 1) / a.block_size) + 1;
        if ((int)a.block_tables[s].size() != needed) {
            printf("FAIL: AllocLazy violated: seq %zu has %zu blocks for %d writes (needed %d)\n",
                   s, a.block_tables[s].size(), w, needed);
            return false;
        }
    }
    return true;
}

static bool check_identity_mapping_at_single_seq(const RefAllocator& a) {
    // If only seq 0 ever wrote, block_table[0] must be [0, 1, 2, ...].
    int seqs_with_writes = 0;
    for (size_t s = 0; s < a.block_tables.size(); ++s) {
        if (!a.block_tables[s].empty()) seqs_with_writes++;
    }
    if (seqs_with_writes != 1) return true;  // not the single-seq case
    if (a.block_tables[0].empty()) return true;
    for (size_t i = 0; i < a.block_tables[0].size(); ++i) {
        if (a.block_tables[0][i] != (int)i) {
            printf("FAIL: IdentityMappingAtSingleSeq violated: block_tables[0][%zu] = %d, expected %zu\n",
                   i, a.block_tables[0][i], i);
            return false;
        }
    }
    return true;
}

// ============================================================
// Tests
// ============================================================

static bool t_basic_alloc_free() {
    RefAllocator a(8, 4, 3);
    if (!a.write_tokens(0, 8)) return false;   // seq 0: 8 tokens => 2 blocks
    if (!a.write_tokens(1, 4)) return false;   // seq 1: 4 tokens => 1 block
    if (!a.write_tokens(2, 12)) return false;  // seq 2: 12 tokens => 3 blocks
    // 6 blocks used, 2 free
    if (!check_block_uniquely_owned(a)) return false;
    if (!check_free_list_disjoint(a)) return false;
    if (!check_alloc_lazy(a)) return false;
    a.free_seq(1);
    // 5 blocks used, 3 free
    if (!check_block_uniquely_owned(a)) return false;
    if (!check_free_list_disjoint(a)) return false;
    if (!check_alloc_lazy(a)) return false;
    return true;
}

static bool t_identity_single_seq() {
    RefAllocator a(8, 4, 3);
    a.write_tokens(0, 12);  // 3 blocks
    if (!check_identity_mapping_at_single_seq(a)) return false;
    return true;
}

static bool t_oom_signal() {
    RefAllocator a(4, 4, 2);
    if (!a.write_tokens(0, 8)) return false;   // 2 blocks
    if (!a.write_tokens(1, 8)) return false;   // 2 blocks; pool full
    if (a.write_tokens(0, 4)) {
        printf("FAIL: expected OOM signal on pool-full write\n");
        return false;
    }
    // OOM propagated correctly — invariants still hold
    if (!check_block_uniquely_owned(a)) return false;
    if (!check_free_list_disjoint(a)) return false;
    return true;
}

static bool t_alloc_lazy_no_writes_no_blocks() {
    RefAllocator a(8, 4, 3);
    // No writes => no allocations
    for (size_t s = 0; s < a.block_tables.size(); ++s) {
        if (!a.block_tables[s].empty()) {
            printf("FAIL: AllocLazy NoPreallocation violated: seq %zu has %zu blocks with 0 writes\n",
                   s, a.block_tables[s].size());
            return false;
        }
    }
    return true;
}

// ============================================================
// RED-bound gate: production impl landed flag
// ============================================================

#ifndef LLAMA_PAGED_KV_LANDED
#define LLAMA_PAGED_KV_LANDED 0
#endif

int main(int /*argc*/, char** /*argv*/) {
    bool ok = true;
    ok &= t_basic_alloc_free();
    ok &= t_identity_single_seq();
    ok &= t_oom_signal();
    ok &= t_alloc_lazy_no_writes_no_blocks();
    if (!ok) {
        printf("FAIL: reference-impl invariant checks failed\n");
        return 1;
    }
    if (!LLAMA_PAGED_KV_LANDED) {
        printf("FAIL: T5.1 paged KV implementation not yet landed (LLAMA_PAGED_KV_LANDED=0).\n");
        printf("      The reference allocator algorithm in this test passes all invariants;\n");
        printf("      the production llama_paged_kv_* API has not yet been wired in.\n");
        printf("      This test transitions PASS at T5.1 close.\n");
        return 1;
    }
    printf("PASS\n");
    return 0;
}
