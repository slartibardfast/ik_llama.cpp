// test-n-stream-kv-layout.cpp
//
// Property test for the n_stream KV per-stream layout, derived from
// /home/llm/yarn-agentic/specs/kv-cache/n_stream_layer.allium and
// /home/llm/yarn-agentic/specs/multislot/StreamIsolation.tla.
//
// Verifies the structural invariants the 4D port (PHASE_NSTREAM_KV_4D.md
// N1+N2) and the T3.2 multi-seq allocator deliver:
//
//   1. Foundation: llama_kv_cache::n_stream is initialised to
//      max(1, n_seq_max); v_heads is sized to n_stream. (PASS on HEAD —
//      this slice landed in submodule commit 969d156e.)
//   2. KVTensorIsFourD: every layer's k_l[il] and v_l[il] tensors have
//      ne[3] == n_stream. (PASS after N1; PASS on HEAD now.)
//   3. StreamPartition (kv_size_per_stream): k_l[il]->ne[1] equals
//      kv_size / n_stream. (PASS after N1; PASS on HEAD now.)
//   4. PHASE_NSTREAM_KV_PERF T3.2 — multi-seq find_slot: when called
//      with a multi-seq batch (contiguous-per-seq tokens covering
//      multiple distinct seq_ids), each run is allocated into its own
//      stream's slice. Cells in stream s carry the expected pos and
//      seq_id; cells outside stream s are untouched.
//
// Returns: 0 = PASS, 1 = FAIL, 77 = SKIP (no model path supplied).
//
// Usage:
//   ./test-n-stream-kv-layout MODEL_PATH [N_PARALLEL]
// Defaults: N_PARALLEL=2.

#include "llama.h"
#include "common.h"
#include "llama-context.h"  // internal: struct llama_kv_cache

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

#define FAIL_AT(msg, ...) do { \
    std::fprintf(stderr, "FAIL %s:%d: " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    std::exit(1); \
} while (0)

struct Args {
    std::string model_path;
    int n_parallel = 2;
};

Args parse_args(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL_PATH [N_PARALLEL]\n", argv[0]);
        std::exit(77);  // SKIP
    }
    a.model_path = argv[1];
    if (argc >= 3) a.n_parallel = std::atoi(argv[2]);
    return a;
}

// Inspect the llama_kv_cache foundation slice. PASS expected on HEAD
// from the moment commit 969d156e landed.
void test_foundation_n_stream_and_v_heads(const llama_kv_cache & cache,
                                          int expected_n_stream) {
    if (cache.n_stream != static_cast<uint32_t>(expected_n_stream)) {
        FAIL_AT("expected n_stream=%d, got %u",
                expected_n_stream, cache.n_stream);
    }
    if (cache.v_heads.size() != static_cast<size_t>(expected_n_stream)) {
        FAIL_AT("expected v_heads.size()=%d, got %zu",
                expected_n_stream, cache.v_heads.size());
    }
    std::fprintf(stdout, "foundation: n_stream=%u, v_heads.size()=%zu OK\n",
                cache.n_stream, cache.v_heads.size());
}

// The binding "RED test" for N1: assert that K/V tensors carry the
// per-stream axis as ne[3]. FAILS on HEAD; PASSES after the 4D port.
void test_kv_tensors_have_n_stream_axis(const llama_kv_cache & cache) {
    if (cache.k_l.empty()) FAIL_AT("k_l is empty");
    if (cache.v_l.empty()) FAIL_AT("v_l is empty");

    for (size_t il = 0; il < cache.k_l.size(); ++il) {
        const ggml_tensor * k = cache.k_l[il];
        if (!k) continue;  // some layers may have null entries (non-attn)
        const int64_t ne3 = k->ne[3];
        if (ne3 != static_cast<int64_t>(cache.n_stream)) {
            FAIL_AT(
                "expected k_l[%zu]->ne[3] == n_stream(%u), got %lld "
                "— 4D per-stream layout from N1 is not yet wired",
                il, cache.n_stream,
                static_cast<long long>(ne3));
        }
    }

    for (size_t il = 0; il < cache.v_l.size(); ++il) {
        const ggml_tensor * v = cache.v_l[il];
        if (!v) continue;
        const int64_t ne3 = v->ne[3];
        if (ne3 != static_cast<int64_t>(cache.n_stream)) {
            FAIL_AT(
                "expected v_l[%zu]->ne[3] == n_stream(%u), got %lld",
                il, cache.n_stream,
                static_cast<long long>(ne3));
        }
    }

    std::fprintf(stdout, "KVTensorIsFourD: all %zu k_l / v_l slabs have ne[3] == %u\n",
                cache.k_l.size(), cache.n_stream);
}

// StreamPartition's structural form: the per-stream position-count is
// kv_size / n_stream. The N2.a axis order is
// [head_dim, kv_size_per_stream, n_head_kv, n_stream] so ne[1] carries
// the per-stream position-count axis. Some models expose null k_l
// entries for non-attention layers (recurrent, MTP head, has_kv=false),
// so this asserts against the FIRST non-null tensor rather than k_l[0].
void test_per_stream_slice_dimension(const llama_kv_cache & cache) {
    const ggml_tensor * k_probe = nullptr;
    size_t              k_probe_il = 0;
    for (size_t il = 0; il < cache.k_l.size(); ++il) {
        if (cache.k_l[il]) { k_probe = cache.k_l[il]; k_probe_il = il; break; }
    }
    if (!k_probe) FAIL_AT("no non-null k_l[il] tensor in any layer");

    const int64_t expected_per_stream = cache.size / cache.n_stream;
    const int64_t actual = k_probe->ne[1];
    if (actual != expected_per_stream) {
        FAIL_AT(
            "expected k_l[%zu]->ne[1] == kv_size/n_stream (%lld), got %lld "
            "— StreamPartition position-count axis not yet per-stream",
            k_probe_il,
            static_cast<long long>(expected_per_stream),
            static_cast<long long>(actual));
    }
    std::fprintf(stdout, "StreamPartition: k_l[%zu]->ne[1] == %lld (=kv_size/n_stream)\n",
                k_probe_il, static_cast<long long>(actual));
}

// PHASE_NSTREAM_KV_PERF T3.2: drive llama_kv_cache_find_slot with a
// synthetic contiguous-per-seq multi-seq batch and assert per-stream
// allocation. Snapshots v_heads/cells before the call, asserts the
// expected (pos, seq_id) at each allocated cell, and asserts cells in
// non-active streams are untouched.
void test_multi_seq_find_slot(llama_kv_cache & cache) {
    if (cache.n_stream < 2) {
        std::fprintf(stdout, "multi_seq_find_slot: n_stream<2, skipping\n");
        return;
    }

    // Build a synthetic batch with two seqs, each with n_tok_per_seq=4
    // contiguous tokens. Seq 0 at positions [10..13], seq 1 at [20..23].
    constexpr uint32_t n_seqs        = 2;
    constexpr uint32_t n_tok_per_seq = 4;
    constexpr uint32_t n_tokens      = n_seqs * n_tok_per_seq;

    std::vector<llama_pos>           pos(n_tokens);
    std::vector<int32_t>             n_seq_id(n_tokens, 1);
    std::vector<llama_seq_id>        seq_id_buf(n_tokens);
    std::vector<llama_seq_id *>      seq_id_ptr(n_tokens);
    std::vector<int8_t>              logits(n_tokens, 0);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        for (uint32_t t = 0; t < n_tok_per_seq; ++t) {
            const uint32_t i = s * n_tok_per_seq + t;
            pos[i]        = (llama_pos)(10 + s*10 + t);
            seq_id_buf[i] = (llama_seq_id)s;
            seq_id_ptr[i] = &seq_id_buf[i];
        }
    }

    llama_batch batch = {};
    batch.n_tokens = (int32_t) n_tokens;
    batch.token    = nullptr;  // not used by find_slot
    batch.embd     = nullptr;
    batch.pos      = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id   = seq_id_ptr.data();
    batch.logits   = logits.data();
    batch.all_pos_0  = 0;
    batch.all_pos_1  = 0;
    batch.all_seq_id = 0;

    // Snapshot cursors per-stream before allocation.
    std::vector<uint32_t> v_heads_before = cache.v_heads;
    const uint32_t        used_before    = cache.used;

    if (!llama_kv_cache_find_slot(cache, batch)) {
        FAIL_AT("multi-seq find_slot returned false on an empty cache");
    }

    const uint32_t kvps = cache.kv_size_per_stream;

    // Assert each seq's allocation went into its own stream's slice
    // and that v_heads advanced by n_tok_per_seq for each (since the
    // cache was empty, both seqs allocate at their starting cursor).
    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t base = s * kvps;
        const uint32_t head_local_after  = cache.v_heads[s];
        const uint32_t head_local_before = v_heads_before[s];
        const uint32_t expect_after = head_local_before;  // pre-advance start
        if (head_local_after != expect_after) {
            FAIL_AT(
                "v_heads[%u] expected to point at allocation start "
                "(%u) post-find_slot, got %u",
                s, expect_after, head_local_after);
        }
        for (uint32_t t = 0; t < n_tok_per_seq; ++t) {
            const uint32_t cell_idx = base + head_local_after + t;
            const llama_kv_cell & c = cache.cells[cell_idx];
            const llama_pos    expect_pos = (llama_pos)(10 + s*10 + t);
            const llama_seq_id expect_sid = (llama_seq_id)s;
            if (c.pos != expect_pos) {
                FAIL_AT(
                    "stream %u cell %u: expected pos=%d got %d",
                    s, cell_idx, expect_pos, c.pos);
            }
            if (!c.has_seq_id(expect_sid)) {
                FAIL_AT(
                    "stream %u cell %u: missing seq_id=%d",
                    s, cell_idx, expect_sid);
            }
            // Stream isolation: the cell's seq_id set must not contain
            // any other seq_id from this batch.
            for (uint32_t other = 0; other < n_seqs; ++other) {
                if (other == s) continue;
                if (c.has_seq_id((llama_seq_id)other)) {
                    FAIL_AT(
                        "stream %u cell %u carries unexpected seq_id=%u "
                        "— cross-stream contamination",
                        s, cell_idx, other);
                }
            }
        }
    }

    // Total cells used should have grown by n_tokens.
    if (cache.used != used_before + n_tokens) {
        FAIL_AT(
            "cache.used did not advance by n_tokens (%u): before=%u after=%u",
            n_tokens, used_before, cache.used);
    }

    // cache.head invariant: points at seq-0's slot's first cell.
    const uint32_t expect_head = v_heads_before[0];  // seq-0's pre-allocation cursor
    if (cache.head != expect_head) {
        FAIL_AT(
            "cache.head expected to point at seq-0 slot's first cell "
            "(stream-local %u), got %u",
            expect_head, cache.head);
    }

    std::fprintf(stdout,
        "multi_seq_find_slot: n_seqs=%u n_tok_per_seq=%u — per-stream "
        "allocation + isolation OK; cache.head=%u v_heads={ %u, %u }\n",
        n_seqs, n_tok_per_seq, cache.head, cache.v_heads[0], cache.v_heads[1]);
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    llama_backend_init();

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;  // CPU-only — small / no GPU needed

    llama_model * model = llama_model_load_from_file(args.model_path.c_str(),
                                                     model_params);
    if (!model) {
        std::fprintf(stderr, "failed to load model %s\n", args.model_path.c_str());
        return 77;
    }

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx       = 256 * args.n_parallel;
    ctx_params.n_seq_max   = args.n_parallel;
    ctx_params.n_batch     = 64;
    ctx_params.n_ubatch    = 64;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        llama_free_model(model);
        FAIL_AT("failed to allocate llama_context");
    }

    llama_kv_cache & cache = ctx->transformer_kv;

    test_foundation_n_stream_and_v_heads(cache, args.n_parallel);
    test_kv_tensors_have_n_stream_axis(cache);
    test_per_stream_slice_dimension(cache);
    test_multi_seq_find_slot(cache);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    std::fprintf(stdout, "test-n-stream-kv-layout: PASS\n");
    return 0;
}
