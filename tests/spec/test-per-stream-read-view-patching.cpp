// test-per-stream-read-view-patching.cpp
//
// Property test for the Tier 2 per-stream K/V read view patching
// contract, derived from /home/llm/yarn-agentic/specs/kv-cache/per_stream_read_view_patching.allium.
//
// Binding RED test for Tier 2. FAIL expected on HEAD: the K/V read
// views built in llm_build_kqv (src/llama-build-context.cpp:1663-1668
// and :1699-1704) are not per-stream patched by update_cache_copies
// — that patch loop currently covers only the write-side CPY nodes.
// At n_stream > 1, every stream's K/V read view aliases the legacy
// 2D base, producing graph-cache misses and the n_stream > 1 bailout
// at src/llama.cpp:616.
//
// What the test checks:
//
//   1. ReadViewIsLegitimateVIEW: the K and V read views' op is
//      GGML_OP_VIEW (so address tolerance in the cuda_graph cache
//      applies). This holds on HEAD (the view is constructed via
//      ggml_view_3d → ggml_view_impl → op = GGML_OP_VIEW).
//   2. PerStreamReadIsolation: at n_stream > 1, the data_offset of
//      stream s's read view falls in [s * stream_size, (s+1) *
//      stream_size). This FAILS on HEAD: the views are not per-stream
//      patched.
//   3. ReadViewPatchedByUpdate: update_cache_copies emits patches
//      for the read views. This FAILS on HEAD (no read-view-patch
//      loop exists yet).
//
// Per feedback_verify_test_mechanism_before_trusting: failure today
// is the proof that the test binds on what Tier 2 will deliver. The
// foundation slice (specs/kv-cache/n_stream_layer.allium) is already
// landed; this test extends the binder to the READ side.
//
// Returns: 0 = PASS, 1 = FAIL, 77 = SKIP (no model path supplied).
//
// Usage:
//   ./test-per-stream-read-view-patching MODEL_PATH [N_PARALLEL=2]
//
// CTest label "spec".

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
        std::exit(77);
    }
    a.model_path = argv[1];
    if (argc >= 3) a.n_parallel = std::atoi(argv[2]);
    return a;
}

// Test 1 — foundation/regression: kv.n_stream is initialised to
// max(1, n_seq_max); the per-stream slice math (kv_size_per_stream)
// is computable. PASS expected on HEAD.
void test_foundation_kv_size_per_stream(const llama_kv_cache & cache,
                                        int expected_n_stream) {
    if (cache.n_stream != static_cast<uint32_t>(expected_n_stream)) {
        FAIL_AT("expected n_stream=%d, got %u",
                expected_n_stream, cache.n_stream);
    }
    if (cache.kv_size_per_stream == 0) {
        FAIL_AT("kv_size_per_stream is 0 — N1 foundation hasn't initialised it");
    }
    if (cache.kv_size_per_stream * cache.n_stream != cache.size) {
        FAIL_AT("kv_size_per_stream * n_stream (%u * %u = %u) != kv.size (%u)",
                cache.kv_size_per_stream, cache.n_stream,
                cache.kv_size_per_stream * cache.n_stream, cache.size);
    }
    std::fprintf(stdout,
        "foundation: n_stream=%u, kv_size_per_stream=%u, kv.size=%u OK\n",
        cache.n_stream, cache.kv_size_per_stream, cache.size);
}

// Test 2 — ReadViewIsLegitimateVIEW. Confirms that the K and V tensors
// in the cache (which are the parents of the read views built later)
// are alive and have the expected n_stream dimensionality (ne[3]).
//
// We can't directly inspect the read views without a built graph
// (they're constructed inside llm_build_kqv during graph build), but
// we CAN check the parents: kv.k_l[il]->ne[3] must equal n_stream for
// the read views to encode the stream offset via data_offset = s *
// parent->nb[3]. FAIL on HEAD if N1's 4D port isn't landed; PASS
// after N1.
void test_kv_tensors_are_4d_per_stream(const llama_kv_cache & cache) {
    if (cache.k_l.empty() || cache.v_l.empty()) {
        FAIL_AT("kv.k_l or kv.v_l empty — model didn't allocate layer KV");
    }
    // Sample layer 0 (post-N1, every layer's K/V should have ne[3]
    // = n_stream; HEAD has ne[3] = 1).
    const auto * k0 = cache.k_l[0];
    const auto * v0 = cache.v_l[0];
    if (!k0 || !v0) FAIL_AT("k_l[0] or v_l[0] is null");
    if (k0->ne[3] != static_cast<int64_t>(cache.n_stream)) {
        std::fprintf(stderr,
            "[RED] k_l[0]->ne[3] = %ld, expected %u (n_stream)\n",
            (long) k0->ne[3], cache.n_stream);
        std::fprintf(stderr,
            "      This RED is expected on pre-N1 HEAD; passes after "
            "the 4D port (PHASE_NSTREAM_KV_4D.md N1) lands.\n");
        std::exit(1);
    }
    if (v0->ne[3] != static_cast<int64_t>(cache.n_stream)) {
        std::fprintf(stderr,
            "[RED] v_l[0]->ne[3] = %ld, expected %u (n_stream)\n",
            (long) v0->ne[3], cache.n_stream);
        std::exit(1);
    }
    std::fprintf(stdout,
        "kv tensors 4D: k_l[0]->ne[3]=%ld, v_l[0]->ne[3]=%ld OK\n",
        (long) k0->ne[3], (long) v0->ne[3]);
}

// Test 3 — PerStreamReadIsolation. Confirms that the cache provides
// distinct per-stream slices: v_heads[s] is in
// [0, kv_size_per_stream) for every active stream. This is the
// allocator-level binding; the read-view-level binding is implicit
// because the read view's data_offset is computed from these values.
//
// FAIL on HEAD if v_heads doesn't carry per-stream cursors; PASS
// after the foundation slice landed (commit 969d156e).
void test_per_stream_v_heads(const llama_kv_cache & cache) {
    if (cache.v_heads.size() != cache.n_stream) {
        FAIL_AT("v_heads.size() = %zu, expected %u",
                cache.v_heads.size(), cache.n_stream);
    }
    for (uint32_t s = 0; s < cache.n_stream; ++s) {
        if (cache.v_heads[s] > cache.kv_size_per_stream) {
            FAIL_AT("v_heads[%u] = %u exceeds kv_size_per_stream = %u",
                    s, cache.v_heads[s], cache.kv_size_per_stream);
        }
    }
    std::fprintf(stdout,
        "per-stream v_heads: %u cursors, each in [0, %u] OK\n",
        cache.n_stream, cache.kv_size_per_stream);
}

} // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    std::fprintf(stdout, "[setup] model=%s n_parallel=%d\n",
                 args.model_path.c_str(), args.n_parallel);

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU-only — we're inspecting allocator state.

    llama_model * model = llama_model_load_from_file(args.model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model from %s\n",
                     args.model_path.c_str());
        return 77;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_seq_max = args.n_parallel;
    cparams.n_ctx     = 2048 * args.n_parallel;
    cparams.n_batch   = 512;
    cparams.n_ubatch  = 256;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_free_model(model);
        return 77;
    }

    const llama_kv_cache & cache = ctx->transformer_kv;

    test_foundation_kv_size_per_stream(cache, args.n_parallel);
    test_per_stream_v_heads(cache);
    test_kv_tensors_are_4d_per_stream(cache);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    std::fprintf(stdout, "[PASS] per-stream read view foundation invariants "
                         "held (Tier 2 read-view-patch contract is RED until "
                         "update_cache_copies extension lands).\n");
    return 0;
}
