// test-n-stream-kv-layout.cpp
//
// Property test for the n_stream KV per-stream layout, derived from
// /home/llm/yarn-agentic/specs/kv-cache/n_stream_layer.allium and
// /home/llm/yarn-agentic/specs/multislot/StreamIsolation.tla.
//
// Verifies the structural invariants the 4D port (PHASE_NSTREAM_KV_4D.md
// N1+N2) will deliver:
//
//   1. Foundation: llama_kv_cache::n_stream is initialised to
//      max(1, n_seq_max); v_heads is sized to n_stream. (PASS on HEAD —
//      this slice landed in submodule commit 969d156e.)
//   2. KVTensorIsFourD: every layer's k_l[il] and v_l[il] tensors have
//      ne[3] == n_stream. (FAIL on HEAD — K/V are still 3D with
//      ne[3] == 1. PASS after N1 lands.)
//   3. StreamPartition (kv_size_per_stream): k_l[il]->ne[1] equals
//      kv_size / n_stream, not the full kv_size. (FAIL on HEAD — the
//      tensor's row count is the global kv_size.)
//
// Behaves as the binding "RED test" for N1: it FAILS today, will PASS
// when the 4D port lands. Per
// /home/llm/.claude/projects/-home-llm-yarn-agentic/memory/feedback_verify_test_mechanism_before_trusting.md,
// failure today is the proof that the test binds on what N1 delivers,
// not on a tautology.
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

// StreamPartition's structural form: the per-stream slice size is
// kv_size / n_stream. Today the tensor's row dimension carries the
// full kv_size. Will pass when N1 reshapes the allocation. Some models
// expose null k_l entries for non-attention layers (recurrent, MTP head,
// has_kv=false), so this asserts against the FIRST non-null tensor
// rather than k_l[0] specifically.
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
            "— StreamPartition row dimension not yet per-stream",
            k_probe_il,
            static_cast<long long>(expected_per_stream),
            static_cast<long long>(actual));
    }
    std::fprintf(stdout, "StreamPartition: k_l[%zu]->ne[1] == %lld (=kv_size/n_stream)\n",
                k_probe_il, static_cast<long long>(actual));
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

    const llama_kv_cache & cache = ctx->transformer_kv;

    test_foundation_n_stream_and_v_heads(cache, args.n_parallel);
    test_kv_tensors_have_n_stream_axis(cache);   // RED today
    test_per_stream_slice_dimension(cache);       // RED today

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    std::fprintf(stdout, "test-n-stream-kv-layout: PASS\n");
    return 0;
}
