// test-kv-shift-per-stream.cpp
//
// T3.6.T1 synthetic test for the K-shift per-stream contract from
// /home/llm/yarn-agentic/specs/kv-cache/k_shift_per_stream.allium
// and the matching TLA+ module KShiftPerStream.tla.
//
// Binding contract (KShiftIsolation, KShiftAppliesPerCell):
//
//   Given a context with n_stream > 1 carrying populated cells across
//   every stream, `llama_kv_cache_seq_add(seq_id, p0, p1, delta)`
//   followed by `llama_kv_cache_update(ctx)` MUST:
//     (a) complete without aborting,
//     (b) advance the position of the shifted seq's cells by `delta`,
//     (c) leave the position of other seqs' cells unchanged.
//
// State of this test on T3.5 HEAD: FAIL via GGML_ASSERT abort in
// `build_k_shift` at ik_llama.cpp/src/llama-build-context.cpp:178
// (assert kv_self.n_stream == 1). On post-T3.6.I.c1 it PASSES.
//
// Defensive coverage: runs the K-shift binding under BOTH
//   - LAYER split (single-device per layer; no CUDA_Split buffer), and
//   - GRAPH split (CUDA_Split per-device row-distributed K/V cache).
//
// LAYER split asserts the full K-shift binding (isolation + per-cell
// rotation). GRAPH split asserts the DOCUMENTED graceful limitation:
// llama_kv_cache_update MUST return rc=1 (and NOT crash with a CUDA
// illegal-memory-access) when can_shift is gated off under
// split_mode=GRAPH + n_stream>1. The gate is a deliberate guard
// (src/llama.cpp:get_can_shift) until the input-population layer is
// restructured to emit one inp_K_shift per stream.
//
// Returns: 0 = PASS, 1 = FAIL, 77 = SKIP (no model path supplied).
//
// Usage:
//   ./test-kv-shift-per-stream MODEL_PATH

#include "llama.h"
#include "common.h"
#include "llama-context.h"

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
    int split_mode_filter = 0;  // 0 = both, 1 = LAYER only, 2 = GRAPH only
};

Args parse_args(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL_PATH [layer|graph|both]\n", argv[0]);
        std::exit(77);
    }
    a.model_path = argv[1];
    if (argc >= 3) {
        std::string m = argv[2];
        if (m == "layer") a.split_mode_filter = 1;
        else if (m == "graph") a.split_mode_filter = 2;
        else if (m == "both") a.split_mode_filter = 0;
        else { std::fprintf(stderr, "split mode must be layer|graph|both\n"); std::exit(77); }
    }
    return a;
}

// Decode one token for each seq_id at pos 0, 1, ..., n_tok_per_seq-1.
// Populates cells across all streams so seq_add has cells to shift.
void prefill_each_stream(llama_context * ctx,
                         const std::vector<llama_token> & tokens,
                         int n_parallel) {
    for (int seq = 0; seq < n_parallel; ++seq) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            llama_token   tok      = tokens[i];
            llama_pos     pos      = (llama_pos)i;
            int32_t       n_seq_id = 1;
            llama_seq_id  seq_buf  = (llama_seq_id)seq;
            llama_seq_id *seq_ptr  = &seq_buf;
            int8_t        logits   = 0;
            llama_batch b = {};
            b.n_tokens = 1;
            b.token    = &tok;
            b.pos      = &pos;
            b.n_seq_id = &n_seq_id;
            b.seq_id   = &seq_ptr;
            b.logits   = &logits;
            const int rc = llama_decode(ctx, b);
            if (rc != 0) FAIL_AT("prefill decode rc=%d seq=%d pos=%d", rc, seq, (int)pos);
        }
    }
}

}  // namespace

namespace {

bool run_binding(const std::string & model_path, llama_split_mode split_mode,
                 const char * split_label) {
    std::fprintf(stdout, "\n=== K-shift binding under split_mode=%s ===\n", split_label);

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 999;
    model_params.split_mode   = split_mode;
    static const float ts[2] = {1.0f, 1.0f};
    if (split_mode == LLAMA_SPLIT_MODE_GRAPH) {
        model_params.tensor_split = ts;
    }

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        std::fprintf(stderr, "failed to load model %s under %s\n",
                     model_path.c_str(), split_label);
        return false;
    }

    constexpr int N_PARALLEL = 4;
    constexpr int N_TOK_PER_SEQ = 4;

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = 256 * N_PARALLEL;
    ctx_params.n_seq_max = N_PARALLEL;
    ctx_params.n_batch   = 64;
    ctx_params.n_ubatch  = 64;
    ctx_params.flash_attn = true;
    ctx_params.type_k    = GGML_TYPE_Q4_0;
    ctx_params.type_v    = GGML_TYPE_Q4_0;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        llama_free_model(model);
        FAIL_AT("failed to allocate llama_context under %s", split_label);
    }

    const char * text = "Hello world from a deterministic test prompt.";
    std::vector<llama_token> all_toks(64);
    int n = llama_tokenize(model, text, (int)strlen(text),
                           all_toks.data(), (int)all_toks.size(), true, false);
    if (n <= 0) FAIL_AT("tokenize returned %d under %s", n, split_label);
    if (n > N_TOK_PER_SEQ) n = N_TOK_PER_SEQ;
    std::vector<llama_token> tokens(all_toks.begin(), all_toks.begin() + n);

    prefill_each_stream(ctx, tokens, N_PARALLEL);

    std::vector<llama_pos> pos_max_pre(N_PARALLEL);
    for (int s = 0; s < N_PARALLEL; ++s) {
        pos_max_pre[s] = llama_kv_cache_seq_pos_max(ctx, (llama_seq_id)s);
    }
    std::fprintf(stdout, "[%s] pre-shift pos_max:", split_label);
    for (int s = 0; s < N_PARALLEL; ++s) {
        std::fprintf(stdout, " seq%d=%d", s, pos_max_pre[s]);
    }
    std::fprintf(stdout, "\n");

    constexpr llama_seq_id SHIFT_SEQ = 2;
    constexpr llama_pos DELTA = 5;
    llama_kv_cache_seq_add(ctx, SHIFT_SEQ, 0, (llama_pos)tokens.size(), DELTA);

    const int rc = llama_kv_cache_update(ctx);
    const bool expect_supported = (split_mode == LLAMA_SPLIT_MODE_LAYER);

    if (expect_supported) {
        if (rc != 0) FAIL_AT("[%s] llama_kv_cache_update rc=%d (expected 0)",
                             split_label, rc);

        std::vector<llama_pos> pos_max_post(N_PARALLEL);
        for (int s = 0; s < N_PARALLEL; ++s) {
            pos_max_post[s] = llama_kv_cache_seq_pos_max(ctx, (llama_seq_id)s);
        }
        std::fprintf(stdout, "[%s] post-shift pos_max:", split_label);
        for (int s = 0; s < N_PARALLEL; ++s) {
            std::fprintf(stdout, " seq%d=%d", s, pos_max_post[s]);
        }
        std::fprintf(stdout, "\n");

        if (pos_max_post[SHIFT_SEQ] != pos_max_pre[SHIFT_SEQ] + DELTA) {
            FAIL_AT("[%s] seq %d pos_max: pre=%d post=%d expected post=%d",
                    split_label, SHIFT_SEQ, pos_max_pre[SHIFT_SEQ],
                    pos_max_post[SHIFT_SEQ], pos_max_pre[SHIFT_SEQ] + DELTA);
        }
        for (int s = 0; s < N_PARALLEL; ++s) {
            if (s == SHIFT_SEQ) continue;
            if (pos_max_post[s] != pos_max_pre[s]) {
                FAIL_AT(
                    "[%s] stream %d pos_max changed under shift of seq %d: "
                    "pre=%d post=%d (cross-stream contamination — F2)",
                    split_label, s, SHIFT_SEQ, pos_max_pre[s], pos_max_post[s]);
            }
        }
    } else {
        // GRAPH split: documented limitation. Expect rc=1 (graceful
        // can_shift=false gate) — NOT a crash. This is the defensive
        // assertion: a future regression that lets K-shift run through
        // and CUDA-crash under graph-split would fail this branch
        // (process abort, exit code != 0).
        if (rc != 1) {
            FAIL_AT("[%s] llama_kv_cache_update rc=%d (expected 1 — "
                    "graph-split + n_stream>1 is gated off in "
                    "get_can_shift). A change of behavior here means "
                    "the gate was lifted prematurely.", split_label, rc);
        }
        std::fprintf(stdout, "[%s] documented graceful rc=1 (K-shift "
                     "gated off under graph-split — see "
                     "src/llama.cpp:get_can_shift)\n", split_label);
    }

    llama_free(ctx);
    llama_free_model(model);
    std::fprintf(stdout, "[%s] OK\n", split_label);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    llama_backend_init();

    bool ran_any = false;
    if (args.split_mode_filter != 2) {
        if (!run_binding(args.model_path, LLAMA_SPLIT_MODE_LAYER, "LAYER")) {
            llama_backend_free();
            return 1;
        }
        ran_any = true;
    }
    if (args.split_mode_filter != 1) {
        if (!run_binding(args.model_path, LLAMA_SPLIT_MODE_GRAPH, "GRAPH")) {
            llama_backend_free();
            return 1;
        }
        ran_any = true;
    }
    if (!ran_any) {
        std::fprintf(stderr, "no split mode selected\n");
        llama_backend_free();
        return 1;
    }

    llama_backend_free();
    std::fprintf(stdout, "\ntest-kv-shift-per-stream: PASS\n");
    return 0;
}
