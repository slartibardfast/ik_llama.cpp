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
};

Args parse_args(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL_PATH\n", argv[0]);
        std::exit(77);
    }
    a.model_path = argv[1];
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

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    llama_backend_init();

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 999;
    // Use LAYER split (single-device per layer) for the K-shift binding
    // test. The CUDA_Split multi-device buffer type interacts badly with
    // per-stream view offsets at K-shift build time; that is a separate
    // problem from the per-stream loop correctness this test binds on.
    model_params.split_mode   = LLAMA_SPLIT_MODE_LAYER;

    llama_model * model = llama_model_load_from_file(args.model_path.c_str(),
                                                     model_params);
    if (!model) {
        std::fprintf(stderr, "failed to load model %s\n", args.model_path.c_str());
        llama_backend_free();
        return 77;
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
        FAIL_AT("failed to allocate llama_context");
    }

    // Tokenize a short prompt and reuse the first N_TOK_PER_SEQ tokens
    // for every seq's prefill.
    const char * text = "Hello world from a deterministic test prompt.";
    std::vector<llama_token> all_toks(64);
    int n = llama_tokenize(model, text, (int)strlen(text),
                           all_toks.data(), (int)all_toks.size(), true, false);
    if (n <= 0) FAIL_AT("tokenize returned %d", n);
    if (n > N_TOK_PER_SEQ) n = N_TOK_PER_SEQ;
    std::vector<llama_token> tokens(all_toks.begin(), all_toks.begin() + n);

    prefill_each_stream(ctx, tokens, N_PARALLEL);

    // Capture pre-shift seq_pos_max for every seq.
    std::vector<llama_pos> pos_max_pre(N_PARALLEL);
    for (int s = 0; s < N_PARALLEL; ++s) {
        pos_max_pre[s] = llama_kv_cache_seq_pos_max(ctx, (llama_seq_id)s);
    }
    std::fprintf(stdout, "pre-shift pos_max:");
    for (int s = 0; s < N_PARALLEL; ++s) {
        std::fprintf(stdout, " seq%d=%d", s, pos_max_pre[s]);
    }
    std::fprintf(stdout, "\n");

    // Shift seq 2 by delta=+5 across its full pos range.
    constexpr llama_seq_id SHIFT_SEQ = 2;
    constexpr llama_pos DELTA = 5;
    llama_kv_cache_seq_add(ctx, SHIFT_SEQ, /*p0=*/0,
                           /*p1=*/(llama_pos)tokens.size(), DELTA);

    // Trigger the K-shift. On T3.5 HEAD this aborts via GGML_ASSERT
    // (kv_self.n_stream == 1) in build_k_shift. On post-T3.6.I.c1 it
    // completes and the per-stream rope loop applies the rotation only
    // to seq 2's cells.
    const int rc = llama_kv_cache_update(ctx);
    if (rc != 0) FAIL_AT("llama_kv_cache_update rc=%d (expected 0)", rc);

    // Capture post-shift seq_pos_max for every seq.
    std::vector<llama_pos> pos_max_post(N_PARALLEL);
    for (int s = 0; s < N_PARALLEL; ++s) {
        pos_max_post[s] = llama_kv_cache_seq_pos_max(ctx, (llama_seq_id)s);
    }
    std::fprintf(stdout, "post-shift pos_max:");
    for (int s = 0; s < N_PARALLEL; ++s) {
        std::fprintf(stdout, " seq%d=%d", s, pos_max_post[s]);
    }
    std::fprintf(stdout, "\n");

    // KShiftAppliesPerCell — shifted seq's max pos advanced by exactly DELTA.
    if (pos_max_post[SHIFT_SEQ] != pos_max_pre[SHIFT_SEQ] + DELTA) {
        FAIL_AT("seq %d pos_max: pre=%d post=%d expected post=%d",
                SHIFT_SEQ, pos_max_pre[SHIFT_SEQ], pos_max_post[SHIFT_SEQ],
                pos_max_pre[SHIFT_SEQ] + DELTA);
    }

    // KShiftIsolation — every other seq's pos_max is unchanged.
    for (int s = 0; s < N_PARALLEL; ++s) {
        if (s == SHIFT_SEQ) continue;
        if (pos_max_post[s] != pos_max_pre[s]) {
            FAIL_AT(
                "stream %d pos_max changed under shift of seq %d: "
                "pre=%d post=%d (cross-stream contamination — F2)",
                s, SHIFT_SEQ, pos_max_pre[s], pos_max_post[s]);
        }
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    std::fprintf(stdout, "test-kv-shift-per-stream: PASS\n");
    return 0;
}
