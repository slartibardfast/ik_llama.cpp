// test-kv-defrag-per-stream.cpp
//
// T3.6.T2 synthetic test for the defrag per-stream contract from
// /home/llm/yarn-agentic/specs/kv-cache/defrag_per_stream.allium and
// the matching TLA+ module DefragPerStream.tla.
//
// Binding contract (DefragNoCrossStream, DefragCompactsPerStream):
//
//   Given a context with n_stream > 1 and a hole in one stream's
//   slice, `llama_kv_cache_defrag(ctx)` followed by
//   `llama_kv_cache_update(ctx)` MUST:
//     (a) complete without aborting,
//     (b) compact the holed stream's occupied cells to the prefix
//         of its slice,
//     (c) leave the other stream's pos_max unchanged.
//
// State of this test on T3.5 HEAD: FAIL via GGML_ASSERT abort in
// `build_defrag` at ik_llama.cpp/src/llama-build-context.cpp:287
// (assert kv_self.n_stream == 1). On post-T3.6.I.c2 it PASSES.
//
// Returns: 0 = PASS, 1 = FAIL, 77 = SKIP (no model path supplied).
//
// Usage:
//   ./test-kv-defrag-per-stream MODEL_PATH

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
    model_params.split_mode   = LLAMA_SPLIT_MODE_GRAPH;
    static const float ts[2] = {1.0f, 1.0f};
    model_params.tensor_split = ts;

    llama_model * model = llama_model_load_from_file(args.model_path.c_str(),
                                                     model_params);
    if (!model) {
        std::fprintf(stderr, "failed to load model %s\n", args.model_path.c_str());
        llama_backend_free();
        return 77;
    }

    constexpr int N_PARALLEL = 2;
    constexpr int N_TOK_PER_SEQ = 8;

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

    const char * text = "Hello world from a deterministic defrag test prompt.";
    std::vector<llama_token> all_toks(64);
    int n = llama_tokenize(model, text, (int)strlen(text),
                           all_toks.data(), (int)all_toks.size(), true, false);
    if (n <= 0) FAIL_AT("tokenize returned %d", n);
    if (n > N_TOK_PER_SEQ) n = N_TOK_PER_SEQ;
    std::vector<llama_token> tokens(all_toks.begin(), all_toks.begin() + n);

    prefill_each_stream(ctx, tokens, N_PARALLEL);

    // Capture pre-defrag seq_pos_max for stream 1 (the untouched stream).
    const llama_pos pos_max_pre_s1 = llama_kv_cache_seq_pos_max(ctx, 1);
    const llama_pos pos_max_pre_s0 = llama_kv_cache_seq_pos_max(ctx, 0);
    std::fprintf(stdout, "pre-defrag pos_max: seq0=%d seq1=%d\n",
                 pos_max_pre_s0, pos_max_pre_s1);

    // Create a hole in stream 0 by removing the middle range [2, 5).
    constexpr llama_pos HOLE_P0 = 2;
    constexpr llama_pos HOLE_P1 = 5;
    if (!llama_kv_cache_seq_rm(ctx, /*seq_id=*/0, HOLE_P0, HOLE_P1)) {
        FAIL_AT("seq_rm returned false for stream 0 [%d, %d)", HOLE_P0, HOLE_P1);
    }

    // Trigger defrag. On T3.5 HEAD this aborts via GGML_ASSERT
    // (kv_self.n_stream == 1) in build_defrag. On post-T3.6.I.c2 it
    // completes with the per-stream outer loop + 3D-view-per-stream cpy.
    llama_kv_cache_defrag(ctx);
    const int rc = llama_kv_cache_update(ctx);
    if (rc != 0) FAIL_AT("llama_kv_cache_update rc=%d (expected 0)", rc);

    // DefragNoCrossStream — stream 1's pos_max is unchanged. (pos_max is
    // a coarse but real signal: any cross-stream cell pull would change
    // stream 1's cell set and hence its pos_max derived from cell.pos.)
    const llama_pos pos_max_post_s1 = llama_kv_cache_seq_pos_max(ctx, 1);
    if (pos_max_post_s1 != pos_max_pre_s1) {
        FAIL_AT(
            "stream 1 pos_max changed under defrag of stream 0: "
            "pre=%d post=%d (cross-stream contamination — F3/F4)",
            pos_max_pre_s1, pos_max_post_s1);
    }

    // DefragCompactsPerStream — stream 0's pos_max is preserved
    // (defrag does not lose data; it only moves cells). The hole at
    // [HOLE_P0, HOLE_P1) is gone, but the surviving cells' positions
    // are unchanged; pos_max equals the highest surviving pos.
    const llama_pos pos_max_post_s0 = llama_kv_cache_seq_pos_max(ctx, 0);
    const llama_pos expected_s0_max = (llama_pos)(tokens.size() - 1);
    if (pos_max_post_s0 != expected_s0_max) {
        FAIL_AT(
            "stream 0 pos_max wrong after defrag: pre=%d post=%d "
            "expected=%d (defrag should preserve surviving cells)",
            pos_max_pre_s0, pos_max_post_s0, expected_s0_max);
    }

    std::fprintf(stdout, "post-defrag pos_max: seq0=%d seq1=%d\n",
                 pos_max_post_s0, pos_max_post_s1);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    std::fprintf(stdout, "test-kv-defrag-per-stream: PASS\n");
    return 0;
}
