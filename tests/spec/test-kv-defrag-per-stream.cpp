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
// Defensive coverage: runs the defrag binding under BOTH
//   - LAYER split (single-device per layer), and
//   - GRAPH split (CUDA_Split per-device row-distributed K/V cache).
// Both modes assert the full defrag binding. A regression that
// re-introduces the F3 flat hole-fill walk OR the F4 2D view bug
// would fail the GRAPH branch.
//
// Returns: 0 = PASS, 1 = FAIL, 77 = SKIP (no model path supplied).
//
// Usage:
//   ./test-kv-defrag-per-stream MODEL_PATH [layer|graph|both]

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

bool run_binding(const std::string & model_path, llama_split_mode split_mode,
                 const char * split_label) {
    std::fprintf(stdout, "\n=== Defrag binding under split_mode=%s ===\n", split_label);

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
        FAIL_AT("[%s] failed to allocate llama_context", split_label);
    }

    const char * text = "Hello world from a deterministic defrag test prompt.";
    std::vector<llama_token> all_toks(64);
    int n = llama_tokenize(model, text, (int)strlen(text),
                           all_toks.data(), (int)all_toks.size(), true, false);
    if (n <= 0) FAIL_AT("[%s] tokenize returned %d", split_label, n);
    if (n > N_TOK_PER_SEQ) n = N_TOK_PER_SEQ;
    std::vector<llama_token> tokens(all_toks.begin(), all_toks.begin() + n);

    prefill_each_stream(ctx, tokens, N_PARALLEL);

    const llama_pos pos_max_pre_s1 = llama_kv_cache_seq_pos_max(ctx, 1);
    const llama_pos pos_max_pre_s0 = llama_kv_cache_seq_pos_max(ctx, 0);
    std::fprintf(stdout, "[%s] pre-defrag pos_max: seq0=%d seq1=%d\n",
                 split_label, pos_max_pre_s0, pos_max_pre_s1);

    // Create a hole in stream 0 by removing the middle range [2, 5).
    constexpr llama_pos HOLE_P0 = 2;
    constexpr llama_pos HOLE_P1 = 5;
    if (!llama_kv_cache_seq_rm(ctx, /*seq_id=*/0, HOLE_P0, HOLE_P1)) {
        FAIL_AT("[%s] seq_rm returned false for stream 0 [%d, %d)",
                split_label, HOLE_P0, HOLE_P1);
    }

    llama_kv_cache_defrag(ctx);
    const int rc = llama_kv_cache_update(ctx);
    if (rc != 0) FAIL_AT("[%s] llama_kv_cache_update rc=%d (expected 0)",
                         split_label, rc);

    // DefragNoCrossStream — stream 1's pos_max is unchanged.
    const llama_pos pos_max_post_s1 = llama_kv_cache_seq_pos_max(ctx, 1);
    if (pos_max_post_s1 != pos_max_pre_s1) {
        FAIL_AT(
            "[%s] stream 1 pos_max changed under defrag of stream 0: "
            "pre=%d post=%d (cross-stream contamination — F3/F4)",
            split_label, pos_max_pre_s1, pos_max_post_s1);
    }

    // DefragCompactsPerStream — stream 0's pos_max preserved (defrag
    // moves cells but doesn't drop content; surviving cells keep their
    // pos values).
    const llama_pos pos_max_post_s0 = llama_kv_cache_seq_pos_max(ctx, 0);
    const llama_pos expected_s0_max = (llama_pos)(tokens.size() - 1);
    if (pos_max_post_s0 != expected_s0_max) {
        FAIL_AT(
            "[%s] stream 0 pos_max wrong after defrag: pre=%d post=%d "
            "expected=%d (defrag should preserve surviving cells)",
            split_label, pos_max_pre_s0, pos_max_post_s0, expected_s0_max);
    }

    std::fprintf(stdout, "[%s] post-defrag pos_max: seq0=%d seq1=%d\n",
                 split_label, pos_max_post_s0, pos_max_post_s1);

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
    std::fprintf(stdout, "\ntest-kv-defrag-per-stream: PASS\n");
    return 0;
}
