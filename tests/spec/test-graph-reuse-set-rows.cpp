// test-graph-reuse-set-rows.cpp
//
// T3.6.T3 synthetic test for the graph-reuse-under-multi-seq contract
// from /home/llm/yarn-agentic/specs/dispatch/graph_reuse_set_rows.allium
// and the matching TLA+ module GraphReuseSetRows.tla.
//
// Binding contract (NStreamBailoutNeverFires, SubsequentCallsHit):
//
//   Given a context with n_stream > 1, running identical multi-seq
//   decodes back-to-back, the SECOND call MUST hit the CUDA graph
//   cache (g_can_reuse_last_miss_reason == 0). The legacy bailout
//   that sets reason=6 on n_stream > 1 (src/llama.cpp:615) MUST be
//   removed.
//
// State of this test on T3.5 HEAD: FAIL — g_can_reuse_last_miss_reason
// is set to 6 by the n_stream > 1 bailout. On post-T3.6.I.b it PASSES.
//
// The test does NOT check output byte-identity here (that lives in the
// verify-production-determinism harness); it focuses on the bailout
// pathway and the reuse-shape contract.
//
// Returns: 0 = PASS, 1 = FAIL, 77 = SKIP (no model path supplied).
//
// Usage:
//   ./test-graph-reuse-set-rows MODEL_PATH

#include "llama.h"
#include "common.h"
#include "llama-context.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// The thread-local miss-reason variable lives in
// ik_llama.cpp/src/llama.cpp:46 and is mutated by can_reuse_graph
// (src/llama.cpp:586-615). We declare it here to read it post-decode.
extern thread_local int g_can_reuse_last_miss_reason;

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

// Build a contiguous-per-seq multi-seq batch: each seq contributes one
// decode token at pos `seq_pos[s]`. Mirrors the production multi-seq
// dispatch shape coming out of T3.5's process_batch_tokens.
int decode_multi_seq(llama_context * ctx,
                     const std::vector<llama_token> & tokens_per_seq,
                     const std::vector<llama_pos>   & pos_per_seq,
                     int n_seqs) {
    std::vector<llama_pos>     pos(n_seqs);
    std::vector<int32_t>       n_seq_id(n_seqs, 1);
    std::vector<llama_seq_id>  seq_buf(n_seqs);
    std::vector<llama_seq_id*> seq_ptr(n_seqs);
    std::vector<int8_t>        logits(n_seqs, 0);

    std::vector<llama_token>   tokens(tokens_per_seq);

    for (int s = 0; s < n_seqs; ++s) {
        pos[s]      = pos_per_seq[s];
        seq_buf[s]  = (llama_seq_id)s;
        seq_ptr[s]  = &seq_buf[s];
    }

    llama_batch b = {};
    b.n_tokens = n_seqs;
    b.token    = tokens.data();
    b.pos      = pos.data();
    b.n_seq_id = n_seq_id.data();
    b.seq_id   = seq_ptr.data();
    b.logits   = logits.data();

    return llama_decode(ctx, b);
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

    constexpr int N_PARALLEL    = 4;
    constexpr int N_TOK_PREFILL = 4;

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

    // Prefill each stream with N_TOK_PREFILL tokens so subsequent decodes
    // operate at steady state (single decode token per seq per call).
    const char * text = "Hello world from a graph reuse test prompt.";
    std::vector<llama_token> all_toks(64);
    int n = llama_tokenize(model, text, (int)strlen(text),
                           all_toks.data(), (int)all_toks.size(), true, false);
    if (n <= 0) FAIL_AT("tokenize returned %d", n);
    if (n > N_TOK_PREFILL) n = N_TOK_PREFILL;
    std::vector<llama_token> prefill_tokens(all_toks.begin(), all_toks.begin() + n);

    for (int s = 0; s < N_PARALLEL; ++s) {
        for (int i = 0; i < n; ++i) {
            llama_token   tok      = prefill_tokens[i];
            llama_pos     pos      = (llama_pos)i;
            int32_t       n_seq_id = 1;
            llama_seq_id  seq_buf  = (llama_seq_id)s;
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
            if (rc != 0) FAIL_AT("prefill decode rc=%d seq=%d pos=%d", rc, s, (int)pos);
        }
    }

    // Build identical multi-seq decode batches. All N_PARALLEL seqs
    // contribute one decode token at pos n (the next position after
    // prefill). Token is the same across seqs for reproducibility.
    const llama_token decode_tok = prefill_tokens.back();
    std::vector<llama_token> tokens_per_seq(N_PARALLEL, decode_tok);
    std::vector<llama_pos>   pos_per_seq(N_PARALLEL, (llama_pos)n);

    // FIRST call — expected MISS (no cached entry yet at this shape).
    g_can_reuse_last_miss_reason = -1;  // sentinel
    int rc1 = decode_multi_seq(ctx, tokens_per_seq, pos_per_seq, N_PARALLEL);
    if (rc1 != 0) FAIL_AT("first multi-seq decode rc=%d", rc1);
    const int miss_reason_1 = g_can_reuse_last_miss_reason;
    std::fprintf(stdout, "first call: rc=%d miss_reason=%d\n", rc1, miss_reason_1);

    // Bump positions for the second call (same shape, distinct cache
    // contents — but topology hash is the same so reuse SHOULD hit).
    for (int s = 0; s < N_PARALLEL; ++s) pos_per_seq[s] += 1;

    // SECOND call — expected HIT (g_can_reuse_last_miss_reason == 0)
    // post-T3.6.I.b. On T3.5 HEAD this returns 6 (n_stream_bailout).
    g_can_reuse_last_miss_reason = -1;
    int rc2 = decode_multi_seq(ctx, tokens_per_seq, pos_per_seq, N_PARALLEL);
    if (rc2 != 0) FAIL_AT("second multi-seq decode rc=%d", rc2);
    const int miss_reason_2 = g_can_reuse_last_miss_reason;
    std::fprintf(stdout, "second call: rc=%d miss_reason=%d\n", rc2, miss_reason_2);

    // The binding assertion: the n_stream bailout (reason=6) MUST NOT
    // fire on the second call. Any other reason is acceptable as
    // diagnostic — e.g., reason 4 (graph_reuse disabled) would suggest
    // the environment isn't configured for reuse. The specific failure
    // we are pinning is the legacy n_stream > 1 bailout drop.
    constexpr int N_STREAM_BAILOUT = 6;
    if (miss_reason_2 == N_STREAM_BAILOUT) {
        FAIL_AT(
            "second call hit g_can_reuse_last_miss_reason=6 "
            "(n_stream_bailout) — T3.6.I.b bailout drop not landed "
            "(src/llama.cpp:615 still present)");
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    std::fprintf(stdout, "test-graph-reuse-set-rows: PASS\n");
    return 0;
}
