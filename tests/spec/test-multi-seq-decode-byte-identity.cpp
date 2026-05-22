// test-multi-seq-decode-byte-identity.cpp
//
// Binding test for PHASE_NSTREAM_KV_PERF T3.3 + T3.4 — unified-stream
// dispatch. Drives llama_decode with a contiguous-per-seq multi-seq
// batch (two sequences, identical prompts) and verifies that the
// returned per-seq logits are byte-identical to two serial single-seq
// decodes at the same starting state.
//
// Binding contract (specs/dispatch/unified_stream_dispatch.allium):
//
//   UnifiedUbatchInvariant     — one llama_decode per tick spans
//                                 N streams via ne[3].
//   UniformShapePerTick        — within the (single) decode, per-stream
//                                 token count is uniform.
//   PreservesBugCAbsence       — mul_mat sees one uniform shape per call.
//   BugCAbsenceByConstruction  — equivalent to N serial single-seq
//                                 decodes at the same starting state.
//
// This test specifically binds the LAST contract: multi-seq output ==
// per-seq serial output, byte-for-byte. PASS expected after T3.3 + T3.4
// land; FAIL on HEAD pre-T3.4 (nstream_demands_subbatch splits the
// multi-seq batch into single-seq runs upstream of the build, so the
// unified path never runs — the test as written would still pass
// because the upstream split already produces the same result as
// serial decoding, but T3.4's gate drop is what actually exercises
// the new 4D build path).
//
// Returns: 0 = PASS, 1 = FAIL, 77 = SKIP (no model path supplied).
//
// Usage:
//   ./test-multi-seq-decode-byte-identity MODEL_PATH [N_TOK]
// Defaults: N_TOK = 4.
//
// Model requirements: --fa on path (PSKV predicate matches), q4_0 KV
// cache, single-GPU or multi-GPU.

#include "llama.h"
#include "common.h"

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
    int n_tok = 4;
};

Args parse_args(int argc, char** argv) {
    Args a;
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL_PATH [N_TOK]\n", argv[0]);
        std::exit(77);
    }
    a.model_path = argv[1];
    if (argc >= 3) a.n_tok = std::atoi(argv[2]);
    if (a.n_tok < 1) a.n_tok = 4;
    return a;
}

// Decode tokens sequentially for one seq_id; return logits of the LAST
// token (n_vocab floats).
std::vector<float> decode_serial(llama_context * ctx,
                                 const std::vector<llama_token> & tokens,
                                 llama_seq_id seq) {
    std::fprintf(stdout, "[decode_serial] enter seq=%d n=%zu\n", seq, tokens.size());
    std::fflush(stdout);
    llama_kv_cache_seq_rm(ctx, seq, -1, -1);
    std::fprintf(stdout, "[decode_serial] seq_rm OK seq=%d\n", seq);
    std::fflush(stdout);
    const llama_model * m = llama_get_model(ctx);
    std::fprintf(stdout, "[decode_serial] get_model %p\n", (void*)m);
    std::fflush(stdout);
    const int n_vocab = llama_n_vocab(m);
    std::fprintf(stdout, "[decode_serial] n_vocab=%d\n", n_vocab);
    std::fflush(stdout);
    for (size_t i = 0; i < tokens.size(); ++i) {
        llama_token         tok       = tokens[i];
        llama_pos           pos       = (llama_pos)i;
        int32_t             n_seq_id  = 1;
        llama_seq_id        seq_buf   = seq;
        llama_seq_id *      seq_ptr   = &seq_buf;
        int8_t              logits_req = (i == tokens.size() - 1) ? 1 : 0;
        llama_batch b = {};
        b.n_tokens   = 1;
        b.token      = &tok;
        b.pos        = &pos;
        b.n_seq_id   = &n_seq_id;
        b.seq_id     = &seq_ptr;
        b.logits     = &logits_req;
        b.all_pos_0  = 0;
        b.all_pos_1  = 0;
        b.all_seq_id = 0;
        std::fprintf(stdout, "[decode_serial] about to decode seq=%d i=%zu tok=%d pos=%d logits=%d\n",
                     seq, i, tok, (int)pos, (int)logits_req);
        std::fflush(stdout);
        const int rc = llama_decode(ctx, b);
        std::fprintf(stdout, "[decode_serial] decode returned rc=%d seq=%d i=%zu\n", rc, seq, i);
        std::fflush(stdout);
        if (rc != 0) FAIL_AT("serial decode rc=%d at seq=%d token %zu", rc, seq, i);
    }
    std::fprintf(stdout, "[decode_serial] loop done seq=%d, about to get_logits_ith(0)\n", seq);
    std::fflush(stdout);
    float * lp = llama_get_logits_ith(ctx, 0);
    std::fprintf(stdout, "[decode_serial] get_logits_ith returned %p\n", (void*)lp);
    std::fflush(stdout);
    if (!lp) FAIL_AT("llama_get_logits_ith returned null for seq=%d", seq);
    return std::vector<float>(lp, lp + n_vocab);
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

    // Tokenize a short prompt deterministically.
    std::fprintf(stdout, "about to tokenize\n");
    std::fflush(stdout);
    const char * text = "Hello world from a deterministic test.";
    std::vector<llama_token> all_toks(64);
    int n = llama_tokenize(model, text, (int)strlen(text),
                           all_toks.data(), (int)all_toks.size(), true, false);
    std::fprintf(stdout, "tokenize returned n=%d\n", n);
    std::fflush(stdout);
    if (n <= 0) FAIL_AT("tokenize returned %d", n);
    if (n > args.n_tok) n = args.n_tok;
    std::vector<llama_token> tokens(all_toks.begin(), all_toks.begin() + n);
    std::fprintf(stdout, "prompt tokens (n=%d):", n);
    for (auto t : tokens) std::fprintf(stdout, " %d", t);
    std::fprintf(stdout, "\n");

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx     = 256;
    const char * nsm_env = std::getenv("TEST_N_SEQ_MAX");
    ctx_params.n_seq_max = nsm_env ? std::atoi(nsm_env) : 2;
    ctx_params.n_batch   = 64;
    ctx_params.n_ubatch  = 64;
    ctx_params.flash_attn = true;
    ctx_params.type_k    = GGML_TYPE_Q4_0;
    ctx_params.type_v    = GGML_TYPE_Q4_0;
    std::fprintf(stdout, "[ctx_params] n_seq_max=%u type_k=q4_0 type_v=q4_0\n",
                 ctx_params.n_seq_max);
    std::fflush(stdout);

    std::fprintf(stdout, "about to init context\n");
    std::fflush(stdout);
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        llama_free_model(model);
        FAIL_AT("failed to allocate llama_context");
    }
    std::fprintf(stdout, "context init OK\n");
    std::fflush(stdout);

    const int n_vocab = llama_n_vocab(model);

    // Serial decodes for each seq, capture last-token logits.
    auto logits_serial_0 = decode_serial(ctx, tokens, /*seq=*/0);
    auto logits_serial_1 = decode_serial(ctx, tokens, /*seq=*/1);

    // Sanity check: the prompt is identical, the model is deterministic,
    // and each seq sees its own clean stream → logits should be byte-identical.
    if (std::memcmp(logits_serial_0.data(), logits_serial_1.data(),
                    (size_t)n_vocab * sizeof(float)) != 0) {
        FAIL_AT("serial seq-0 vs seq-1 logits differ — model/cache non-determinism in serial baseline");
    }
    std::fprintf(stdout, "serial baseline: seq-0 == seq-1 byte-identical (%d floats)\n", n_vocab);

    // Unified multi-seq decode: contiguous-per-seq batch
    //   [seq=0 tok 0..n-1, seq=1 tok 0..n-1]
    llama_kv_cache_seq_rm(ctx, 0, -1, -1);
    llama_kv_cache_seq_rm(ctx, 1, -1, -1);

    const int multi_n = 2 * n;
    std::vector<llama_token>    multi_tokens(multi_n);
    std::vector<llama_pos>      multi_pos(multi_n);
    std::vector<int32_t>        multi_n_seq(multi_n, 1);
    std::vector<llama_seq_id>   multi_seq_buf(multi_n);
    std::vector<llama_seq_id *> multi_seq_ptr(multi_n);
    std::vector<int8_t>         multi_logits_req(multi_n, 0);
    for (int s = 0; s < 2; ++s) {
        for (int t = 0; t < n; ++t) {
            int i = s * n + t;
            multi_tokens[i]    = tokens[t];
            multi_pos[i]       = t;
            multi_seq_buf[i]   = (llama_seq_id)s;
            multi_seq_ptr[i]   = &multi_seq_buf[i];
            if (t == n - 1) multi_logits_req[i] = 1;
        }
    }

    llama_batch multi = {};
    multi.n_tokens   = multi_n;
    multi.token      = multi_tokens.data();
    multi.pos        = multi_pos.data();
    multi.n_seq_id   = multi_n_seq.data();
    multi.seq_id     = multi_seq_ptr.data();
    multi.logits     = multi_logits_req.data();
    multi.all_pos_0  = 0;
    multi.all_pos_1  = 0;
    multi.all_seq_id = 0;

    const int rc = llama_decode(ctx, multi);
    if (rc != 0) FAIL_AT("multi-seq decode rc=%d", rc);

    // Read back per-seq last-token logits. ith=0 corresponds to the
    // first logits-requesting token (seq 0's last); ith=1 to the second
    // (seq 1's last).
    float * mlogits_0 = llama_get_logits_ith(ctx, 0);
    float * mlogits_1 = llama_get_logits_ith(ctx, 1);
    if (!mlogits_0 || !mlogits_1) FAIL_AT("multi-seq logits null");

    bool ok_0 = std::memcmp(mlogits_0, logits_serial_0.data(),
                            (size_t)n_vocab * sizeof(float)) == 0;
    bool ok_1 = std::memcmp(mlogits_1, logits_serial_1.data(),
                            (size_t)n_vocab * sizeof(float)) == 0;
    if (!ok_0) {
        // Report first divergent index.
        for (int i = 0; i < n_vocab; ++i) {
            if (mlogits_0[i] != logits_serial_0[i]) {
                std::fprintf(stderr,
                    "FAIL: multi-seq seq-0 logits[%d] = %.9g, serial = %.9g (Δ=%.3e)\n",
                    i, mlogits_0[i], logits_serial_0[i],
                    mlogits_0[i] - logits_serial_0[i]);
                break;
            }
        }
        FAIL_AT("multi-seq seq-0 logits NOT byte-identical to serial seq-0");
    }
    if (!ok_1) {
        for (int i = 0; i < n_vocab; ++i) {
            if (mlogits_1[i] != logits_serial_1[i]) {
                std::fprintf(stderr,
                    "FAIL: multi-seq seq-1 logits[%d] = %.9g, serial = %.9g (Δ=%.3e)\n",
                    i, mlogits_1[i], logits_serial_1[i],
                    mlogits_1[i] - logits_serial_1[i]);
                break;
            }
        }
        FAIL_AT("multi-seq seq-1 logits NOT byte-identical to serial seq-1");
    }

    std::fprintf(stdout,
        "multi-seq logits byte-identical to serial per-seq logits (n_vocab=%d, n_tok=%d)\n",
        n_vocab, n);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    std::fprintf(stdout, "test-multi-seq-decode-byte-identity: PASS\n");
    return 0;
}
