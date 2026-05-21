// test-dflash-multi-cycle-restore-drift.cpp
//
// L3 — verify-batch vs autoregressive batch-shape invariance test.
//
// History: this test was originally written to bind multi-cycle
// save→restore drift after L2 falsified the single-cycle restore
// hypothesis (Suspect 4). The first run revealed a much sharper
// finding: a SINGLE 5-token verify decode produces DIFFERENT per-row
// argmaxes than the equivalent 1-token autoregressive decode at the
// SAME effective context. Drift across cycles is moot; the verify
// batch diverges from autoregressive at cycle 0 row 1.
//
// The test is now structured as:
//
//   Path A — spec-none autoregressive, 1-token-at-a-time. Decodes
//   N_GEN tokens at positions [P..P+N_GEN-1] using greedy argmax.
//   Captures the canonical token sequence seq_A[0..N_GEN-1].
//
//   Path B — one or more verify-batch decodes (n_tokens=verify_bs,
//   default 5) at non-overlapping position windows. For each window:
//     - Build the batch using seq_A's tokens at the window's positions.
//     - Decode, capture argmax of each row.
//     - Compare row k's argmax to seq_A's expected next token
//       (= the autoregressive prediction at position window_start+k+1).
//
// No spec-ckpt save/restore is involved in Path B. This isolates
// the batch-shape variance from the spec-ckpt machinery (L1 already
// showed the latter is neutral on a single verify decode).
//
// Gate: every row k of every verify batch must produce an argmax
// byte-identical to seq_A's token at position (window_start + k + 1).
//
// PASS interpretation: verify-batch decoding at n_tokens=5 produces
// the SAME per-row argmaxes as 1-token-at-a-time autoregressive
// decoding at the same effective context. The model's forward at
// any position is invariant to batch shape. If L3 PASSES, the bug
// in DFlash CLI is NOT batch-shape variance — it must lie in the
// drafter pipeline or in interaction with the cache update path.
//
// FAIL interpretation: verify-batch != autoregressive. The model's
// forward depends on batch shape. The CLI's degenerate output is
// (at least partly) explained by the fact that verify-batch decoding
// produces a different — possibly degenerate — sequence than
// autoregressive would. The first-divergent (window, row) is
// reported, narrowing the search to which op's batch-shape behavior
// changes (FA kernel for full-attn layers, DeltaNet kernel for
// recurrent layers, or some norm / MLP path).
//
// Mechanism check: per-cycle log dumps the verify-batch argmax
// vector vs the expected (autoregressive-derived) tokens. Tail
// inspection makes the pattern (which rows agree / disagree)
// directly visible.
//
// Env:
//   LLAMA_TEST_TARGET — target GGUF (skip 77 if unset).
//
// Optional env:
//   LLAMA_TEST_PROMPT  — override prompt.
//   LLAMA_TEST_NGEN    — number of autoregressive tokens to generate
//                        for the baseline (default 25 ⇒ 5 windows of 5).
//   LLAMA_TEST_VERIFY_BS — verify-batch width (default 5).

#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static llama_token argmax_row(const float * logits, int row, int n_vocab) {
    const float * r = logits + (size_t) row * (size_t) n_vocab;
    int best = 0;
    float best_v = r[0];
    for (int v = 1; v < n_vocab; ++v) {
        if (r[v] > best_v) { best_v = r[v]; best = v; }
    }
    return (llama_token) best;
}

static int decode_tokens(llama_context * ctx,
                         const std::vector<llama_token> & toks,
                         llama_pos start_pos,
                         llama_seq_id seq_id) {
    const int n = (int) toks.size();
    if (n == 0) return 0;
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        common_batch_add(b, toks[i], start_pos + i, {seq_id}, true);
    }
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

static llama_context * fresh_ctx(llama_model * model) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = 4096;
    cparams.n_batch     = 2048;
    cparams.n_ubatch    = 2048;
    cparams.n_seq_max   = 1;
    cparams.type_k      = GGML_TYPE_Q4_0;
    cparams.type_v      = GGML_TYPE_Q4_0;
    cparams.flash_attn  = true;
    cparams.k_cache_hadamard = true;
    cparams.v_cache_hadamard = true;
    return llama_init_from_model(model, cparams);
}

int main() {
    const char * target = std::getenv("LLAMA_TEST_TARGET");
    if (!target) {
        std::fprintf(stderr, "SKIP: set LLAMA_TEST_TARGET\n");
        return 77;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    mparams.split_mode   = LLAMA_SPLIT_MODE_GRAPH;
    static const char * dev_csv = "CUDA0,CUDA1";
    mparams.devices = dev_csv;
    llama_model * model = llama_model_load_from_file(target, mparams);
    if (!model) {
        std::fprintf(stderr, "load failed: %s\n", target);
        return 77;
    }

    const int n_vocab = llama_n_vocab(model);

    const std::string prompt = []() {
        const char * env = std::getenv("LLAMA_TEST_PROMPT");
        if (env && *env) return std::string(env);
        return std::string(
            "The capital of France is Paris. The capital of Germany is");
    }();

    const int n_gen = []() {
        const char * env = std::getenv("LLAMA_TEST_NGEN");
        if (env && *env) return std::atoi(env);
        return 25;
    }();
    const int verify_bs = []() {
        const char * env = std::getenv("LLAMA_TEST_VERIFY_BS");
        if (env && *env) return std::atoi(env);
        return 5;
    }();
    if (verify_bs < 2) {
        std::fprintf(stderr, "[L3] verify_bs must be >= 2 (got %d)\n", verify_bs);
        llama_free_model(model);
        return 1;
    }

    // Tokenise prompt.
    std::vector<llama_token> prompt_tokens;
    {
        llama_context * tmp = fresh_ctx(model);
        if (!tmp) { llama_free_model(model); std::fprintf(stderr, "tmp ctx fail\n"); return 77; }
        prompt_tokens = common_tokenize(tmp, prompt, true, true);
        llama_free(tmp);
    }
    if ((int) prompt_tokens.size() < 4) {
        std::fprintf(stderr, "[L3] prompt too short (%zu tokens)\n", prompt_tokens.size());
        llama_free_model(model);
        return 1;
    }
    const llama_pos P = (llama_pos) prompt_tokens.size();

    // Need seq_A length = n_gen+1 so that the LAST verify-batch row can
    // be checked against the autoregressive prediction at position
    // P + window_start + verify_bs.
    const int seq_A_target_len = n_gen + 1;
    std::fprintf(stderr, "[L3] prompt P = %d, seq_A target len = %d, verify_bs = %d\n",
                 (int) P, seq_A_target_len, verify_bs);

    // -------- Path A: spec-none autoregressive greedy.
    std::vector<llama_token> seq_A; seq_A.reserve(seq_A_target_len);
    {
        llama_context * ctx = fresh_ctx(model);
        if (!ctx) { llama_free_model(model); std::fprintf(stderr, "ctxA fail\n"); return 1; }
        if (decode_tokens(ctx, prompt_tokens, 0, 0) != 0) {
            std::fprintf(stderr, "[A] prefill failed\n");
            llama_free(ctx); llama_free_model(model); return 1;
        }
        llama_token id_last = argmax_row(llama_get_logits(ctx), (int)(P - 1), n_vocab);
        seq_A.push_back(id_last);  // seq_A[0] = token at position P
        for (int i = 1; i < seq_A_target_len; ++i) {
            // decode seq_A[i-1] at position P+i-1; row 0 logits predict pos P+i = seq_A[i].
            if (decode_tokens(ctx, {id_last}, P + i - 1, 0) != 0) {
                std::fprintf(stderr, "[A] decode at pos %d failed\n", (int)(P + i - 1));
                llama_free(ctx); llama_free_model(model); return 1;
            }
            id_last = argmax_row(llama_get_logits(ctx), 0, n_vocab);
            seq_A.push_back(id_last);
        }
        llama_free(ctx);
    }
    std::fprintf(stderr, "[A] seq_A length = %zu; first 10 tokens:", seq_A.size());
    for (int i = 0; i < 10 && i < (int)seq_A.size(); ++i) std::fprintf(stderr, " %d", (int)seq_A[i]);
    std::fprintf(stderr, "\n");

    // -------- Path B: one or more verify-batch decodes (no spec_ckpt).
    // Use a single fresh context that processes consecutive windows
    // back-to-back. Each window is a verify_bs-token batch supplied
    // with seq_A's tokens, so each batch sees the "canonical"
    // (autoregressive) context at every row.
    const int n_windows = n_gen / verify_bs;
    int total_mismatches = 0;
    int first_bad_window = -1;
    int first_bad_row    = -1;
    {
        llama_context * ctx = fresh_ctx(model);
        if (!ctx) { llama_free_model(model); std::fprintf(stderr, "ctxB fail\n"); return 1; }
        if (decode_tokens(ctx, prompt_tokens, 0, 0) != 0) {
            std::fprintf(stderr, "[B] prefill failed\n");
            llama_free(ctx); llama_free_model(model); return 1;
        }

        for (int w = 0; w < n_windows; ++w) {
            // Window w occupies positions [P + w*verify_bs, P + (w+1)*verify_bs).
            // Tokens in the window: seq_A[w*verify_bs .. w*verify_bs+verify_bs-1].
            // Row k's argmax should match seq_A[w*verify_bs + k + 1] (the
            // autoregressive prediction at position P + w*verify_bs + k + 1).
            const llama_pos pos_start = P + (llama_pos)(w * verify_bs);
            std::vector<llama_token> window_toks(verify_bs);
            for (int k = 0; k < verify_bs; ++k) {
                window_toks[k] = seq_A[w * verify_bs + k];
            }
            if (decode_tokens(ctx, window_toks, pos_start, 0) != 0) {
                std::fprintf(stderr, "[B w=%d] decode failed\n", w);
                llama_free(ctx); llama_free_model(model); return 1;
            }
            const float * gl = llama_get_logits(ctx);
            if (!gl) {
                std::fprintf(stderr, "[B w=%d] logits NULL\n", w);
                llama_free(ctx); llama_free_model(model); return 1;
            }

            std::fprintf(stderr, "[B w=%d pos=%d] tokens=[", w, (int) pos_start);
            for (int k = 0; k < verify_bs; ++k) std::fprintf(stderr, " %d", (int) window_toks[k]);
            std::fprintf(stderr, " ] sampled_at=[");
            int w_mismatches = 0;
            for (int k = 0; k < verify_bs; ++k) {
                const llama_token got = argmax_row(gl, k, n_vocab);
                // expected: seq_A's token at position (pos_start + k + 1)
                //   = seq_A[w*verify_bs + k + 1]
                const int exp_idx = w * verify_bs + k + 1;
                const llama_token expected = (exp_idx < (int) seq_A.size())
                    ? seq_A[exp_idx]
                    : (llama_token)(-1);
                const bool ok = (expected >= 0) && (got == expected);
                std::fprintf(stderr, " %d%s", (int) got, ok ? "" : "(!)");
                if (!ok && expected >= 0) {
                    ++w_mismatches;
                    if (first_bad_window < 0) { first_bad_window = w; first_bad_row = k; }
                }
            }
            std::fprintf(stderr, " ] expected=[");
            for (int k = 0; k < verify_bs; ++k) {
                const int exp_idx = w * verify_bs + k + 1;
                if (exp_idx < (int) seq_A.size())
                    std::fprintf(stderr, " %d", (int) seq_A[exp_idx]);
                else
                    std::fprintf(stderr, " ?");
            }
            std::fprintf(stderr, " ] mismatches=%d\n", w_mismatches);
            total_mismatches += w_mismatches;
        }
        llama_free(ctx);
    }

    if (total_mismatches == 0) {
        std::printf("[PASS] L3 verify-batch == autoregressive across %d windows × "
                    "%d tokens. Batch-shape invariance holds at production target. "
                    "The DFlash CLI bug is NOT explained by verify-batch's per-row "
                    "argmaxes diverging from autoregressive.\n",
                    n_windows, verify_bs);
        llama_free_model(model); llama_backend_free();
        return 0;
    }
    std::printf("[FAIL] L3 verify-batch DIVERGES from autoregressive: "
                "%d mismatched rows across %d windows × %d. First divergence at "
                "window %d row %d. The DFlash CLI's degenerate output is partly "
                "explained by the verify-batch decoder producing argmaxes that "
                "diverge from the autoregressive 1-token decoder at the same "
                "effective context. This is a batch-shape invariance bug in the "
                "n_tokens=%d same-slot decode path — NOT a save→restore drift.\n",
                total_mismatches, n_windows, verify_bs, first_bad_window,
                first_bad_row, verify_bs);
    llama_free_model(model); llama_backend_free();
    return 1;
}
