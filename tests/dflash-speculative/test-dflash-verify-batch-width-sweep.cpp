// test-dflash-verify-batch-width-sweep.cpp
//
// L3' — verify-batch width sweep, localises the n_tokens threshold at
// which batch-shape variance first appears on the production target.
//
// Background: L3 bound a real divergence between a 5-token verify
// decode and 1-token-at-a-time autoregressive at the same effective
// context (PHASE_NSTREAM_KV_PERF.md "L3 result"). This test sweeps
// verify_bs ∈ {1, 2, 3, 4, 5, 6, 8} and reports per-width mismatch
// counts. The smallest width with mismatches localises the bug:
//
//   - verify_bs=1 (control): single-token decode == autoregressive
//     by construction. Should PASS.
//   - verify_bs=2 mismatches: bug is in any same-slot multi-token
//     dispatch (most ops in the model have a > 1 token branch
//     somewhere — FA, conv1d, norm reduction tile size, etc.).
//   - verify_bs ≥ N mismatches but not below: bug is shape-specific
//     dispatch at threshold N (e.g. FA tile-size decision at
//     n_tokens crossing a power-of-2 boundary).
//
// Setup mirrors L3: same model, same prompt, same cparams. Path A
// runs autoregressive to populate seq_A. Path B then runs verify
// decodes at each width using seq_A's tokens, comparing per-row
// argmax to seq_A's expected continuation. No spec_ckpt anywhere.
//
// Env:
//   LLAMA_TEST_TARGET — target GGUF (skip 77 if unset).
//
// Optional env:
//   LLAMA_TEST_PROMPT — override prompt.
//   LLAMA_TEST_WIDTHS — comma-separated list of widths (default
//                       "1,2,3,4,5,6,8").
//
// Gate: report per-width PASS/FAIL. Test exit code is 0 iff verify_bs=1
// (the autoregressive control) PASSES. Width-> mismatch results are
// always printed regardless.

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

static std::vector<int> parse_widths_csv(const char * csv) {
    std::vector<int> out;
    if (!csv || !*csv) {
        out = {1, 2, 3, 4, 5, 6, 8};
        return out;
    }
    const char * p = csv;
    while (*p) {
        char * end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p) break;
        if (v >= 1) out.push_back((int) v);
        p = end;
        while (*p == ',' || *p == ' ') ++p;
    }
    if (out.empty()) out = {1, 2, 3, 4, 5, 6, 8};
    return out;
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
    const std::vector<int> widths = parse_widths_csv(std::getenv("LLAMA_TEST_WIDTHS"));
    int max_width = 0;
    for (int w : widths) if (w > max_width) max_width = w;

    // We need seq_A of length max_width + 1 to verify the largest width.
    const int seq_A_len = max_width + 1;

    std::vector<llama_token> prompt_tokens;
    {
        llama_context * tmp = fresh_ctx(model);
        if (!tmp) { llama_free_model(model); std::fprintf(stderr, "tmp ctx fail\n"); return 77; }
        prompt_tokens = common_tokenize(tmp, prompt, true, true);
        llama_free(tmp);
    }
    if ((int) prompt_tokens.size() < 4) {
        std::fprintf(stderr, "[L3'] prompt too short (%zu tokens)\n", prompt_tokens.size());
        llama_free_model(model);
        return 1;
    }
    const llama_pos P = (llama_pos) prompt_tokens.size();
    std::fprintf(stderr, "[L3'] prompt P=%d, widths=[", (int) P);
    for (int w : widths) std::fprintf(stderr, " %d", w);
    std::fprintf(stderr, " ], seq_A_len=%d\n", seq_A_len);

    // Path A: autoregressive baseline.
    std::vector<llama_token> seq_A; seq_A.reserve(seq_A_len);
    {
        llama_context * ctx = fresh_ctx(model);
        if (!ctx) { llama_free_model(model); std::fprintf(stderr, "ctxA fail\n"); return 1; }
        if (decode_tokens(ctx, prompt_tokens, 0, 0) != 0) {
            std::fprintf(stderr, "[A] prefill failed\n");
            llama_free(ctx); llama_free_model(model); return 1;
        }
        llama_token id_last = argmax_row(llama_get_logits(ctx), (int)(P - 1), n_vocab);
        seq_A.push_back(id_last);
        for (int i = 1; i < seq_A_len; ++i) {
            if (decode_tokens(ctx, {id_last}, P + i - 1, 0) != 0) {
                std::fprintf(stderr, "[A] decode at pos %d failed\n", (int)(P + i - 1));
                llama_free(ctx); llama_free_model(model); return 1;
            }
            id_last = argmax_row(llama_get_logits(ctx), 0, n_vocab);
            seq_A.push_back(id_last);
        }
        llama_free(ctx);
    }
    std::fprintf(stderr, "[A] seq_A:");
    for (int i = 0; i < seq_A_len; ++i) std::fprintf(stderr, " %d", (int) seq_A[i]);
    std::fprintf(stderr, "\n");

    // Path B: for each verify_bs, fresh context, run ONE batch.
    int control_failures = 0;
    std::vector<std::pair<int,int>> results;  // {width, mismatches}
    for (int w : widths) {
        if (w > max_width || w > seq_A_len - 1) {
            std::fprintf(stderr, "[B w=%d] SKIP — width exceeds seq_A range\n", w);
            results.emplace_back(w, -1);
            continue;
        }
        llama_context * ctx = fresh_ctx(model);
        if (!ctx) { llama_free_model(model); std::fprintf(stderr, "ctxB%d fail\n", w); return 1; }
        if (decode_tokens(ctx, prompt_tokens, 0, 0) != 0) {
            std::fprintf(stderr, "[B w=%d] prefill failed\n", w);
            llama_free(ctx); llama_free_model(model); return 1;
        }
        std::vector<llama_token> batch_toks(seq_A.begin(), seq_A.begin() + w);
        if (decode_tokens(ctx, batch_toks, P, 0) != 0) {
            std::fprintf(stderr, "[B w=%d] decode failed\n", w);
            llama_free(ctx); llama_free_model(model); return 1;
        }
        const float * gl = llama_get_logits(ctx);
        if (!gl) {
            std::fprintf(stderr, "[B w=%d] logits NULL\n", w);
            llama_free(ctx); llama_free_model(model); return 1;
        }

        int n_mismatch = 0;
        std::fprintf(stderr, "[B w=%d] tokens=[", w);
        for (int k = 0; k < w; ++k) std::fprintf(stderr, " %d", (int) batch_toks[k]);
        std::fprintf(stderr, " ] sampled=[");
        for (int k = 0; k < w; ++k) {
            const llama_token got = argmax_row(gl, k, n_vocab);
            const llama_token exp = seq_A[k + 1];  // pos P+k+1
            const bool ok = (got == exp);
            std::fprintf(stderr, " %d%s", (int) got, ok ? "" : "(!)");
            if (!ok) ++n_mismatch;
        }
        std::fprintf(stderr, " ] expected=[");
        for (int k = 0; k < w; ++k) std::fprintf(stderr, " %d", (int) seq_A[k + 1]);
        std::fprintf(stderr, " ] mismatches=%d/%d\n", n_mismatch, w);
        results.emplace_back(w, n_mismatch);
        if (w == 1 && n_mismatch != 0) ++control_failures;
        llama_free(ctx);
    }

    // Summary table.
    std::printf("\n=== L3' verify-batch width sweep ===\n");
    std::printf("verify_bs : mismatches / total\n");
    int first_failing_width = -1;
    for (auto & r : results) {
        if (r.second < 0) {
            std::printf("  bs=%d    : SKIP\n", r.first);
        } else {
            std::printf("  bs=%d    : %d / %d %s\n",
                        r.first, r.second, r.first,
                        r.second == 0 ? "PASS" : "FAIL");
            if (r.second > 0 && first_failing_width < 0) {
                first_failing_width = r.first;
            }
        }
    }
    if (first_failing_width >= 0) {
        std::printf("First failing width = %d. The bug surface is whatever code "
                    "path activates at n_tokens=%d but not below.\n",
                    first_failing_width, first_failing_width);
    } else {
        std::printf("All widths PASS — batch-shape invariance holds.\n");
    }

    llama_free_model(model);
    llama_backend_free();
    return control_failures == 0 ? 0 : 1;
}
