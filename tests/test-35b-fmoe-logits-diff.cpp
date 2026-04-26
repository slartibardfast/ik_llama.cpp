// Cross-config logits comparison for the fmoe BI drift.
//
// Decodes the same N-token batch twice — once with the fused_moe_up_gate
// path enabled, once with it disabled via cparams.fused_moe_up_gate=false
// (same code path as `-no-fmoe` at the CLI). After each decode, reads back
// the pos-0 logits slice and reports the max and average |Δ| between the
// two configs.
//
// If the delta is 0 → fmoe path is byte-identical to unfused at step 0, and
// the full-accept-drift step-4 failure must be accumulation across multiple
// decode calls (state tensors carrying drift forward).
//
// If the delta is non-zero → fmoe produces different pos-0 logits already at
// step 0, despite the op-level BI test passing. That narrows the drift site
// to somewhere inside the fused-moe op on real model inputs (weights
// distribution, non-contiguous input, router-derived ids, etc.) that the
// synthetic op test misses.

#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

// Returns a vector of n_tokens logit slices, each of length n_vocab.
// logits_all[p] is the logits at position p.
static int run_once(gpt_params params, bool enable_fmoe, int n_tokens,
                    std::vector<std::vector<float>> & logits_all, int & n_vocab_out) {
    params.fused_moe_up_gate = enable_fmoe;
    llama_init_result init = llama_init_from_gpt_params(params);
    if (!init.model || !init.context) { fprintf(stderr, "init failed\n"); return 1; }

    const std::string prompt = "The capital of France is a";
    std::vector<llama_token> toks = llama_tokenize(llama_model_get_vocab(init.model), prompt, true);
    if ((int)toks.size() < n_tokens) { fprintf(stderr, "prompt too short\n"); return 1; }

    std::vector<llama_token> use(toks.begin(), toks.begin() + n_tokens);
    llama_batch batch = llama_batch_get_one(use.data(), (int32_t)use.size(), 0, 0);
    std::vector<int8_t> logits_flags(n_tokens, 1);
    batch.logits = logits_flags.data();
    int rc = llama_decode(init.context, batch);
    if (rc != 0) { fprintf(stderr, "decode rc=%d\n", rc); llama_free(init.context); llama_free_model(init.model); return 1; }

    n_vocab_out = llama_n_vocab(init.model);
    logits_all.resize(n_tokens);
    for (int p = 0; p < n_tokens; p++) {
        float * ptr = llama_get_logits_ith(init.context, p);
        if (!ptr) { fprintf(stderr, "no logits at pos %d\n", p); llama_free(init.context); llama_free_model(init.model); return 1; }
        logits_all[p].assign(ptr, ptr + n_vocab_out);
    }

    llama_free(init.context);
    llama_free_model(init.model);
    return 0;
}

// Wrapper that grabs only pos-0.
static int run_once_pos0(gpt_params p, bool fmoe, int n_tokens, std::vector<float> & l, int & nv) {
    std::vector<std::vector<float>> all;
    int rc = run_once(p, fmoe, n_tokens, all, nv);
    if (rc != 0) return rc;
    l = std::move(all[0]);
    return 0;
}

int main(int argc, char ** argv) {
    gpt_params params;
    params.n_ctx   = 256;
    params.n_batch = 32;
    params.n_ubatch = 32;
    params.n_gpu_layers = 99;
    if (!gpt_params_parse(argc, argv, params)) return 1;
    if (params.model.empty()) { fprintf(stderr, "usage: %s -m <gguf>\n", argv[0]); return 1; }

    llama_backend_init();

    printf("\n=== N=1 ===\n");
    {
        std::vector<float> logits_on, logits_off;
        int nv = 0;
        if (run_once_pos0(params, true,  1, logits_on,  nv) != 0) return 1;
        if (run_once_pos0(params, false, 1, logits_off, nv) != 0) return 1;
        size_t diff_count = 0; double max_abs = 0.0, sum_abs = 0.0;
        for (int i = 0; i < nv; i++) {
            double d = std::fabs((double)logits_on[i] - (double)logits_off[i]);
            if (logits_on[i] != logits_off[i]) { diff_count++; sum_abs += d; if (d > max_abs) max_abs = d; }
        }
        int argmax_on = 0, argmax_off = 0;
        for (int i = 1; i < nv; i++) {
            if (logits_on[i]  > logits_on[argmax_on])   argmax_on = i;
            if (logits_off[i] > logits_off[argmax_off]) argmax_off = i;
        }
        printf("N=1  diff/n_vocab=%zu/%d  max|Δ|=%.3e  mean|Δ|=%.3e  argmax_on=%d argmax_off=%d  (flip=%d)\n",
               diff_count, nv, max_abs, diff_count ? sum_abs / diff_count : 0.0,
               argmax_on, argmax_off, argmax_on != argmax_off);
    }

    printf("\n=== N=2 ===\n");
    {
        std::vector<float> logits_on, logits_off;
        int nv = 0;
        if (run_once_pos0(params, true,  2, logits_on,  nv) != 0) return 1;
        if (run_once_pos0(params, false, 2, logits_off, nv) != 0) return 1;
        size_t diff_count = 0; double max_abs = 0.0, sum_abs = 0.0;
        for (int i = 0; i < nv; i++) {
            double d = std::fabs((double)logits_on[i] - (double)logits_off[i]);
            if (logits_on[i] != logits_off[i]) { diff_count++; sum_abs += d; if (d > max_abs) max_abs = d; }
        }
        int argmax_on = 0, argmax_off = 0;
        for (int i = 1; i < nv; i++) {
            if (logits_on[i]  > logits_on[argmax_on])   argmax_on = i;
            if (logits_off[i] > logits_off[argmax_off]) argmax_off = i;
        }
        printf("N=2  diff/n_vocab=%zu/%d  max|Δ|=%.3e  mean|Δ|=%.3e  argmax_on=%d argmax_off=%d  (flip=%d)\n",
               diff_count, nv, max_abs, diff_count ? sum_abs / diff_count : 0.0,
               argmax_on, argmax_off, argmax_on != argmax_off);
    }

    // Within-config per-position BI: for each pos 0..N-1, compare the
    // logits computed in a batched decode at batch=N against a sequential
    // decode through positions 0..pos. Matches the pos-i test's structure
    // but with full-logit deltas, not just argmax flips.
    printf("\n=== Per-position BI (logits, N=4) ===\n");
    const int BATCH_N = 4;
    for (int cfg = 0; cfg < 2; cfg++) {
        bool fmoe = (cfg == 0);
        // Batched decode produces logits at every position in one call.
        std::vector<std::vector<float>> batched;
        int nv = 0;
        if (run_once(params, fmoe, BATCH_N, batched, nv) != 0) return 1;

        // Sequential decode: run 1 token at a time, feeding each token
        // through a fresh decode, and read the logits from llama_get_logits_ith(0)
        // after each one. For pos i, we need to decode tokens 0..i and read pos i.
        std::vector<std::vector<float>> sequential(BATCH_N);
        for (int pos = 0; pos < BATCH_N; pos++) {
            gpt_params p = params;
            p.fused_moe_up_gate = fmoe;
            llama_init_result init = llama_init_from_gpt_params(p);
            if (!init.model || !init.context) { fprintf(stderr, "seq init failed\n"); return 1; }
            std::vector<llama_token> toks = llama_tokenize(llama_model_get_vocab(init.model), "The capital of France is a", true);

            // Feed tokens 0..pos-1 without reading logits (warmup).
            if (pos > 0) {
                std::vector<llama_token> pre(toks.begin(), toks.begin() + pos);
                llama_batch b = llama_batch_get_one(pre.data(), (int32_t)pre.size(), 0, 0);
                if (llama_decode(init.context, b) != 0) { fprintf(stderr, "warmup rc bad\n"); return 1; }
            }
            // Decode token at position `pos` alone and read its logits.
            std::vector<llama_token> one = { toks[pos] };
            llama_batch b = llama_batch_get_one(one.data(), 1, (int32_t)pos, 0);
            if (llama_decode(init.context, b) != 0) { fprintf(stderr, "seq decode pos=%d rc bad\n", pos); return 1; }
            float * p0 = llama_get_logits_ith(init.context, 0);
            sequential[pos].assign(p0, p0 + nv);
            llama_free(init.context);
            llama_free_model(init.model);
        }

        printf("fmoe=%s\n", fmoe ? "ON " : "OFF");
        for (int pos = 0; pos < BATCH_N; pos++) {
            size_t diff_count = 0; double max_abs = 0.0;
            for (int i = 0; i < nv; i++) {
                if (batched[pos][i] != sequential[pos][i]) {
                    diff_count++;
                    double d = std::fabs((double)batched[pos][i] - (double)sequential[pos][i]);
                    if (d > max_abs) max_abs = d;
                }
            }
            int argmax_b = 0, argmax_s = 0;
            for (int i = 1; i < nv; i++) {
                if (batched[pos][i]    > batched[pos][argmax_b])    argmax_b = i;
                if (sequential[pos][i] > sequential[pos][argmax_s]) argmax_s = i;
            }
            printf("  pos=%d  diff=%zu/%d  max|Δ|=%.3e  argmax_batch=%d argmax_seq=%d  (flip=%d)\n",
                   pos, diff_count, nv, max_abs,
                   argmax_b, argmax_s, argmax_b != argmax_s);
        }
    }

    llama_backend_free();
    return 0;
}
