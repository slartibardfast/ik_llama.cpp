// test-cy-np2-multi-step-decode.cpp
//
// CY.F.17 — probe the NP=2 multi-step decode race that survives Phase CY.F.16
// Option A's arch-force F32 reduce.
//
// What we know empirically (V4 production harness, 11 runs):
//   - Slot 0 at NP=2 ALWAYS produces the canonical NP=1-matching output.
//   - Slot 1 at NP=2 produces canonical ~60% of runs, alt-A or alt-B otherwise.
//   - With N_PREDICT=1 (single decode step), NP=2 always passes.
//   - With N_PREDICT=32 (32 decode steps), NP=2 fails ~40%.
//   - CY.F.9b unit test (single batched prefill + 1 logit read) always passes.
//   - So the race is in MULTI-STEP DECODE at NP=2.
//
// This test reproduces the failure WITHOUT the HTTP server. We:
//   1. Init a context with n_seq_max=2.
//   2. Prefill seq=0 and seq=1 in one batched call with same prompt.
//   3. Decode N steps with batched (2 tokens per step, one per seq).
//   4. Capture the full 32-token sequence per seq.
//   5. Run 5 iterations.
//   6. Report whether seq=0 and seq=1 token sequences match each other AND
//      a baseline NP=1 run.
//
// If the race manifests here, we have an HTTP-free repro.
// If it doesn't, the race is in the HTTP server's request scheduling.

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static llama_token greedy_argmax(const float * logits, int n_vocab) {
    llama_token best = 0;
    float bv = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        const float v = logits[i];
        if (v > bv) { bv = v; best = i; }
    }
    return best;
}

static std::vector<llama_token> decode_n_steps(
        llama_context * ctx,
        const std::vector<llama_token> & prompt_tokens,
        int n_predict, int n_seq_max, int np_active,
        std::vector<std::vector<llama_token>> & per_seq_out,
        bool serial_prefill = false) {

    const int n_vocab  = llama_n_vocab(llama_get_model(ctx));
    const int n_prompt = (int) prompt_tokens.size();
    per_seq_out.assign(np_active, std::vector<llama_token>());

    // Clear all seqs.
    for (int sid = 0; sid < n_seq_max; ++sid) {
        llama_kv_cache_seq_rm(ctx, (llama_seq_id) sid, -1, -1);
    }

    std::vector<llama_pos> next_pos(np_active, n_prompt);
    std::vector<llama_token> last_tok(np_active);

    if (serial_prefill) {
        // Each seq prefilled in its own llama_decode call (closer to
        // production server's per-request scheduling at low concurrency).
        for (int sid = 0; sid < np_active; ++sid) {
            llama_batch batch = llama_batch_init(n_prompt, 0, 1);
            for (int i = 0; i < n_prompt; i++) {
                common_batch_add(batch, prompt_tokens[i], i, {(llama_seq_id) sid}, i == n_prompt - 1);
            }
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "serial prefill seq=%d failed\n", sid);
                llama_batch_free(batch);
                return {};
            }
            float * logits = llama_get_logits_ith(ctx, n_prompt - 1);
            if (!logits) { llama_batch_free(batch); return {}; }
            last_tok[sid] = greedy_argmax(logits, n_vocab);
            per_seq_out[sid].push_back(last_tok[sid]);
            llama_batch_free(batch);
        }
    } else {
        // Batched prefill: np_active seqs × n_prompt tokens, interleaved seq
        // order (matches production server's cont_batching layout when both
        // requests arrive together).
        llama_batch batch = llama_batch_init(np_active * n_prompt, 0, 1);
        for (int sid = 0; sid < np_active; ++sid) {
            for (int i = 0; i < n_prompt; i++) {
                common_batch_add(batch, prompt_tokens[i], i, {(llama_seq_id) sid}, i == n_prompt - 1);
            }
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "batched prefill failed\n");
            llama_batch_free(batch);
            return {};
        }
        for (int sid = 0; sid < np_active; ++sid) {
            const int idx = (sid + 1) * n_prompt - 1;
            float * logits = llama_get_logits_ith(ctx, idx);
            if (!logits) { llama_batch_free(batch); return {}; }
            last_tok[sid] = greedy_argmax(logits, n_vocab);
            per_seq_out[sid].push_back(last_tok[sid]);
        }
        llama_batch_free(batch);
    }

    // Decode loop: one token per seq per step, batched.
    for (int step = 1; step < n_predict; ++step) {
        llama_batch dec_batch = llama_batch_init(np_active, 0, 1);
        for (int sid = 0; sid < np_active; ++sid) {
            common_batch_add(dec_batch, last_tok[sid], next_pos[sid], {(llama_seq_id) sid}, true);
            next_pos[sid]++;
        }
        if (llama_decode(ctx, dec_batch) != 0) {
            fprintf(stderr, "decode step %d failed\n", step);
            llama_batch_free(dec_batch);
            return {};
        }
        // CY.F.17: at NP=2, compare per-seq logits bit-by-bit each step.
        // First step where slot 0's logits differ from slot 1's pinpoints
        // when the singlewarp slots' state actually diverges.
        if (np_active == 2 && std::getenv("LLAMA_TEST_COMPARE_LOGITS")) {
            float * l0 = llama_get_logits_ith(ctx, 0);
            float * l1 = llama_get_logits_ith(ctx, 1);
            if (l0 && l1) {
                int diffs = 0; float maxd = 0.0f;
                for (int v = 0; v < n_vocab; ++v) {
                    uint32_t a, b;
                    std::memcpy(&a, &l0[v], 4); std::memcpy(&b, &l1[v], 4);
                    if (a != b) { ++diffs; float d=std::fabs(l0[v]-l1[v]); if (d>maxd) maxd=d; }
                }
                fprintf(stderr, "[step %2d] logits: %d/%d differ max|Δ|=%.3e\n", step, diffs, n_vocab, maxd);
            }
        }
        for (int sid = 0; sid < np_active; ++sid) {
            float * logits = llama_get_logits_ith(ctx, sid);
            if (!logits) { llama_batch_free(dec_batch); return {}; }
            last_tok[sid] = greedy_argmax(logits, n_vocab);
            per_seq_out[sid].push_back(last_tok[sid]);
        }
        llama_batch_free(dec_batch);
    }

    return last_tok;
}

int main() {
    const char * target = std::getenv("LLAMA_TEST_TARGET");
    if (!target) { fprintf(stderr, "SKIP: set LLAMA_TEST_TARGET\n"); return 77; }
    const int n_predict = std::getenv("LLAMA_TEST_N_PREDICT")
        ? std::atoi(std::getenv("LLAMA_TEST_N_PREDICT")) : 32;
    const int n_runs = std::getenv("LLAMA_TEST_N_RUNS")
        ? std::atoi(std::getenv("LLAMA_TEST_N_RUNS")) : 5;
    const char * prompt_env = std::getenv("LLAMA_TEST_PROMPT");
    // Long production-realistic prompt by default (~256 tokens of text).
    const std::string prompt = prompt_env ? prompt_env :
        "The history of artificial intelligence began in earnest with the work of Alan Turing, "
        "who in 1950 published the influential paper 'Computing Machinery and Intelligence', "
        "introducing the imitation game now widely known as the Turing test. "
        "Following Turing's pioneering ideas, the field saw rapid growth during the 1956 "
        "Dartmouth workshop organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, "
        "and Claude Shannon. McCarthy coined the term 'artificial intelligence' for the workshop. "
        "Through the 1960s and 1970s, researchers developed expert systems, theorem provers, "
        "and natural language interfaces, though hardware limitations of the era constrained "
        "the scale at which these systems could operate. Funding cycles produced two notable "
        "AI winters before deep learning, building on three decades of neural network research, "
        "transformed the field starting in the 2010s. The transformer architecture, introduced "
        "in 2017 by Vaswani et al., became the foundation for modern large language models. "
        "These models demonstrate emergent capabilities including reasoning, summarization, and";

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    mparams.split_mode   = LLAMA_SPLIT_MODE_GRAPH;
    static const char * dev_csv = "CUDA0,CUDA1";
    mparams.devices = dev_csv;
    llama_model * model = llama_model_load_from_file(target, mparams);
    if (!model) { fprintf(stderr, "load failed\n"); return 77; }

    // Tokenize.
    std::vector<llama_token> tokens;
    {
        llama_context_params tmp_cp = llama_context_default_params();
        tmp_cp.n_ctx = 4096;
        tmp_cp.n_seq_max = 1;
        llama_context * tmp_ctx = llama_init_from_model(model, tmp_cp);
        tokens = common_tokenize(tmp_ctx, prompt, true, true);
        llama_free(tmp_ctx);
    }
    const int n_prompt = (int) tokens.size();
    fprintf(stderr, "[CY.F.17] n_prompt=%d n_predict=%d n_runs=%d\n",
            n_prompt, n_predict, n_runs);

    // Run NP=1 baseline once for reference.
    fprintf(stderr, "[CY.F.17] NP=1 baseline...\n");
    std::vector<llama_token> np1_baseline;
    {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 4096 * 8;
        cp.n_batch = 2048; cp.n_ubatch = 2048;
        cp.n_seq_max = 1;
        cp.type_k = GGML_TYPE_Q4_0; cp.type_v = GGML_TYPE_Q4_0;
        cp.flash_attn = true; cp.mla_attn = 3;
        cp.k_cache_hadamard = true; cp.v_cache_hadamard = true;
        llama_context * ctx = llama_init_from_model(model, cp);
        std::vector<std::vector<llama_token>> per_seq;
        decode_n_steps(ctx, tokens, n_predict, 1, 1, per_seq);
        np1_baseline = per_seq[0];
        llama_free(ctx);
    }
    fprintf(stderr, "[CY.F.17] NP=1 baseline first 8 tokens:");
    for (int i = 0; i < 8 && i < (int)np1_baseline.size(); ++i) fprintf(stderr, " %d", np1_baseline[i]);
    fprintf(stderr, "\n");

    // ALSO check: at end of BATCHED PREFILL, do slot 0 and slot 1 produce
    // bit-identical logits? Same prompt → should be bit-identical. If not,
    // the singlewarp kernel produces different output for seq=0 and seq=1
    // CTAs in the same n_seqs=2 grid, even with identical inputs.
    {
        fprintf(stderr, "[CY.F.17] Bit-compare slot 0 vs slot 1 last-prefill-token logits...\n");
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 4096 * 8;
        cp.n_batch = 2048; cp.n_ubatch = 2048;
        cp.n_seq_max = 2;
        cp.type_k = GGML_TYPE_Q4_0; cp.type_v = GGML_TYPE_Q4_0;
        cp.flash_attn = true; cp.mla_attn = 3;
        cp.k_cache_hadamard = true; cp.v_cache_hadamard = true;
        llama_context * ctx = llama_init_from_model(model, cp);
        llama_batch batch = llama_batch_init(2 * n_prompt, 0, 1);
        for (int sid = 0; sid < 2; ++sid) {
            for (int i = 0; i < n_prompt; i++) {
                common_batch_add(batch, tokens[i], i, {(llama_seq_id) sid}, i == n_prompt - 1);
            }
        }
        if (llama_decode(ctx, batch) == 0) {
            const int idx0 = (0 + 1) * n_prompt - 1;
            const int idx1 = (1 + 1) * n_prompt - 1;
            float * l0 = llama_get_logits_ith(ctx, idx0);
            float * l1 = llama_get_logits_ith(ctx, idx1);
            if (l0 && l1) {
                const int nv = llama_n_vocab(model);
                int diffs = 0; float maxd = 0.0f;
                int idx_first = -1;
                for (int v = 0; v < nv; ++v) {
                    uint32_t a, b;
                    std::memcpy(&a, &l0[v], 4); std::memcpy(&b, &l1[v], 4);
                    if (a != b) {
                        ++diffs;
                        float d = std::fabs(l0[v]-l1[v]);
                        if (d > maxd) maxd = d;
                        if (idx_first < 0) idx_first = v;
                    }
                }
                fprintf(stderr, "[CY.F.17] slot0 vs slot1 prefill last-tok logits: %d/%d differ, max|Δ|=%.3e (first diff vocab=%d)\n",
                        diffs, nv, maxd, idx_first);
                if (diffs > 0) {
                    fprintf(stderr, "[CY.F.17] => singlewarp kernel produces different output for seq=0 vs seq=1 CTAs in same n_seqs=2 grid.\n");
                }
            }
        }
        llama_batch_free(batch);
        llama_free(ctx);
    }

    // Run NP=2 N_RUNS times.
    int slot0_match_baseline = 0;
    int slot1_match_baseline = 0;
    int slot0_eq_slot1 = 0;
    std::vector<std::vector<llama_token>> slot0_outs(n_runs);
    std::vector<std::vector<llama_token>> slot1_outs(n_runs);
    for (int run = 0; run < n_runs; ++run) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 4096 * 8;
        cp.n_batch = 2048; cp.n_ubatch = 2048;
        cp.n_seq_max = 2;
        cp.type_k = GGML_TYPE_Q4_0; cp.type_v = GGML_TYPE_Q4_0;
        cp.flash_attn = true; cp.mla_attn = 3;
        cp.k_cache_hadamard = true; cp.v_cache_hadamard = true;
        llama_context * ctx = llama_init_from_model(model, cp);
        std::vector<std::vector<llama_token>> per_seq;
        const bool serial_pref = std::getenv("LLAMA_TEST_SERIAL_PREFILL") &&
                                 std::strcmp(std::getenv("LLAMA_TEST_SERIAL_PREFILL"), "1") == 0;
        decode_n_steps(ctx, tokens, n_predict, 2, 2, per_seq, serial_pref);
        slot0_outs[run] = per_seq[0];
        slot1_outs[run] = per_seq[1];
        bool s0 = per_seq[0] == np1_baseline;
        bool s1 = per_seq[1] == np1_baseline;
        bool eq = per_seq[0] == per_seq[1];
        if (s0) slot0_match_baseline++;
        if (s1) slot1_match_baseline++;
        if (eq) slot0_eq_slot1++;
        fprintf(stderr, "[CY.F.17] run %d: slot0 %s NP=1, slot1 %s NP=1, slot0 %s slot1\n",
                run, s0?"==":"!=", s1?"==":"!=", eq?"==":"!=");
        if (!s0 || !s1) {
            int first_div_s0 = -1, first_div_s1 = -1;
            for (int i = 0; i < (int)per_seq[0].size() && i < (int)np1_baseline.size(); ++i) {
                if (first_div_s0 < 0 && per_seq[0][i] != np1_baseline[i]) first_div_s0 = i;
                if (first_div_s1 < 0 && per_seq[1][i] != np1_baseline[i]) first_div_s1 = i;
            }
            fprintf(stderr, "  first divergent step: slot0=%d slot1=%d\n", first_div_s0, first_div_s1);
            if (first_div_s0 >= 0) {
                fprintf(stderr, "  slot0[%d..%d]:", first_div_s0, std::min(first_div_s0+4, (int)per_seq[0].size()-1));
                for (int i = first_div_s0; i < first_div_s0+5 && i < (int)per_seq[0].size(); ++i) fprintf(stderr, " %d", per_seq[0][i]);
                fprintf(stderr, "\n  NP=1 [%d..%d]:", first_div_s0, std::min(first_div_s0+4, (int)np1_baseline.size()-1));
                for (int i = first_div_s0; i < first_div_s0+5 && i < (int)np1_baseline.size(); ++i) fprintf(stderr, " %d", np1_baseline[i]);
                fprintf(stderr, "\n");
            }
            if (first_div_s1 >= 0) {
                fprintf(stderr, "  slot1[%d..%d]:", first_div_s1, std::min(first_div_s1+4, (int)per_seq[1].size()-1));
                for (int i = first_div_s1; i < first_div_s1+5 && i < (int)per_seq[1].size(); ++i) fprintf(stderr, " %d", per_seq[1][i]);
                fprintf(stderr, "\n  NP=1 [%d..%d]:", first_div_s1, std::min(first_div_s1+4, (int)np1_baseline.size()-1));
                for (int i = first_div_s1; i < first_div_s1+5 && i < (int)np1_baseline.size(); ++i) fprintf(stderr, " %d", np1_baseline[i]);
                fprintf(stderr, "\n");
            }
        }
        llama_free(ctx);
    }

    fprintf(stderr, "\n[CY.F.17] Summary across %d runs:\n", n_runs);
    fprintf(stderr, "  slot0 matches NP=1: %d/%d\n", slot0_match_baseline, n_runs);
    fprintf(stderr, "  slot1 matches NP=1: %d/%d\n", slot1_match_baseline, n_runs);
    fprintf(stderr, "  slot0 == slot1:     %d/%d\n", slot0_eq_slot1, n_runs);

    llama_free_model(model);
    llama_backend_free();
    return (slot0_match_baseline == n_runs && slot1_match_baseline == n_runs) ? 0 : 1;
}
