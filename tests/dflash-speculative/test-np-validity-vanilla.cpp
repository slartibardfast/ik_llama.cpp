// test-np-validity-vanilla.cpp
//
// PHASE_DFLASH.md T9 (revised) — validity lockdown of vanilla
// (--spec none) at np > 1. Gates the DFlash multi-slot API extension
// work; if vanilla fails validity here, DFlash multi-slot is moot.
//
// What "validity" means (5 falsifiable per-slot asserts):
//   1. All n_gen tokens emitted (no early break / decode failure).
//   2. PPL of output under target ∈ [1.0, 50.0] — production-coherence
//      band (greedy baseline ~1, modest drift ~10, pathological >50).
//   3. ≥ 95% of emitted tokens in-vocab (catches glitch-token cascades).
//   4. llama_decode returned 0 on every generation step.
//   5. No NaN / Inf in captured logits at any decode step.
//
// Cross-slot byte-identity is NOT a pass requirement; the prior MTP
// multi-slot investigation (project_mtp_multislot_determinism…)
// established batch-shape divergence exists on this hybrid stack. We
// test per-slot validity, not identity.
//
// Env:
//   LLAMA_TEST_TARGET     — target GGUF (required)
//   LLAMA_TEST_PROMPT_DIR — dir with p0.txt..p{NP-1}.txt (required)
//   LLAMA_TEST_NP         — slot count (required; e.g. 2, 4, 8)
//   LLAMA_TEST_N_GEN      — tokens per slot (default 64)
//   LLAMA_TEST_JSON       — output path (optional; if set, writes structured JSON)
//
// Exit:
//   0  = all slots pass all 5 asserts
//   1  = at least one slot failed (details on stderr + JSON if set)
//   77 = SKIP (missing env)

#include "common.h"
#include "perplexity.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct slot_state {
    int                       id;
    std::string               prompt_path;
    std::vector<llama_token>  prompt_tokens;
    std::vector<llama_token>  generated;
    llama_pos                 next_pos = 0;
    llama_token               id_last  = 0;
    // assert flags
    bool decode_ok        = true;
    bool no_nan_inf       = true;
    int  in_vocab_count   = 0;
    bool terminated       = false;
    double ppl_of_output  = 0.0;
    bool   ppl_ok         = false;
    bool   in_vocab_ok    = false;
};

static llama_token greedy_argmax(const float * logits, int n_vocab, bool & no_nan_inf) {
    llama_token best = 0;
    float bv = logits[0];
    if (!std::isfinite(bv)) no_nan_inf = false;
    for (int i = 1; i < n_vocab; i++) {
        const float v = logits[i];
        if (!std::isfinite(v)) { no_nan_inf = false; }
        if (v > bv) { bv = v; best = i; }
    }
    return best;
}

static double compute_per_slot_ppl(llama_context * ctx,
                                   const std::vector<llama_token> & prompt_tokens,
                                   const std::vector<llama_token> & generated,
                                   llama_seq_id seq_id) {
    const int n_prompt = (int) prompt_tokens.size();
    const int n_gen    = (int) generated.size();
    if (n_prompt < 1 || n_gen < 1) return 0.0;
    const llama_model * model = llama_get_model(ctx);
    const int n_vocab = llama_n_vocab(model);

    // Isolate the PPL pass: clear KV, re-decode this slot's full sequence
    // alone (no batch-shape interference from other slots).
    llama_kv_cache_seq_rm(ctx, seq_id, 0, -1);

    std::vector<llama_token> all_tokens = prompt_tokens;
    all_tokens.insert(all_tokens.end(), generated.begin(), generated.end());
    const int n_total = (int) all_tokens.size();

    std::vector<float> logits;
    logits.reserve((size_t) n_gen * (size_t) n_vocab);

    const int n_batch_cap = (int) llama_n_batch(ctx);
    int n_processed = 0;
    while (n_processed < n_total) {
        int n_tokens = std::min(n_batch_cap, n_total - n_processed);
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (int i = 0; i < n_tokens; i++) {
            int pos = n_processed + i;
            bool need = (pos >= n_prompt - 1 && pos < n_prompt + n_gen - 1);
            common_batch_add(batch, all_tokens[pos], pos, {seq_id}, need);
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return 0.0;
        }
        for (int i = 0; i < n_tokens; i++) {
            int pos = n_processed + i;
            if (pos >= n_prompt - 1 && pos < n_prompt + n_gen - 1) {
                float * row = llama_get_logits_ith(ctx, i);
                if (!row) { llama_batch_free(batch); return 0.0; }
                logits.insert(logits.end(), row, row + n_vocab);
            }
        }
        llama_batch_free(batch);
        n_processed += n_tokens;
    }
    if ((int) logits.size() != n_gen * n_vocab) return 0.0;

    std::vector<llama_token> seq(n_gen + 1);
    seq[0] = prompt_tokens.back();
    for (int i = 0; i < n_gen; i++) seq[i + 1] = generated[i];
    double nll = 0.0, nll2 = 0.0;
    std::vector<float> lh(n_gen), ph(n_gen);
    int n_workers = std::max(1, (int) std::thread::hardware_concurrency() - 1);
    std::vector<std::thread> workers(n_workers);
    process_logits(n_vocab, logits.data(), seq.data(), n_gen,
                   workers, nll, nll2, lh.data(), ph.data());
    return std::exp(nll / (double) n_gen);
}

int main() {
    const char * target  = std::getenv("LLAMA_TEST_TARGET");
    const char * pd      = std::getenv("LLAMA_TEST_PROMPT_DIR");
    const char * np_env  = std::getenv("LLAMA_TEST_NP");
    if (!target || !pd || !np_env) {
        fprintf(stderr, "SKIP: set LLAMA_TEST_TARGET, LLAMA_TEST_PROMPT_DIR, LLAMA_TEST_NP\n");
        return 77;
    }
    const int N = std::atoi(np_env);
    const int n_gen = std::getenv("LLAMA_TEST_N_GEN") ? std::atoi(std::getenv("LLAMA_TEST_N_GEN")) : 64;
    const char * json_out = std::getenv("LLAMA_TEST_JSON");
    if (N < 2 || N > 16) { fprintf(stderr, "LLAMA_TEST_NP must be in [2,16]\n"); return 1; }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    mparams.split_mode   = LLAMA_SPLIT_MODE_GRAPH;
    static const char * dev_csv = "CUDA0,CUDA1";
    mparams.devices = dev_csv;
    llama_model * model = llama_model_load_from_file(target, mparams);
    if (!model) { fprintf(stderr, "load failed: %s\n", target); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = 4096;
    cparams.n_batch     = 2048;
    cparams.n_ubatch    = 512;
    cparams.n_seq_max   = (uint32_t) N;
    cparams.type_k      = GGML_TYPE_Q4_0;
    cparams.type_v      = GGML_TYPE_Q4_0;
    cparams.flash_attn  = true;
    cparams.mla_attn    = 3;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { llama_free_model(model); fprintf(stderr, "ctx init failed\n"); return 1; }

    const int n_vocab = llama_n_vocab(model);

    // Tokenise prompts.
    std::vector<slot_state> slots(N);
    for (int s = 0; s < N; s++) {
        slots[s].id = s;
        char path[512];
        snprintf(path, sizeof(path), "%s/p%d.txt", pd, s);
        slots[s].prompt_path = path;
        std::ifstream f(path); std::stringstream ss; ss << f.rdbuf();
        slots[s].prompt_tokens = common_tokenize(ctx, ss.str(), true, true);
        if (slots[s].prompt_tokens.empty()) {
            fprintf(stderr, "slot %d: empty prompt at %s\n", s, path);
            llama_free(ctx); llama_free_model(model); return 1;
        }
    }

    // Per-slot prefill (sequential, distinct seq_id). Mirrors server's
    // arrival-time prefill pattern.
    fprintf(stderr, "[T9-vanilla] N=%d n_gen=%d — per-slot prefill\n", N, n_gen);
    for (int s = 0; s < N; s++) {
        const int n_prompt = (int) slots[s].prompt_tokens.size();
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++) {
            bool last = (i == n_prompt - 1);
            common_batch_add(batch, slots[s].prompt_tokens[i], i, {(llama_seq_id) s}, last);
        }
        if (llama_decode(ctx, batch) != 0) {
            slots[s].decode_ok = false;
            fprintf(stderr, "slot %d: prefill decode FAILED\n", s);
            llama_batch_free(batch);
            continue;
        }
        float * logits = llama_get_logits_ith(ctx, n_prompt - 1);
        if (!logits) { slots[s].decode_ok = false; llama_batch_free(batch); continue; }
        slots[s].id_last = greedy_argmax(logits, n_vocab, slots[s].no_nan_inf);
        slots[s].next_pos = (llama_pos) n_prompt;
        llama_batch_free(batch);
    }

    // Concurrent generation: each step is one batch of N tokens (one per
    // alive slot at its current pos with its own seq_id). This is the
    // np > 1 path the prior MTP investigation hit batch-shape bugs on.
    fprintf(stderr, "[T9-vanilla] generation loop (concurrent batches of N=%d)\n", N);
    for (int step = 0; step < n_gen; step++) {
        llama_batch batch = llama_batch_init(N, 0, 1);
        std::vector<int> active_slots;
        active_slots.reserve(N);
        for (int s = 0; s < N; s++) {
            if (!slots[s].decode_ok) continue;
            common_batch_add(batch, slots[s].id_last, slots[s].next_pos, {(llama_seq_id) s}, true);
            active_slots.push_back(s);
        }
        if (batch.n_tokens == 0) break;
        if (llama_decode(ctx, batch) != 0) {
            for (int s : active_slots) slots[s].decode_ok = false;
            fprintf(stderr, "[T9-vanilla] step %d: llama_decode FAILED for all active slots\n", step);
            llama_batch_free(batch);
            break;
        }
        for (size_t k = 0; k < active_slots.size(); k++) {
            int s = active_slots[k];
            float * logits = llama_get_logits_ith(ctx, (int32_t) k);
            if (!logits) {
                slots[s].decode_ok = false;
                continue;
            }
            llama_token tok = greedy_argmax(logits, n_vocab, slots[s].no_nan_inf);
            if (tok >= 0 && tok < n_vocab) slots[s].in_vocab_count++;
            slots[s].generated.push_back(tok);
            slots[s].id_last = tok;
            slots[s].next_pos += 1;
        }
        llama_batch_free(batch);
    }
    for (auto & sl : slots) sl.terminated = ((int) sl.generated.size() == n_gen);

    // Per-slot PPL pass. Clears KV between slots so each PPL is isolated
    // from cross-slot batch-shape effects in the eval pass itself.
    fprintf(stderr, "[T9-vanilla] per-slot PPL pass\n");
    for (auto & sl : slots) {
        if (sl.generated.empty()) continue;
        sl.ppl_of_output = compute_per_slot_ppl(ctx, sl.prompt_tokens, sl.generated, (llama_seq_id) sl.id);
        sl.ppl_ok = std::isfinite(sl.ppl_of_output) && sl.ppl_of_output >= 1.0 && sl.ppl_of_output <= 50.0;
        sl.in_vocab_ok = (sl.in_vocab_count >= (int) (0.95 * (double) sl.generated.size()));
    }

    // Assert + report.
    int n_fail = 0;
    for (auto & sl : slots) {
        bool pass = sl.terminated && sl.ppl_ok && sl.in_vocab_ok && sl.decode_ok && sl.no_nan_inf;
        if (!pass) n_fail++;
        fprintf(stderr,
            "slot %d: %s | term=%d decode=%d nonan=%d invocab=%d/%zu (%s) ppl=%.4f (%s)\n",
            sl.id, pass ? "PASS" : "FAIL",
            (int) sl.terminated, (int) sl.decode_ok, (int) sl.no_nan_inf,
            sl.in_vocab_count, sl.generated.size(),
            sl.in_vocab_ok ? "ok" : "FAIL",
            sl.ppl_of_output, sl.ppl_ok ? "ok" : "FAIL");
    }

    if (json_out) {
        FILE * f = std::fopen(json_out, "w");
        if (f) {
            fprintf(f, "{\n  \"np\": %d,\n  \"n_gen\": %d,\n  \"spec\": \"none\",\n  \"slots\": [\n", N, n_gen);
            for (int s = 0; s < N; s++) {
                auto & sl = slots[s];
                fprintf(f,
                    "    {\"slot\": %d, \"prompt\": \"%s\", "
                    "\"terminated\": %s, \"decode_ok\": %s, \"no_nan_inf\": %s, "
                    "\"in_vocab\": %d, \"n_gen\": %zu, \"in_vocab_ok\": %s, "
                    "\"ppl\": %.6f, \"ppl_ok\": %s, "
                    "\"first_tokens\": [%d, %d, %d, %d, %d]}%s\n",
                    sl.id, sl.prompt_path.c_str(),
                    sl.terminated ? "true" : "false",
                    sl.decode_ok  ? "true" : "false",
                    sl.no_nan_inf ? "true" : "false",
                    sl.in_vocab_count, sl.generated.size(),
                    sl.in_vocab_ok ? "true" : "false",
                    sl.ppl_of_output, sl.ppl_ok ? "true" : "false",
                    sl.generated.size() > 0 ? (int) sl.generated[0] : -1,
                    sl.generated.size() > 1 ? (int) sl.generated[1] : -1,
                    sl.generated.size() > 2 ? (int) sl.generated[2] : -1,
                    sl.generated.size() > 3 ? (int) sl.generated[3] : -1,
                    sl.generated.size() > 4 ? (int) sl.generated[4] : -1,
                    s + 1 < N ? "," : "");
            }
            fprintf(f, "  ],\n  \"n_fail\": %d\n}\n", n_fail);
            std::fclose(f);
            fprintf(stderr, "wrote %s\n", json_out);
        }
    }

    fprintf(stderr, "[T9-vanilla] N=%d: %d/%d slots PASS\n", N, N - n_fail, N);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return n_fail ? 1 : 0;
}
