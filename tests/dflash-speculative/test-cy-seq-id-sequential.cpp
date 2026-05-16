// test-cy-seq-id-sequential.cpp
//
// Phase CY.F.9 — umbrella driver for the slot-id > 0 addressing question.
//
// On a context with n_seq_max=N, prefills seq_id=0 alone, captures next-token
// logits + argmax. Then prefills seq_id=K (K ∈ {1..N-1}) alone with the SAME
// prompt, captures logits + argmax. Compares bit-wise.
//
// Each seq's prefill is SEQUENTIAL (no concurrent decode). This eliminates
// concurrency-induced non-determinism. The only variable is the seq_id used
// in `common_batch_add`.
//
// Predicted outcomes:
//   - If logits[seq=0] == logits[seq=K] bit-wise for all K: no slot-id bug
//     in serial path. The NP=8 intra-NP failure comes purely from concurrent
//     batch processing (race conditions, scheduling order, etc.).
//   - If logits differ: slot-id addressing bug exists in serial path. The
//     focused fixtures (CY.F.10..13) narrow down which subsystem.
//
// Env:
//   LLAMA_TEST_TARGET     — target GGUF
//   LLAMA_TEST_PROMPT     — prompt string (optional; default: production prompt)
//   LLAMA_TEST_N_SEQ_MAX  — n_seq_max for the context (default: 8)
//   LLAMA_TEST_MAX_SEQ_ID — max seq_id to test (default: n_seq_max-1)
//
// Exit:
//   0  = all seq_ids produce bit-identical logits
//   1  = at least one seq_id differs
//   77 = SKIP (missing target / load failed)

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
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

int main() {
    const char * target  = std::getenv("LLAMA_TEST_TARGET");
    if (!target) {
        fprintf(stderr, "SKIP: set LLAMA_TEST_TARGET\n");
        return 77;
    }
    const int n_seq_max = std::getenv("LLAMA_TEST_N_SEQ_MAX")
        ? std::atoi(std::getenv("LLAMA_TEST_N_SEQ_MAX")) : 8;
    const int max_seq_id = std::getenv("LLAMA_TEST_MAX_SEQ_ID")
        ? std::atoi(std::getenv("LLAMA_TEST_MAX_SEQ_ID")) : (n_seq_max - 1);
    const char * prompt_env = std::getenv("LLAMA_TEST_PROMPT");
    const std::string prompt = prompt_env
        ? prompt_env
        : "The history of artificial intelligence began in earnest with the work of";

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    mparams.split_mode   = LLAMA_SPLIT_MODE_GRAPH;
    static const char * dev_csv = "CUDA0,CUDA1";
    mparams.devices = dev_csv;
    llama_model * model = llama_model_load_from_file(target, mparams);
    if (!model) { fprintf(stderr, "load failed: %s\n", target); return 77; }

    const int n_vocab = llama_n_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = 4096 * (uint32_t) n_seq_max;
    cparams.n_batch     = 2048;
    cparams.n_ubatch    = 512;
    cparams.n_seq_max   = (uint32_t) n_seq_max;
    cparams.type_k      = GGML_TYPE_Q4_0;
    cparams.type_v      = GGML_TYPE_Q4_0;
    cparams.flash_attn  = true;
    cparams.mla_attn    = 3;
    cparams.k_cache_hadamard = true;
    cparams.v_cache_hadamard = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { llama_free_model(model); fprintf(stderr, "ctx init failed\n"); return 77; }

    // Tokenize prompt once.
    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true, true);
    if (tokens.empty()) {
        fprintf(stderr, "empty tokenization\n");
        return 1;
    }
    const int n_prompt = (int) tokens.size();
    fprintf(stderr, "[CY.F.9] target=%s n_seq_max=%d max_seq_id=%d n_prompt=%d\n",
            target, n_seq_max, max_seq_id, n_prompt);

    // For each seq_id ∈ {0..max_seq_id}, prefill that seq alone and capture
    // the post-prefill logits over the full vocab.
    std::vector<std::vector<float>> per_seq_logits(max_seq_id + 1);
    std::vector<llama_token>        per_seq_argmax(max_seq_id + 1);

    for (int sid = 0; sid <= max_seq_id; ++sid) {
        // Clear this seq's KV (in case prior iterations touched it).
        llama_kv_cache_seq_rm(ctx, (llama_seq_id) sid, -1, -1);

        // Prefill seq=sid in one batch, positions 0..n_prompt-1.
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++) {
            bool last = (i == n_prompt - 1);
            common_batch_add(batch, tokens[i], i, {(llama_seq_id) sid}, last);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[CY.F.9] seq=%d prefill decode FAILED\n", sid);
            llama_batch_free(batch);
            return 1;
        }
        float * logits = llama_get_logits_ith(ctx, n_prompt - 1);
        if (!logits) {
            fprintf(stderr, "[CY.F.9] seq=%d: no logits\n", sid);
            llama_batch_free(batch);
            return 1;
        }
        per_seq_logits[sid].assign(logits, logits + n_vocab);
        per_seq_argmax[sid] = greedy_argmax(logits, n_vocab);
        llama_batch_free(batch);
        fprintf(stderr, "[CY.F.9] seq=%d  argmax=%d  logits[0..3]=%.6f %.6f %.6f %.6f\n",
                sid, per_seq_argmax[sid],
                per_seq_logits[sid][0], per_seq_logits[sid][1],
                per_seq_logits[sid][2], per_seq_logits[sid][3]);
    }

    // Compare seq=0 baseline to each seq=K ∈ {1..max_seq_id}.
    int total_diffs = 0;
    for (int sid = 1; sid <= max_seq_id; ++sid) {
        int byte_diffs = 0;
        float max_abs_diff = 0.0f;
        int max_idx = -1;
        for (int v = 0; v < n_vocab; ++v) {
            uint32_t a, b;
            std::memcpy(&a, &per_seq_logits[0][v], 4);
            std::memcpy(&b, &per_seq_logits[sid][v], 4);
            if (a != b) {
                ++byte_diffs;
                float d = std::fabs(per_seq_logits[0][v] - per_seq_logits[sid][v]);
                if (d > max_abs_diff) { max_abs_diff = d; max_idx = v; }
            }
        }
        if (byte_diffs == 0) {
            fprintf(stderr, "[CY.F.9] seq=%d: BYTE-IDENTICAL to seq=0 (n_vocab=%d)\n",
                    sid, n_vocab);
        } else {
            fprintf(stderr, "[CY.F.9] seq=%d: %d/%d logits differ (max|Δ|=%.3e at vocab idx %d). argmax: seq0=%d seq%d=%d\n",
                    sid, byte_diffs, n_vocab, max_abs_diff, max_idx,
                    per_seq_argmax[0], sid, per_seq_argmax[sid]);
            total_diffs += byte_diffs;
        }
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    if (total_diffs == 0) {
        fprintf(stderr, "[CY.F.9] PASS — all seq_ids bit-identical in serial path\n");
        return 0;
    } else {
        fprintf(stderr, "[CY.F.9] FAIL — slot-id > 0 addressing bug confirmed in serial path\n");
        return 1;
    }
}
