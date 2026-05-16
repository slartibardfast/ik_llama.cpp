// test-cy-seq-id-batched.cpp
//
// Phase CY.F.9b — does batched multi-seq prefill/decode produce per-seq logits
// that bit-match the serial-per-seq baseline?
//
// CY.F.9 (serial) showed: all 8 seqs produce IDENTICAL logits when each is
// prefilled alone. So no slot-id > 0 bug in the serial path.
//
// This test puts MULTIPLE seqs in ONE batch (interleaved tokens across
// seq_ids 0..N-1) and a SINGLE llama_decode call. Then it extracts each
// seq's last-token logits and compares to the serial baseline.
//
// If batched logits[seq=K] differs from serial logits[seq=K] for any K, the
// bug is in how the build graph handles multi-seq batches (NOT in per-slot
// addressing — that's already proven clean).
//
// Env:
//   LLAMA_TEST_TARGET     — target GGUF
//   LLAMA_TEST_N_SEQ_MAX  — n_seq_max (default 8)
//   LLAMA_TEST_NP_BATCH   — how many seq_ids to include in the batched call
//                            (default = n_seq_max)

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

static std::vector<std::vector<float>> capture_serial(
        llama_context * ctx, const std::vector<llama_token> & tokens,
        int max_seq, int n_vocab) {
    const int n_prompt = (int) tokens.size();
    std::vector<std::vector<float>> per_seq_logits(max_seq + 1);
    for (int sid = 0; sid <= max_seq; ++sid) {
        llama_kv_cache_seq_rm(ctx, (llama_seq_id) sid, -1, -1);
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++) {
            bool last = (i == n_prompt - 1);
            common_batch_add(batch, tokens[i], i, {(llama_seq_id) sid}, last);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[serial] seq=%d failed\n", sid);
            llama_batch_free(batch);
            return {};
        }
        float * logits = llama_get_logits_ith(ctx, n_prompt - 1);
        if (!logits) { llama_batch_free(batch); return {}; }
        per_seq_logits[sid].assign(logits, logits + n_vocab);
        llama_batch_free(batch);
    }
    return per_seq_logits;
}

int main() {
    const char * target  = std::getenv("LLAMA_TEST_TARGET");
    if (!target) { fprintf(stderr, "SKIP: set LLAMA_TEST_TARGET\n"); return 77; }
    const int n_seq_max = std::getenv("LLAMA_TEST_N_SEQ_MAX")
        ? std::atoi(std::getenv("LLAMA_TEST_N_SEQ_MAX")) : 8;
    const int np_batch = std::getenv("LLAMA_TEST_NP_BATCH")
        ? std::atoi(std::getenv("LLAMA_TEST_NP_BATCH")) : n_seq_max;
    const std::string prompt =
        "The history of artificial intelligence began in earnest with the work of";

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

    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true, true);
    const int n_prompt = (int) tokens.size();
    fprintf(stderr, "[CY.F.9b] target=%s n_seq_max=%d np_batch=%d n_prompt=%d\n",
            target, n_seq_max, np_batch, n_prompt);

    // ---- Phase 1: serial baseline (re-uses CY.F.9 logic). ----
    fprintf(stderr, "[CY.F.9b] Phase 1: serial baseline per seq_id\n");
    auto serial_logits = capture_serial(ctx, tokens, np_batch - 1, n_vocab);
    if (serial_logits.empty()) return 1;

    // Confirm serial logits all match seq=0 (CY.F.9 already proved this).
    for (int sid = 1; sid < np_batch; ++sid) {
        if (std::memcmp(serial_logits[0].data(), serial_logits[sid].data(),
                        (size_t) n_vocab * sizeof(float)) != 0) {
            fprintf(stderr, "[CY.F.9b] WARN: serial seq=%d differs from seq=0 (this should not happen given CY.F.9)\n", sid);
        }
    }

    // ---- Phase 2: batched np_batch seqs in ONE llama_decode call. ----
    fprintf(stderr, "[CY.F.9b] Phase 2: batched prefill of %d seqs in ONE decode call\n", np_batch);
    // Clear all seqs.
    for (int sid = 0; sid < np_batch; ++sid) {
        llama_kv_cache_seq_rm(ctx, (llama_seq_id) sid, -1, -1);
    }

    // Build batch: emit ALL seqs' tokens. To match the production server, we
    // emit each seq's prompt sequentially in seq order (seq=0's all tokens,
    // then seq=1's, ...). The server's behavior may differ in exact ordering;
    // adjust later if it matters.
    llama_batch batched = llama_batch_init(np_batch * n_prompt, 0, 1);
    for (int sid = 0; sid < np_batch; ++sid) {
        for (int i = 0; i < n_prompt; i++) {
            bool last = (i == n_prompt - 1);
            common_batch_add(batched, tokens[i], i, {(llama_seq_id) sid}, last);
        }
    }
    if (llama_decode(ctx, batched) != 0) {
        fprintf(stderr, "[CY.F.9b] batched decode FAILED\n");
        llama_batch_free(batched);
        return 1;
    }

    // Extract per-seq last-token logits. The logits array layout matches
    // batched.logits[i] order. With our layout (seq 0's tokens first, then
    // seq 1's, ...), the last-token positions are at indices:
    //   (1)*n_prompt - 1  for seq 0
    //   (2)*n_prompt - 1  for seq 1
    //   ...
    //   (np_batch)*n_prompt - 1  for seq np_batch-1
    std::vector<std::vector<float>> batched_logits(np_batch);
    for (int sid = 0; sid < np_batch; ++sid) {
        const int idx = (sid + 1) * n_prompt - 1;
        float * logits = llama_get_logits_ith(ctx, idx);
        if (!logits) {
            fprintf(stderr, "[CY.F.9b] batched seq=%d: no logits at idx=%d\n", sid, idx);
            llama_batch_free(batched);
            return 1;
        }
        batched_logits[sid].assign(logits, logits + n_vocab);
    }
    llama_batch_free(batched);

    // ---- Phase 3: compare batched[seq=K] vs serial[seq=K] ----
    int total_diffs = 0;
    for (int sid = 0; sid < np_batch; ++sid) {
        int byte_diffs = 0;
        float max_abs_diff = 0.0f;
        int max_idx = -1;
        for (int v = 0; v < n_vocab; ++v) {
            uint32_t a, b;
            std::memcpy(&a, &serial_logits[sid][v], 4);
            std::memcpy(&b, &batched_logits[sid][v], 4);
            if (a != b) {
                ++byte_diffs;
                float d = std::fabs(serial_logits[sid][v] - batched_logits[sid][v]);
                if (d > max_abs_diff) { max_abs_diff = d; max_idx = v; }
            }
        }
        const llama_token sa = greedy_argmax(serial_logits[sid].data(), n_vocab);
        const llama_token ba = greedy_argmax(batched_logits[sid].data(), n_vocab);
        if (byte_diffs == 0) {
            fprintf(stderr, "[CY.F.9b] seq=%d: BATCHED == SERIAL (argmax=%d)\n", sid, sa);
        } else {
            fprintf(stderr, "[CY.F.9b] seq=%d: BATCHED != SERIAL — %d/%d logits differ, max|Δ|=%.3e at idx %d. argmax: serial=%d  batched=%d\n",
                    sid, byte_diffs, n_vocab, max_abs_diff, max_idx, sa, ba);
            total_diffs += byte_diffs;
        }
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    if (total_diffs == 0) {
        fprintf(stderr, "[CY.F.9b] PASS — batched logits match serial baseline per seq\n");
        return 0;
    } else {
        fprintf(stderr, "[CY.F.9b] FAIL — batched multi-seq introduces per-seq logit drift\n");
        return 1;
    }
}
