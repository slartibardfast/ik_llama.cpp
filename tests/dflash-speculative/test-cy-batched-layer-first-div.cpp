// test-cy-batched-layer-first-div.cpp
//
// Phase CY.F.14 — find the FIRST LAYER where batched multi-seq output
// (seq=0's row of the multi-seq batch) differs from serial single-seq output.
//
// CY.F.9b confirmed batched != serial at the final logits. This drills into
// per-layer residuals via the dflash_extract_layers mechanism (authoritative;
// uses dtype branching matching libllama's internal capture).
//
// Approach:
//   1. Initialize context with n_seq_max=8.
//   2. Configure extract_layers for all 64 transformer layers.
//   3. Phase A (serial): prefill seq=0 alone (12 tokens), capture all 64
//      layers' residuals (12 rows × 5120 floats per layer). Store row 11
//      (last prefill token of seq=0) per layer.
//   4. Phase B (batched): prefill seqs 0..7 in ONE batch (96 tokens),
//      capture all 64 layers' residuals (96 rows × 5120 per layer). Extract
//      row 11 per layer (seq=0's last prefill token; seq=0's tokens come
//      first in our batch layout).
//   5. Compare row-11 serial vs row-11 batched per layer. First differing
//      layer is the source.
//
// Env:
//   LLAMA_TEST_TARGET   — target GGUF
//   LLAMA_TEST_N_SEQ_MAX — n_seq_max (default 8)
//   LLAMA_TEST_NP_BATCH — seqs in batch (default 8)

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main() {
    const char * target = std::getenv("LLAMA_TEST_TARGET");
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

    const int n_layer = llama_n_layer(model);
    const int n_embd  = llama_model_n_embd(model);
    fprintf(stderr, "[CY.F.14] n_layer=%d n_embd=%d n_seq_max=%d np_batch=%d\n",
            n_layer, n_embd, n_seq_max, np_batch);
    if (n_layer > 80) {
        fprintf(stderr, "[CY.F.14] FAIL: n_layer=%d > extract-layers cap 80\n", n_layer);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = 4096 * (uint32_t) n_seq_max;
    cparams.n_batch     = 2048;
    cparams.n_ubatch    = 2048;  // ensure full prefill in single ubatch
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
    fprintf(stderr, "[CY.F.14] n_prompt=%d\n", n_prompt);

    // Configure all-layer extract.
    std::vector<int32_t> all_layers(n_layer);
    for (int l = 0; l < n_layer; ++l) all_layers[l] = l;
    llama_set_dflash_extract_layers(ctx, all_layers.data(), (int32_t) n_layer);

    // ---- Phase A: SERIAL — prefill seq=0 alone ----
    fprintf(stderr, "[CY.F.14] Phase A: serial prefill seq=0\n");
    llama_kv_cache_seq_rm(ctx, 0, -1, -1);
    {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++) {
            bool last = (i == n_prompt - 1);
            common_batch_add(batch, tokens[i], i, {(llama_seq_id) 0}, last);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[CY.F.14] serial decode FAILED\n");
            llama_batch_free(batch); return 1;
        }
        llama_batch_free(batch);
    }

    // Each layer's buffer should have n_prompt rows × n_embd floats.
    std::vector<std::vector<float>> serial_layer(n_layer);
    const size_t serial_per_layer_size = (size_t) n_embd * (size_t) n_prompt;
    for (int l = 0; l < n_layer; ++l) {
        std::vector<float> staging(serial_per_layer_size);
        const size_t got = llama_get_dflash_extract_data(ctx, l, staging.data(), staging.size());
        if (got == 0) continue;  // MTP/aux skip
        if (got != serial_per_layer_size) {
            fprintf(stderr, "[CY.F.14] serial layer %d: got %zu floats, expected %zu\n",
                    l, got, serial_per_layer_size);
            // Continue anyway; some layers may have other shapes.
        }
        serial_layer[l] = std::move(staging);
    }

    // ---- Phase B: BATCHED — prefill np_batch seqs in ONE call ----
    fprintf(stderr, "[CY.F.14] Phase B: batched prefill of %d seqs\n", np_batch);
    for (int sid = 0; sid < np_batch; ++sid) {
        llama_kv_cache_seq_rm(ctx, (llama_seq_id) sid, -1, -1);
    }
    // Reset extract buffers to clear serial captures.
    llama_set_dflash_extract_layers(ctx, all_layers.data(), (int32_t) n_layer);

    {
        const int total = np_batch * n_prompt;
        llama_batch batch = llama_batch_init(total, 0, 1);
        for (int sid = 0; sid < np_batch; ++sid) {
            for (int i = 0; i < n_prompt; i++) {
                bool last = (i == n_prompt - 1);
                common_batch_add(batch, tokens[i], i, {(llama_seq_id) sid}, last);
            }
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[CY.F.14] batched decode FAILED\n");
            llama_batch_free(batch); return 1;
        }
        llama_batch_free(batch);
    }

    std::vector<std::vector<float>> batched_layer(n_layer);
    const size_t batched_per_layer_size = (size_t) n_embd * (size_t) np_batch * (size_t) n_prompt;
    for (int l = 0; l < n_layer; ++l) {
        std::vector<float> staging(batched_per_layer_size);
        const size_t got = llama_get_dflash_extract_data(ctx, l, staging.data(), staging.size());
        if (got == 0) continue;
        if (got != batched_per_layer_size) {
            fprintf(stderr, "[CY.F.14] batched layer %d: got %zu floats, expected %zu\n",
                    l, got, batched_per_layer_size);
        }
        batched_layer[l] = std::move(staging);
    }

    // ---- Phase C: compare seq=0's last-token row per layer ----
    fprintf(stderr, "[CY.F.14] Phase C: comparing seq=0 last-token row per layer\n");
    // Serial: row = n_prompt-1 (the last token of seq=0, which is the only seq).
    // Batched: same row index n_prompt-1 (seq=0's last token, since seq=0's
    //          tokens come first in our batch layout).
    int first_diff_layer = -1;
    for (int l = 0; l < n_layer; ++l) {
        if (serial_layer[l].empty() || batched_layer[l].empty()) continue;
        const float * sr = serial_layer[l].data()  + (size_t)(n_prompt - 1) * n_embd;
        const float * br = batched_layer[l].data() + (size_t)(n_prompt - 1) * n_embd;
        int ndiff = 0; float maxd = 0.0f;
        for (int i = 0; i < n_embd; ++i) {
            uint32_t a, b;
            std::memcpy(&a, &sr[i], 4);
            std::memcpy(&b, &br[i], 4);
            if (a != b) {
                ++ndiff;
                float d = std::fabs(sr[i] - br[i]);
                if (d > maxd) maxd = d;
            }
        }
        if (ndiff > 0 && first_diff_layer < 0) first_diff_layer = l;
        fprintf(stderr, "  layer %2d: ndiff=%5d/%d  max|Δ|=%.3e\n",
                l, ndiff, n_embd, maxd);
    }
    fprintf(stderr, "[CY.F.14] FIRST DIVERGENT LAYER (serial seq=0 last-tok vs batched seq=0 last-tok): %d\n", first_diff_layer);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
