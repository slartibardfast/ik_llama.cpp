// test-deltanet-d1-capture.cpp
//
// PLAN.md D1 — Per-layer per-slot residual capture at np ∈ {1,2,4,8}.
//
// What it does:
//   - Tokenises N prompts (one per slot) following the same offset
//     pattern as test-np-validity-vanilla.cpp.
//   - Prefills each slot sequentially (distinct seq_id).
//   - Configures llama_set_dflash_extract_layers with ALL layer indices
//     [0..n_layer-1] (requires the cap-bump from 16 → 80 in libllama).
//   - Runs ONE decode step at np = N (one token per slot), capturing
//     the residual stream at every layer.
//   - Writes per-slot per-layer residuals to disk as raw float32 binaries:
//       <out_dir>/d1-np{N}-offset{O}-layer{L}-slot{S}.bin   (n_embd floats)
//     and a manifest JSON:
//       <out_dir>/d1-np{N}-offset{O}-manifest.json
//
// Env:
//   LLAMA_TEST_TARGET     — target GGUF (required)
//   LLAMA_TEST_PROMPT_DIR — dir with p{0..NP-1}.txt (required)
//   LLAMA_TEST_NP         — slot count (required; e.g. 1, 2, 4, 8)
//   LLAMA_TEST_OUT_DIR    — output directory (required; must exist)
//   LLAMA_TEST_PROMPT_OFFSET — slot 0 reads p{offset}.txt (default 0)
//
// Exit:
//   0  = capture wrote all expected files
//   1  = at least one decode failed or buffer was unexpectedly empty
//   77 = SKIP (missing env)

#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct slot_state {
    int                       id;
    std::string               prompt_path;
    std::vector<llama_token>  prompt_tokens;
    llama_token               id_last  = 0;
    llama_pos                 next_pos = 0;
    bool                      prefill_ok = true;
};

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
    const char * pd      = std::getenv("LLAMA_TEST_PROMPT_DIR");
    const char * np_env  = std::getenv("LLAMA_TEST_NP");
    const char * out_dir = std::getenv("LLAMA_TEST_OUT_DIR");
    if (!target || !pd || !np_env || !out_dir) {
        fprintf(stderr,
            "SKIP: set LLAMA_TEST_TARGET, LLAMA_TEST_PROMPT_DIR, "
            "LLAMA_TEST_NP, LLAMA_TEST_OUT_DIR\n");
        return 77;
    }
    const int N = std::atoi(np_env);
    const int prompt_offset = std::getenv("LLAMA_TEST_PROMPT_OFFSET")
                            ? std::atoi(std::getenv("LLAMA_TEST_PROMPT_OFFSET")) : 0;
    if (N < 1 || N > 8) {
        fprintf(stderr, "LLAMA_TEST_NP must be in [1,8]\n");
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    mparams.split_mode   = LLAMA_SPLIT_MODE_GRAPH;
    static const char * dev_csv = "CUDA0,CUDA1";
    mparams.devices = dev_csv;
    llama_model * model = llama_model_load_from_file(target, mparams);
    if (!model) { fprintf(stderr, "load failed: %s\n", target); return 1; }

    const int n_layer = llama_n_layer(model);
    const int n_embd  = llama_model_n_embd(model);
    const int n_vocab = llama_n_vocab(model);
    fprintf(stderr, "[D1] model: n_layer=%d n_embd=%d n_vocab=%d\n",
            n_layer, n_embd, n_vocab);

    if (n_layer > 80) {
        fprintf(stderr,
            "[D1] FAIL: n_layer=%d exceeds extract-layers cap (80). "
            "Bump LLAMA_DFLASH_MAX_EXTRACT_LAYERS in libllama.\n", n_layer);
        llama_free_model(model);
        return 1;
    }

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

    std::vector<slot_state> slots(N);
    for (int s = 0; s < N; s++) {
        slots[s].id = s;
        char path[512];
        snprintf(path, sizeof(path), "%s/p%d.txt", pd, s + prompt_offset);
        slots[s].prompt_path = path;
        std::ifstream f(path); std::stringstream ss; ss << f.rdbuf();
        slots[s].prompt_tokens = common_tokenize(ctx, ss.str(), true, true);
        if (slots[s].prompt_tokens.empty()) {
            fprintf(stderr, "slot %d: empty prompt at %s\n", s, path);
            llama_free(ctx); llama_free_model(model); return 1;
        }
    }

    fprintf(stderr, "[D1] N=%d offset=%d — per-slot prefill\n", N, prompt_offset);
    for (int s = 0; s < N; s++) {
        const int n_prompt = (int) slots[s].prompt_tokens.size();
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++) {
            bool last = (i == n_prompt - 1);
            common_batch_add(batch, slots[s].prompt_tokens[i], i, {(llama_seq_id) s}, last);
        }
        if (llama_decode(ctx, batch) != 0) {
            slots[s].prefill_ok = false;
            fprintf(stderr, "[D1] slot %d: prefill decode FAILED\n", s);
            llama_batch_free(batch);
            llama_free(ctx); llama_free_model(model); return 1;
        }
        float * logits = llama_get_logits_ith(ctx, n_prompt - 1);
        if (!logits) { llama_batch_free(batch); llama_free(ctx); llama_free_model(model); return 1; }
        slots[s].id_last  = greedy_argmax(logits, n_vocab);
        slots[s].next_pos = (llama_pos) n_prompt;
        llama_batch_free(batch);
    }

    // Configure all-layers extract. Per llama-cparams.h the cap is 80;
    // n_layer ≤ 80 was checked above.
    std::vector<int32_t> all_layers(n_layer);
    for (int l = 0; l < n_layer; ++l) all_layers[l] = l;
    llama_set_dflash_extract_layers(ctx, all_layers.data(), (int32_t) n_layer);

    // One decode step at np = N. n_tokens = N, one token per slot.
    fprintf(stderr, "[D1] capture decode: N=%d tokens (one per slot)\n", N);
    llama_batch batch = llama_batch_init(N, 0, 1);
    for (int s = 0; s < N; s++) {
        common_batch_add(batch, slots[s].id_last, slots[s].next_pos, {(llama_seq_id) s}, true);
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "[D1] capture decode FAILED\n");
        llama_batch_free(batch);
        llama_free(ctx); llama_free_model(model); return 1;
    }
    llama_batch_free(batch);

    // Read each layer's buffer, slice per slot, write to disk.
    // Buffer layout per cb_eval: [n_embd, n_tokens] column-major after
    // F16→F32 conversion (or already F32). Slot k's residual at this
    // layer is contiguous floats [k*n_embd, (k+1)*n_embd).
    const size_t expected_per_layer = (size_t) n_embd * (size_t) N;
    std::vector<float> staging(expected_per_layer);

    int n_written = 0;
    char manifest_path[1024];
    snprintf(manifest_path, sizeof(manifest_path),
        "%s/d1-np%d-offset%d-manifest.json", out_dir, N, prompt_offset);
    FILE * manifest = std::fopen(manifest_path, "w");
    if (!manifest) {
        fprintf(stderr, "[D1] cannot open manifest %s\n", manifest_path);
        llama_free(ctx); llama_free_model(model); return 1;
    }
    fprintf(manifest,
        "{\n  \"np\": %d,\n  \"offset\": %d,\n  \"n_layer\": %d,\n  \"n_embd\": %d,\n  "
        "\"slots\": [\n", N, prompt_offset, n_layer, n_embd);
    for (int s = 0; s < N; s++) {
        fprintf(manifest,
            "    {\"slot\": %d, \"prompt\": \"%s\", \"n_prompt_tokens\": %zu, "
            "\"id_last\": %d}%s\n",
            s, slots[s].prompt_path.c_str(), slots[s].prompt_tokens.size(),
            (int) slots[s].id_last, s + 1 < N ? "," : "");
    }
    fprintf(manifest, "  ],\n  \"layers\": [\n");

    // The qwen35 build graph emits cb(cur, "l_out", il) only for
    // transformer layers (`for il < n_layer - nextn_predict_layers`).
    // For Qwen 3.6 27B (n_layer=65, nextn_predict_layers=1) layers 0..63
    // are captured; layer 64 is the MTP head and not on the vanilla path.
    // Skip layers that produced no buffer rather than erroring.
    int n_layers_captured = 0;
    bool first_layer_in_manifest = true;
    for (int l = 0; l < n_layer; l++) {
        const size_t got = llama_get_dflash_extract_data(
            ctx, l, staging.data(), staging.size());
        if (got == 0) {
            fprintf(stderr,
                "[D1] layer %d: no buffer (likely MTP/aux layer not on vanilla path) — skipping\n", l);
            continue;
        }
        if (got != expected_per_layer) {
            fprintf(stderr,
                "[D1] layer %d: got %zu floats, expected %zu (n_embd=%d * N=%d) — "
                "buffer-size mismatch indicates cb_eval dtype split or kernel scope issue\n",
                l, got, expected_per_layer, n_embd, N);
            std::fclose(manifest);
            llama_free(ctx); llama_free_model(model); return 1;
        }
        for (int s = 0; s < N; s++) {
            char fp[1024];
            snprintf(fp, sizeof(fp),
                "%s/d1-np%d-offset%d-layer%d-slot%d.bin",
                out_dir, N, prompt_offset, l, s);
            FILE * f = std::fopen(fp, "wb");
            if (!f) {
                fprintf(stderr, "[D1] cannot open %s\n", fp);
                std::fclose(manifest);
                llama_free(ctx); llama_free_model(model); return 1;
            }
            const size_t to_write = (size_t) n_embd;
            std::fwrite(staging.data() + (size_t) s * (size_t) n_embd,
                        sizeof(float), to_write, f);
            std::fclose(f);
            n_written++;
        }
        fprintf(manifest,
            "%s    {\"layer\": %d, \"floats_per_slot\": %d}",
            first_layer_in_manifest ? "" : ",\n", l, n_embd);
        first_layer_in_manifest = false;
        n_layers_captured++;
    }
    fprintf(manifest, "\n  ],\n  \"n_layers_captured\": %d,\n  \"n_files_written\": %d\n}\n",
            n_layers_captured, n_written);
    std::fclose(manifest);

    fprintf(stderr, "[D1] wrote %d per-slot per-layer files to %s\n", n_written, out_dir);
    fprintf(stderr, "[D1] manifest: %s\n", manifest_path);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
