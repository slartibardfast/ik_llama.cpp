// test-deltanet-s23-op-capture.cpp
//
// PLAN.md Stage 2 S2.3 — Op-level intermediate capture at a specified
// target layer. Generalizes test-deltanet-d1-capture.cpp: instead of
// capturing only the per-layer residual (l_out-<il>), capture EVERY
// named intermediate tensor at one or more target layers. Used to
// localize WHICH op within a first-divergent layer (S2.1 named the
// layer; this names the op).
//
// Mechanism: install a custom ggml_backend_sched_eval_callback via
// llama_context_params.cb_eval. The callback inspects every computed
// node's name and captures the data if the name ends with "-<target>"
// for any target in our list. Output: one .bin file per
// (target_layer, tensor_name, slot).
//
// Env:
//   LLAMA_TEST_TARGET     — target GGUF (required)
//   LLAMA_TEST_PROMPT_DIR — dir with p{0..NP-1}.txt (required)
//   LLAMA_TEST_NP         — slot count (required; e.g. 2, 4)
//   LLAMA_TEST_OUT_DIR    — output directory (required; must exist)
//   LLAMA_TEST_PROMPT_OFFSET — slot 0 reads p{offset}.txt (default 0)
//   LLAMA_TEST_TARGET_LAYERS — comma-separated layer indices to capture
//                              (default "0,3,19" — the 3 S2.1 first-div layers)
//
// Exit:
//   0  = captured at least one tensor at each target layer per slot
//   1  = decode failed or no tensors captured
//   77 = SKIP (missing env)

#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
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

struct captured_tensor {
    std::string name;
    int layer;
    int64_t ne[4];           // full shape (ne[0], ne[1], ne[2], ne[3])
    ggml_type dtype;
    std::vector<float> data; // size = product of ne[*] (F32)
};

struct capture_state {
    std::set<int> target_layers;
    std::vector<captured_tensor> captured;
    bool enabled = false;  // false during prefill, true only during the diagnostic decode
};

// Custom cb_eval: capture every named tensor whose name suffix is "-<L>"
// for L in target_layers. CRITICAL: the `ask` arm must return FALSE for
// non-matching tensors — returning true at ask preserves the tensor and
// breaks scheduler fusions. Pre-fix this harness returned true for all
// ask queries which forced the scheduler to preserve every tensor and
// perturbed numerical output by ~4 max_abs_diff vs the production path.
// Matching D1's libllama cb_eval pattern: filter in both ask AND
// post-compute phases identically.
static bool s23_capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (capture_state *) user_data;
    if (!st->enabled) return false;
    // Name match: "<base>-<layer>" with layer in target set.
    const char * name = t->name;
    const char * dash = std::strrchr(name, '-');
    if (!dash) return false;
    const char * tail = dash + 1;
    if (*tail == '\0') return false;
    char * end = nullptr;
    long layer = std::strtol(tail, &end, 10);
    if (end == tail || *end != '\0') return false;
    if (st->target_layers.find((int) layer) == st->target_layers.end()) return false;
    if (ask) return true;  // we want this tensor; preserve through scheduler
    if (t->buffer == nullptr) return true;

    // Capture. Handle F32 directly; F16 via row-conversion. Skip other dtypes.
    captured_tensor c;
    c.name = std::string(name, dash - name);
    c.layer = (int) layer;
    for (int i = 0; i < 4; i++) c.ne[i] = t->ne[i];
    c.dtype = t->type;
    const size_t nelems = (size_t) ggml_nelements(t);
    const size_t nbytes = ggml_nbytes(t);

    if (t->type == GGML_TYPE_F32) {
        c.data.resize(nelems);
        ggml_backend_tensor_get(t, c.data.data(), 0, nbytes);
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> staging(nelems);
        ggml_backend_tensor_get(t, staging.data(), 0, nbytes);
        c.data.resize(nelems);
        ggml_fp16_to_fp32_row(staging.data(), c.data.data(), nelems);
    } else {
        return true;
    }
    st->captured.push_back(std::move(c));
    return true;
}

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
    // PLAN.md S2.3 n_kv-pad confirmation test: when set, all slots
    // load the SAME prompt (file p<prompt_offset>.txt). Holds n_kv
    // identical across NP at the prefill boundary, isolating entry
    // point (2) — Q-column padding / WMMA fragment scheduling
    // sensitivity — from entry point (1) — n_kv-variation roundoff.
    const bool use_same_prompt = std::getenv("LLAMA_TEST_USE_SAME_PROMPT")
                                && std::atoi(std::getenv("LLAMA_TEST_USE_SAME_PROMPT")) != 0;
    if (N < 1 || N > 8) {
        fprintf(stderr, "LLAMA_TEST_NP must be in [1,8]\n");
        return 1;
    }

    capture_state cap;
    const char * tl_env = std::getenv("LLAMA_TEST_TARGET_LAYERS");
    const std::string tl = tl_env ? tl_env : "0,3,19";
    {
        std::stringstream ss(tl);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) cap.target_layers.insert(std::atoi(item.c_str()));
        }
    }
    fprintf(stderr, "[S2.3] target_layers = {");
    bool first = true;
    for (int l : cap.target_layers) { fprintf(stderr, "%s%d", first ? "" : ",", l); first = false; }
    fprintf(stderr, "}\n");

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    mparams.split_mode   = LLAMA_SPLIT_MODE_GRAPH;
    static const char * dev_csv = "CUDA0,CUDA1";
    mparams.devices = dev_csv;
    llama_model * model = llama_model_load_from_file(target, mparams);
    if (!model) { fprintf(stderr, "load failed: %s\n", target); return 1; }

    const int n_vocab = llama_n_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = 4096;
    cparams.n_batch     = 2048;
    cparams.n_ubatch    = 512;
    cparams.n_seq_max   = (uint32_t) N;
    cparams.type_k      = GGML_TYPE_Q4_0;
    cparams.type_v      = GGML_TYPE_Q4_0;
    cparams.flash_attn  = true;
    cparams.mla_attn    = 3;
    cparams.cb_eval     = s23_capture_cb;
    cparams.cb_eval_user_data = &cap;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { llama_free_model(model); fprintf(stderr, "ctx init failed\n"); return 1; }

    std::vector<slot_state> slots(N);
    for (int s = 0; s < N; s++) {
        slots[s].id = s;
        char path[512];
        // When use_same_prompt is set, every slot loads p<prompt_offset>.txt
        // (n_kv-pad confirmation test). Otherwise the usual per-slot mapping.
        const int prompt_idx = use_same_prompt ? prompt_offset : (s + prompt_offset);
        snprintf(path, sizeof(path), "%s/p%d.txt", pd, prompt_idx);
        slots[s].prompt_path = path;
        std::ifstream f(path); std::stringstream ss; ss << f.rdbuf();
        slots[s].prompt_tokens = common_tokenize(ctx, ss.str(), true, true);
        if (slots[s].prompt_tokens.empty()) {
            fprintf(stderr, "slot %d: empty prompt at %s\n", s, path);
            llama_free(ctx); llama_free_model(model); return 1;
        }
    }

    fprintf(stderr, "[S2.3] N=%d offset=%d — per-slot prefill (cb_eval fires; ignored until decode)\n", N, prompt_offset);
    // Clear any captures from prefill — we only want the decode step.
    for (int s = 0; s < N; s++) {
        const int n_prompt = (int) slots[s].prompt_tokens.size();
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++) {
            bool last = (i == n_prompt - 1);
            common_batch_add(batch, slots[s].prompt_tokens[i], i, {(llama_seq_id) s}, last);
        }
        if (llama_decode(ctx, batch) != 0) {
            slots[s].prefill_ok = false;
            fprintf(stderr, "[S2.3] slot %d: prefill decode FAILED\n", s);
            llama_batch_free(batch);
            llama_free(ctx); llama_free_model(model); return 1;
        }
        float * logits = llama_get_logits_ith(ctx, n_prompt - 1);
        if (!logits) { llama_batch_free(batch); llama_free(ctx); llama_free_model(model); return 1; }
        slots[s].id_last  = greedy_argmax(logits, n_vocab);
        slots[s].next_pos = (llama_pos) n_prompt;
        llama_batch_free(batch);
    }
    fprintf(stderr, "[S2.3] prefill done; arming capture for decode\n");
    cap.enabled = true;
    cap.captured.clear();

    fprintf(stderr, "[S2.3] capture decode: N=%d tokens (one per slot)\n", N);
    llama_batch batch = llama_batch_init(N, 0, 1);
    for (int s = 0; s < N; s++) {
        common_batch_add(batch, slots[s].id_last, slots[s].next_pos, {(llama_seq_id) s}, true);
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "[S2.3] capture decode FAILED\n");
        llama_batch_free(batch);
        llama_free(ctx); llama_free_model(model); return 1;
    }
    llama_batch_free(batch);

    fprintf(stderr, "[S2.3] captured %zu tensor fires (multiple fires per tensor expected from graph splits)\n",
            cap.captured.size());

    // Group captured tensors by (name, layer). For each group, the FIRST
    // fire holds the batched [n_embd, n_tokens] tensor matching the
    // production forward path. Subsequent fires are graph-split
    // artifacts (slot-by-slot). Take the first fire only per group.
    std::map<std::string, captured_tensor> first_fire;
    for (auto & c : cap.captured) {
        char key[256];
        snprintf(key, sizeof(key), "%s|%d", c.name.c_str(), c.layer);
        if (first_fire.find(key) == first_fire.end()) {
            first_fire[key] = c;
        }
    }
    fprintf(stderr, "[S2.3] %zu distinct (name, layer) groups captured\n", first_fire.size());

    int n_written = 0;
    char manifest_path[1024];
    snprintf(manifest_path, sizeof(manifest_path),
        "%s/s23-np%d-offset%d-manifest.json", out_dir, N, prompt_offset);
    FILE * manifest = std::fopen(manifest_path, "w");
    if (!manifest) {
        fprintf(stderr, "[S2.3] cannot open manifest %s\n", manifest_path);
        llama_free(ctx); llama_free_model(model); return 1;
    }
    fprintf(manifest, "{\n  \"np\": %d,\n  \"offset\": %d,\n  \"target_layers\": [", N, prompt_offset);
    bool first_l = true;
    for (int l : cap.target_layers) { fprintf(manifest, "%s%d", first_l ? "" : ",", l); first_l = false; }
    fprintf(manifest, "],\n  \"slots\": [\n");
    for (int s = 0; s < N; s++) {
        fprintf(manifest,
            "    {\"slot\": %d, \"prompt\": \"%s\", \"n_prompt_tokens\": %zu, \"id_last\": %d}%s\n",
            s, slots[s].prompt_path.c_str(), slots[s].prompt_tokens.size(),
            (int) slots[s].id_last, s + 1 < N ? "," : "");
    }
    fprintf(manifest, "  ],\n  \"captures\": [\n");
    bool first_cap = true;
    for (auto & kv : first_fire) {
        captured_tensor & c = kv.second;
        const int64_t nelems = c.ne[0] * c.ne[1] * c.ne[2] * c.ne[3];
        if (nelems <= 0 || (int64_t) c.data.size() != nelems) continue;
        // Dump the FULL tensor — Python slices per-slot based on shape.
        char fp[1024];
        snprintf(fp, sizeof(fp),
            "%s/s23-np%d-offset%d-layer%d-name_%s-full.bin",
            out_dir, N, prompt_offset, c.layer, c.name.c_str());
        FILE * f = std::fopen(fp, "wb");
        if (!f) {
            fprintf(stderr, "[S2.3] cannot open %s\n", fp);
            continue;
        }
        std::fwrite(c.data.data(), sizeof(float), c.data.size(), f);
        std::fclose(f);
        n_written++;
        fprintf(manifest,
            "%s    {\"name\": \"%s\", \"layer\": %d, "
            "\"ne\": [%lld, %lld, %lld, %lld], \"dtype\": \"%s\"}",
            first_cap ? "" : ",\n", c.name.c_str(), c.layer,
            (long long) c.ne[0], (long long) c.ne[1],
            (long long) c.ne[2], (long long) c.ne[3],
            ggml_type_name(c.dtype));
        first_cap = false;
    }
    fprintf(manifest, "\n  ],\n  \"n_files_written\": %d\n}\n", n_written);
    std::fclose(manifest);

    fprintf(stderr, "[S2.3] wrote %d per-slot per-(name,layer) files\n", n_written);
    fprintf(stderr, "[S2.3] manifest: %s\n", manifest_path);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return n_written == 0 ? 1 : 0;
}
