// test-cy-f18-layer-bisect.cpp
//
// CY.F.18 layer-bisect probe — find which layer first races across two
// NP=2 decode-step-1 runs.
//
// Background: after CY.F.17 (MMQ stream_K disable), slot 0 of NP=2 is
// byte-identical to NP=1 across all runs. Slot 1 still races
// (~20% pass rate). Race fires at decode step 1 itself with logit
// magnitude ~0.62 — substantive, not 1-ULP noise.
//
// This probe captures layer-by-layer residual snapshots at decode step 1
// using the existing llama_set_dflash_extract_layers API. Runs NP=2 twice
// (with identical inputs, identical batched prefill, same first decode
// step), compares snapshots layer-by-layer across runs and across slots.
//
// Reports: first layer where slot 1 cross-run divergence appears.
// Slot 0 is the control — expected byte-identical across runs at every
// layer. If slot 0 ever differs, the assumption breaks and we have a
// different problem.
//
// Build: this is a unit-style probe; uses the production target GGUF
// via LLAMA_TEST_TARGET. Requires GGML_CUDA_MMQ_DISABLE_STREAM_K=1
// (or test will fail at slot 0 by stream_K shape-dep, not slot 1 race).
//
// Usage:
//   GGML_CUDA_MMQ_DISABLE_STREAM_K=1 \
//   LLAMA_TEST_TARGET=/opt/models/.../qwen3.6-27b.gguf \
//   LLAMA_PSKV_MODE=singlewarp \
//   LLAMA_FATTN_PER_SLOT_KV_ENABLE=1 \
//     ./build/bin/test-cy-f18-layer-bisect

#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

static llama_token greedy_argmax(const float * logits, int n_vocab) {
    llama_token best = 0;
    float bv = logits[0];
    for (int i = 1; i < n_vocab; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

struct LayerSnapshot {
    int layer_id;
    int n_floats;
    std::vector<float> data;  // [n_floats] flat row-major
};

struct CaptureResult {
    std::vector<LayerSnapshot> layers;
    // Last-prefill logits per slot (full vocab vector).
    std::vector<float> slot0_prefill_logits;
    std::vector<float> slot1_prefill_logits;
    // Decode step 1 logits per slot (full vocab vector).
    std::vector<float> slot0_logits;
    std::vector<float> slot1_logits;
    llama_token slot0_prefill_argmax{0};
    llama_token slot1_prefill_argmax{0};
};

// Returns per-layer snapshots after decode step 1 (i.e., the first
// llama_decode() call after batched prefill).
// If layer_ids is empty, skips the dflash_extract setup entirely — useful
// to test whether the extract setOutput markers suppress an otherwise-active
// race in LM head.
static CaptureResult run_one_capture(
        llama_model * model,
        const std::vector<llama_token> & prompt_tokens,
        const std::vector<int32_t> & layer_ids,
        int n_seq_max,
        int np_active) {
    CaptureResult result;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096 * 8;
    cp.n_batch = 2048; cp.n_ubatch = 2048;
    cp.n_seq_max = n_seq_max;
    cp.type_k = GGML_TYPE_Q4_0; cp.type_v = GGML_TYPE_Q4_0;
    cp.flash_attn = true; cp.mla_attn = 3;
    cp.k_cache_hadamard = true; cp.v_cache_hadamard = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return result;

    const int n_vocab = llama_n_vocab(model);
    const int n_prompt = (int) prompt_tokens.size();

    // Clear all seqs.
    for (int sid = 0; sid < n_seq_max; ++sid) {
        llama_kv_cache_seq_rm(ctx, (llama_seq_id) sid, -1, -1);
    }

    // Batched prefill.
    {
        llama_batch batch = llama_batch_init(np_active * n_prompt, 0, 1);
        for (int sid = 0; sid < np_active; ++sid) {
            for (int i = 0; i < n_prompt; ++i) {
                common_batch_add(batch, prompt_tokens[i], i, {(llama_seq_id) sid}, i == n_prompt - 1);
            }
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "prefill failed\n");
            llama_batch_free(batch);
            llama_free(ctx);
            return result;
        }
        llama_batch_free(batch);
    }

    // Argmax of each slot's last prefill logit → first decode tok.
    // ALSO capture full prefill logits per slot for cross-run comparison.
    std::vector<llama_token> last_tok(np_active);
    for (int sid = 0; sid < np_active; ++sid) {
        const int idx = (sid + 1) * n_prompt - 1;
        float * logits = llama_get_logits_ith(ctx, idx);
        if (!logits) { llama_free(ctx); return result; }
        last_tok[sid] = greedy_argmax(logits, n_vocab);
        if (sid == 0) { result.slot0_prefill_logits.assign(logits, logits + n_vocab); result.slot0_prefill_argmax = last_tok[sid]; }
        if (sid == 1) { result.slot1_prefill_logits.assign(logits, logits + n_vocab); result.slot1_prefill_argmax = last_tok[sid]; }
    }

    // Configure layer extraction BEFORE decode step 1, ONLY if layer_ids is
    // non-empty. With layer_ids empty, no markers are set — pure decode.
    if (!layer_ids.empty()) {
        llama_set_dflash_extract_layers(ctx, layer_ids.data(), (int32_t) layer_ids.size());
    }

    // Decode step 1 (batched, one token per slot).
    {
        llama_batch dec_batch = llama_batch_init(np_active, 0, 1);
        for (int sid = 0; sid < np_active; ++sid) {
            common_batch_add(dec_batch, last_tok[sid], (llama_pos)(n_prompt), {(llama_seq_id) sid}, true);
        }
        if (llama_decode(ctx, dec_batch) != 0) {
            fprintf(stderr, "decode step 1 failed\n");
            llama_batch_free(dec_batch);
            llama_free(ctx);
            return result;
        }
        llama_batch_free(dec_batch);
    }

    // ALWAYS capture decode step 1 logits for slot 0 and slot 1 (the
    // authoritative comparison endpoint).
    {
        float * l0 = llama_get_logits_ith(ctx, 0);
        float * l1 = llama_get_logits_ith(ctx, 1);
        if (l0) result.slot0_logits.assign(l0, l0 + n_vocab);
        if (l1) result.slot1_logits.assign(l1, l1 + n_vocab);
    }

    // Pull each layer's snapshot. Expected layout per dflash_extract API:
    // [n_embd, n_tokens] = [5120, 2] = 10240 floats per layer at np=2 decode.
    const size_t max_floats = 65536;
    for (size_t i = 0; i < layer_ids.size(); ++i) {
        LayerSnapshot snap;
        snap.layer_id = layer_ids[i];
        snap.data.resize(max_floats, 0.0f);
        size_t got = llama_get_dflash_extract_data(ctx, (int32_t) i, snap.data.data(), max_floats);
        snap.n_floats = (int) got;
        snap.data.resize(got);
        result.layers.push_back(std::move(snap));
    }

    llama_free(ctx);
    return result;
}

int main() {
    const char * target = std::getenv("LLAMA_TEST_TARGET");
    if (!target) { fprintf(stderr, "SKIP: set LLAMA_TEST_TARGET\n"); return 77; }

    const char * prompt_env = std::getenv("LLAMA_TEST_PROMPT");
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
        "AI winters, in the late 1970s and again in the late 1980s, when reductions in "
        "investment and skepticism about the field's progress curtailed many projects. "
        "Despite these contractions, work continued in academic and industrial labs. ";

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    mparams.split_mode = LLAMA_SPLIT_MODE_GRAPH;
    static const char * dev_csv = "CUDA0,CUDA1";
    mparams.devices = dev_csv;
    llama_model * model = llama_model_load_from_file(target, mparams);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }

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
    fprintf(stderr, "[CY.F.18 layer-bisect] n_prompt=%d\n", (int)tokens.size());

    // Capture snapshot of all 64 layers at decode step 1.
    // n_layer for qwen3.6 27B is 64; extract API supports up to 80.
    const int n_layer = 64;
    std::vector<int32_t> layer_ids;
    for (int i = 0; i < n_layer; ++i) layer_ids.push_back(i);

    // CONTROL run pair: NO layer markers set. Pure decode. Tests whether
    // the race fires when nothing is forcing graph serialization.
    const bool skip_extract = std::getenv("LLAMA_TEST_NO_EXTRACT") != nullptr;
    std::vector<int32_t> use_layer_ids = layer_ids;
    if (skip_extract) {
        fprintf(stderr, "[CY.F.18] LLAMA_TEST_NO_EXTRACT=1 — disabling per-layer markers\n");
        use_layer_ids.clear();
    }

    // Run TWICE under identical inputs.
    fprintf(stderr, "[CY.F.18] run 0...\n");
    auto run0 = run_one_capture(model, tokens, use_layer_ids, 2, 2);
    fprintf(stderr, "[CY.F.18] run 1...\n");
    auto run1 = run_one_capture(model, tokens, use_layer_ids, 2, 2);

    // Logit comparison (always done, even when extract is disabled).
    if (!run0.slot0_logits.empty() && !run1.slot0_logits.empty()) {
        const int n_vocab = llama_n_vocab(model);
        // 1) Prefill last-token logits cross-run.
        for (int sid = 0; sid < 2; ++sid) {
            const auto & a = (sid == 0) ? run0.slot0_prefill_logits : run0.slot1_prefill_logits;
            const auto & b = (sid == 0) ? run1.slot0_prefill_logits : run1.slot1_prefill_logits;
            int diffs = 0; float maxd = 0.0f;
            for (int v = 0; v < n_vocab; ++v) {
                uint32_t ua, ub;
                std::memcpy(&ua, &a[v], 4);
                std::memcpy(&ub, &b[v], 4);
                if (ua != ub) { ++diffs; float d = std::fabs(a[v]-b[v]); if (d>maxd) maxd=d; }
            }
            fprintf(stderr, "[CY.F.18] PREFILL last-tok slot %d cross-run: %d/%d differ, max|Δ|=%.3e (argmax run0=%d run1=%d %s)\n",
                    sid, diffs, n_vocab, maxd,
                    sid == 0 ? run0.slot0_prefill_argmax : run0.slot1_prefill_argmax,
                    sid == 0 ? run1.slot0_prefill_argmax : run1.slot1_prefill_argmax,
                    (sid == 0 ? run0.slot0_prefill_argmax == run1.slot0_prefill_argmax
                              : run0.slot1_prefill_argmax == run1.slot1_prefill_argmax) ? "match" : "DIFFER");
        }
        // 2) Decode step 1 logits cross-run.
        for (int sid = 0; sid < 2; ++sid) {
            const auto & a = (sid == 0) ? run0.slot0_logits : run0.slot1_logits;
            const auto & b = (sid == 0) ? run1.slot0_logits : run1.slot1_logits;
            int diffs = 0; float maxd = 0.0f;
            for (int v = 0; v < n_vocab; ++v) {
                uint32_t ua, ub;
                std::memcpy(&ua, &a[v], 4);
                std::memcpy(&ub, &b[v], 4);
                if (ua != ub) { ++diffs; float d = std::fabs(a[v]-b[v]); if (d>maxd) maxd=d; }
            }
            fprintf(stderr, "[CY.F.18] DECODE-step1 slot %d cross-run: %d/%d differ, max|Δ|=%.3e\n",
                    sid, diffs, n_vocab, maxd);
        }
    }

    if (skip_extract) {
        fprintf(stderr, "[CY.F.18] Skipping layer compare (no markers set)\n");
        llama_free_model(model);
        llama_backend_free();
        return 0;
    }

    if (run0.layers.empty() || run1.layers.empty()) {
        fprintf(stderr, "FAIL: captures empty\n");
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }
    if (run0.layers.size() != run1.layers.size()) {
        fprintf(stderr, "FAIL: layer count mismatch (run0=%zu, run1=%zu)\n",
                run0.layers.size(), run1.layers.size());
        llama_free_model(model);
        llama_backend_free();
        return 1;
    }

    // Compare per layer. Snapshots are [n_embd, n_tokens=2] = 2 slots' rows.
    // For slot 0: first half (or stride 0 / 1 depending on layout).
    // For slot 1: second half.
    // We'll just compare full snapshot row-by-row, but report which slot
    // diverged by splitting the n_floats in half.
    fprintf(stderr, "\n[CY.F.18] per-layer cross-run divergence (run0 vs run1):\n");
    int first_div_layer_slot0 = -1;
    int first_div_layer_slot1 = -1;
    for (size_t li = 0; li < run0.layers.size(); ++li) {
        const auto & a = run0.layers[li];
        const auto & b = run1.layers[li];
        if (a.n_floats != b.n_floats) {
            fprintf(stderr, "  layer %d: SIZE MISMATCH (a=%d b=%d) — skipping\n",
                    a.layer_id, a.n_floats, b.n_floats);
            continue;
        }
        const int half = a.n_floats / 2;
        int s0_diffs = 0, s1_diffs = 0;
        float s0_max = 0.0f, s1_max = 0.0f;
        // dflash_extract layout is [n_embd, n_tokens] row-major:
        // index = embd * n_tokens + tok. For n_tokens=2: slot 0 at even
        // indices (tok=0), slot 1 at odd indices (tok=1).
        for (int j = 0; j < a.n_floats; ++j) {
            uint32_t ua, ub;
            std::memcpy(&ua, &a.data[j], 4);
            std::memcpy(&ub, &b.data[j], 4);
            if (ua != ub) {
                float d = std::fabs(a.data[j] - b.data[j]);
                if ((j & 1) == 0) {
                    ++s0_diffs;
                    if (d > s0_max) s0_max = d;
                } else {
                    ++s1_diffs;
                    if (d > s1_max) s1_max = d;
                }
            }
        }
        fprintf(stderr, "  layer %2d: slot0=%5d/%d diffs max|Δ|=%.3e   slot1=%5d/%d diffs max|Δ|=%.3e\n",
                a.layer_id,
                s0_diffs, half, s0_max,
                s1_diffs, half, s1_max);
        if (s0_diffs > 0 && first_div_layer_slot0 < 0) first_div_layer_slot0 = a.layer_id;
        if (s1_diffs > 0 && first_div_layer_slot1 < 0) first_div_layer_slot1 = a.layer_id;
    }
    fprintf(stderr, "\n[CY.F.18] First divergent layer:\n");
    fprintf(stderr, "  slot 0: %s (expected: -1, no race)\n",
            first_div_layer_slot0 < 0 ? "(none)" : (std::to_string(first_div_layer_slot0)).c_str());
    fprintf(stderr, "  slot 1: %s\n",
            first_div_layer_slot1 < 0 ? "(none)" : (std::to_string(first_div_layer_slot1)).c_str());

    llama_free_model(model);
    llama_backend_free();
    return 0;
}
