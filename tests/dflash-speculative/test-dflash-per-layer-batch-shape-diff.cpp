// test-dflash-per-layer-batch-shape-diff.cpp
//
// L4 — per-layer batch-shape divergence localiser. After L3' bound the
// variance threshold to n_tokens=2 and ruled out the delta_net
// graph-build conditional (via a forced-branch diagnostic), L4
// captures each transformer layer's residual stream ("l_out-<il>")
// for two paths at the SAME positions and diffs layer-by-layer to
// identify the smallest layer index where the two paths diverge.
//
// Paths:
//   A — two sequential 1-token decodes at positions [P, P+1].
//       Each decode produces one row of l_out per layer.
//   B — one 2-token batch at positions [P, P+1].
//       Produces two rows of l_out per layer.
//
// Both paths feed the SAME tokens at the SAME positions. The model
// state at the END of each path is logically identical. Per-layer
// l_out for row 0 (at position P) and row 1 (at position P+1) must
// be byte-identical between paths.
//
// The DFlash extract mechanism (cb_eval hook installed by
// llama_set_dflash_extract_layers) captures l_out-<il> for the
// configured layer indices, accumulating rows across decodes per
// seq_id. We arm extract on a wide set spanning the full layer
// stack and compare byte-by-byte.
//
// First-divergent layer index narrows the bug to whatever op runs
// between the previous (matching) layer's l_out and this layer's
// l_out — typically RMS norm, attn (FA or DeltaNet), MLP, residual
// add, in that order.
//
// Env:
//   LLAMA_TEST_TARGET — target GGUF (skip 77 if unset).
//
// Optional env:
//   LLAMA_TEST_PROMPT — override prompt.
//   LLAMA_TEST_LAYERS — comma-separated layer ids (default sweeps
//                       every 4 layers across the 62-layer stack).

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

static std::vector<int32_t> parse_layers_csv(const char * csv, int n_transformer_layer) {
    std::vector<int32_t> out;
    if (!csv || !*csv) {
        // Default: every 4 layers across [0, n_transformer_layer).
        // The MTP/nextn layer at the top of the stack doesn't emit
        // l_out-il via the cb_eval hook used by extract.
        for (int i = 0; i < n_transformer_layer; i += 4) out.push_back((int32_t) i);
        if (out.empty() || out.back() != n_transformer_layer - 1) {
            out.push_back((int32_t)(n_transformer_layer - 1));
        }
        return out;
    }
    const char * p = csv;
    while (*p) {
        char * end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p) break;
        if (v >= 0 && v < n_transformer_layer) out.push_back((int32_t) v);
        p = end;
        while (*p == ',' || *p == ' ') ++p;
    }
    if (out.empty()) {
        for (int i = 0; i < n_transformer_layer; i += 4) out.push_back((int32_t) i);
    }
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
    const int n_embd  = llama_model_n_embd(model);
    const int n_layer = llama_n_layer(model);
    const int n_nextn = llama_model_n_nextn_layer(model);
    const int n_transformer_layer = n_layer - n_nextn;
    std::fprintf(stderr, "[L4] n_embd=%d n_layer=%d nextn=%d → transformer_layers=%d\n",
                 n_embd, n_layer, n_nextn, n_transformer_layer);

    const std::string prompt = []() {
        const char * env = std::getenv("LLAMA_TEST_PROMPT");
        if (env && *env) return std::string(env);
        return std::string(
            "The capital of France is Paris. The capital of Germany is");
    }();

    std::vector<int32_t> layers = parse_layers_csv(std::getenv("LLAMA_TEST_LAYERS"), n_transformer_layer);
    std::fprintf(stderr, "[L4] extracting layers:");
    for (int32_t il : layers) std::fprintf(stderr, " %d", (int) il);
    std::fprintf(stderr, " (%zu total)\n", layers.size());

    // Tokenise prompt + get the first sampled token (T_0) and second
    // sampled token (T_1) from a baseline run.
    std::vector<llama_token> prompt_tokens;
    llama_token T_0, T_1;
    {
        llama_context * ctx = fresh_ctx(model);
        if (!ctx) { llama_free_model(model); std::fprintf(stderr, "tmp ctx fail\n"); return 77; }
        prompt_tokens = common_tokenize(ctx, prompt, true, true);
        if ((int) prompt_tokens.size() < 4) {
            std::fprintf(stderr, "[L4] prompt too short\n");
            llama_free(ctx); llama_free_model(model);
            return 1;
        }
        if (decode_tokens(ctx, prompt_tokens, 0, 0) != 0) {
            std::fprintf(stderr, "[tmp] prefill fail\n");
            llama_free(ctx); llama_free_model(model); return 1;
        }
        // T_0 = argmax of last prompt row.
        const float * gl = llama_get_logits(ctx);
        const int P_ = (int) prompt_tokens.size();
        const float * row_last = gl + (size_t)(P_ - 1) * (size_t) n_vocab;
        int best = 0; float bv = row_last[0];
        for (int v = 1; v < n_vocab; ++v) {
            if (row_last[v] > bv) { bv = row_last[v]; best = v; }
        }
        T_0 = (llama_token) best;
        // Decode T_0 and get T_1.
        if (decode_tokens(ctx, {T_0}, P_, 0) != 0) {
            std::fprintf(stderr, "[tmp] T0 decode fail\n");
            llama_free(ctx); llama_free_model(model); return 1;
        }
        const float * gl2 = llama_get_logits(ctx);
        best = 0; bv = gl2[0];
        for (int v = 1; v < n_vocab; ++v) {
            if (gl2[v] > bv) { bv = gl2[v]; best = v; }
        }
        T_1 = (llama_token) best;
        llama_free(ctx);
    }
    const llama_pos P = (llama_pos) prompt_tokens.size();
    std::fprintf(stderr, "[L4] P=%d T_0=%d T_1=%d\n", (int) P, (int) T_0, (int) T_1);

    // Capture helper: arm extract on `layers`, run the path's decodes,
    // pull each layer's extract buffer into `out_layers[slot]`. Both
    // paths share the same extract-layers arming so slot indexing
    // matches.
    auto run_path = [&](bool path_B, std::vector<std::vector<float>> & out_layers) -> int {
        llama_context * ctx = fresh_ctx(model);
        if (!ctx) return 1;
        llama_set_dflash_extract_layers(ctx, layers.data(), (int32_t) layers.size());

        if (decode_tokens(ctx, prompt_tokens, 0, 0) != 0) {
            std::fprintf(stderr, "[%c] prefill fail\n", path_B ? 'B' : 'A');
            llama_free(ctx);
            return 1;
        }
        // Prefill produced P rows in the extract buffer. After the
        // path's decodes the buffer will hold P+2 rows total; we
        // pull all P+2 below and slice the last 2 (post-prefill) rows
        // for comparison. No trim needed.

        if (!path_B) {
            // Path A: two sequential 1-token decodes.
            if (decode_tokens(ctx, {T_0}, P, 0) != 0) {
                std::fprintf(stderr, "[A] decode T_0 fail\n");
                llama_free(ctx); return 1;
            }
            if (decode_tokens(ctx, {T_1}, P + 1, 0) != 0) {
                std::fprintf(stderr, "[A] decode T_1 fail\n");
                llama_free(ctx); return 1;
            }
        } else {
            // Path B: one 2-token batch.
            if (decode_tokens(ctx, {T_0, T_1}, P, 0) != 0) {
                std::fprintf(stderr, "[B] 2-token decode fail\n");
                llama_free(ctx); return 1;
            }
        }

        // Pull each layer's extract buffer. Both paths produce
        // P + 2 rows total (prefill + 2 new tokens). The new rows
        // are at indices [P, P+2).
        out_layers.assign(layers.size(), {});
        for (size_t s = 0; s < layers.size(); ++s) {
            const size_t need_floats = (size_t)(P + 2) * (size_t) n_embd;
            std::vector<float> buf(need_floats, 0.0f);
            const size_t got = llama_get_dflash_extract_data_seq(ctx, (int32_t) s, 0,
                                                                  buf.data(), buf.size());
            if (got < need_floats) {
                std::fprintf(stderr,
                             "[%c slot=%zu il=%d] got=%zu < needed=%zu — extract not populated\n",
                             path_B ? 'B' : 'A', s, (int) layers[s], got, need_floats);
                llama_free(ctx);
                return 1;
            }
            // Keep only the last 2 rows (the post-prefill ones).
            out_layers[s].assign(buf.begin() + (size_t) P * (size_t) n_embd,
                                 buf.begin() + (size_t)(P + 2) * (size_t) n_embd);
        }
        llama_free(ctx);
        return 0;
    };

    std::vector<std::vector<float>> A_layers, B_layers;
    if (run_path(false, A_layers) != 0) {
        llama_free_model(model); llama_backend_free();
        return 1;
    }
    if (run_path(true,  B_layers) != 0) {
        llama_free_model(model); llama_backend_free();
        return 1;
    }
    std::fprintf(stderr, "[L4] both paths captured %zu layers × 2 rows × %d floats\n",
                 layers.size(), n_embd);

    // Diff per layer, per row.
    std::printf("\n=== L4 per-layer per-row diff ===\n");
    std::printf("layer | row0_diffs | row1_diffs | first_diff_layer_so_far\n");
    int first_divergent_slot = -1;
    int first_divergent_row  = -1;
    int total_mismatches     = 0;
    for (size_t s = 0; s < layers.size(); ++s) {
        const float * a0 = A_layers[s].data();
        const float * b0 = B_layers[s].data();
        const float * a1 = a0 + n_embd;
        const float * b1 = b0 + n_embd;
        int diff0 = 0, diff1 = 0;
        float max_abs_0 = 0.0f, max_abs_1 = 0.0f;
        for (int i = 0; i < n_embd; ++i) {
            uint32_t ai, bi;
            std::memcpy(&ai, &a0[i], 4); std::memcpy(&bi, &b0[i], 4);
            if (ai != bi) { ++diff0; const float d = std::abs(a0[i] - b0[i]); if (d > max_abs_0) max_abs_0 = d; }
            std::memcpy(&ai, &a1[i], 4); std::memcpy(&bi, &b1[i], 4);
            if (ai != bi) { ++diff1; const float d = std::abs(a1[i] - b1[i]); if (d > max_abs_1) max_abs_1 = d; }
        }
        if ((diff0 > 0 || diff1 > 0) && first_divergent_slot < 0) {
            first_divergent_slot = (int) s;
            first_divergent_row  = (diff0 > 0) ? 0 : 1;
        }
        total_mismatches += diff0 + diff1;
        std::printf("  il=%d : row0=%d/%d (max|Δ|=%.3e), row1=%d/%d (max|Δ|=%.3e)\n",
                    (int) layers[s], diff0, n_embd, max_abs_0,
                    diff1, n_embd, max_abs_1);
    }

    if (total_mismatches == 0) {
        std::printf("[PASS] all extracted layers byte-identical between path A and "
                    "path B. Batch-shape variance does NOT appear in any of the "
                    "%zu sampled layers — needs a denser layer sweep or maybe the "
                    "divergence is in a non-l_out tensor (norm output, attn output, "
                    "MLP intermediate).\n",
                    layers.size());
        llama_free_model(model); llama_backend_free();
        return 0;
    }
    std::printf("[FAIL-DIAG] first-divergent extract slot=%d (layer il=%d, row=%d). "
                "Bug surface: somewhere between layer %d's l_out and layer %d's l_out — "
                "the attn block + MLP + residuals INSIDE layer %d, OR an upstream op "
                "between the prior extracted layer and this one.\n",
                first_divergent_slot, (int) layers[first_divergent_slot],
                first_divergent_row,
                first_divergent_slot > 0 ? (int) layers[first_divergent_slot - 1] : -1,
                (int) layers[first_divergent_slot],
                (int) layers[first_divergent_slot]);
    llama_free_model(model); llama_backend_free();
    return 1;
}
