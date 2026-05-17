// llama-state-capture: dump every named-graph-tensor matching a name pattern
// at selected layers, for the audit foundation (task F.1).
//
// Output:
//   {out_dir}/manifest.json — array of records {prompt_id, name, layer, shape,
//                              dtype, n_seq_max, ubatch_idx, file}
//   {out_dir}/{file}        — raw fp32 row-major data (auto-converted from
//                              F16/BF16/Q4_0 if encountered)
//
// Matches names of the form "{prefix}-{N}" where {prefix} is in --tensors
// and {N} is in --layers (or --layers=all to match any).
//
// Usage:
//   llama-state-capture -m MODEL --prompt-file PROMPT \
//       --tensors q,k,v,l_out,Qcur,Kcur_hadamard,Vcur,kqv_out \
//       --layers 0,15,31,47,63 \
//       --np 2 \
//       --out-dir data/capture/{prompt_id}/
//
// Pass-through args (-ngl, --split-mode, --tensor-split, --device, --ctx-size,
// --batch-size, --ubatch-size, --cache-type-k/v, --k-cache-hadamard,
// --v-cache-hadamard) are forwarded to gpt_params_parse for parity with the
// production server config.

#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

struct capture_state {
    std::unordered_set<std::string> name_prefixes;
    std::set<int>                   layers;
    bool                            all_layers = false;
    bool                            all_in_layer = false; // bypass prefix filter
    bool                            decode_only = false;  // skip prefill ubatches
    std::string                     out_dir;
    int                             n_seq_max = 1;
    std::string                     prompt_id;
    // Execution-phase tracking. set_phase() updates `phase_label` (used in
    // filenames + manifest) and resets per-phase counters so each phase has
    // its own ubatch_idx sequence keyed by tensor name.
    std::string                     phase_label = "prefill";
    bool                            in_decode   = false;
    int                             decode_step_idx = -1;
    std::unordered_map<std::string, int> ubatch_counter;
    // Per-(phase, name) order index — increments in cb_eval fire order so the
    // manifest preserves the actual execution sequence for the diff walker.
    int                             order_idx = 0;
    std::vector<std::string>        manifest_records;
    int                             n_captured = 0;
    int                             n_skipped_type = 0;
};

static void capture_set_phase(capture_state * st, const std::string & label, bool decode, int step) {
    st->phase_label   = label;
    st->in_decode     = decode;
    st->decode_step_idx = step;
    st->ubatch_counter.clear();
    st->order_idx = 0;
}

static std::vector<std::string> split_csv(const std::string & s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        // trim
        size_t a = tok.find_first_not_of(" \t");
        size_t b = tok.find_last_not_of(" \t");
        if (a == std::string::npos) continue;
        out.push_back(tok.substr(a, b - a + 1));
    }
    return out;
}

// Parse name into (prefix, layer). Names look like "q-12" or "l_out-3"
// or "KQ_mask" (no layer; we treat layer = -1).
static bool parse_name(const std::string & name, std::string & prefix, int & layer) {
    size_t dash = name.find_last_of('-');
    if (dash == std::string::npos) {
        prefix = name;
        layer = -1;
        return true;
    }
    const char * tail = name.c_str() + dash + 1;
    // ensure tail is all digits
    if (!*tail) {
        prefix = name;
        layer = -1;
        return true;
    }
    for (const char * c = tail; *c; ++c) {
        if (*c < '0' || *c > '9') {
            prefix = name;
            layer = -1;
            return true;
        }
    }
    prefix = name.substr(0, dash);
    layer = std::atoi(tail);
    return true;
}

static bool capture_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (capture_state *) user_data;
    if (!t || !t->name[0]) return false;
    std::string prefix;
    int layer = -1;
    if (!parse_name(t->name, prefix, layer)) return false;

    // Phase gating — `--decode-only` skips prefill.
    if (st->decode_only && !st->in_decode) return false;

    // Layer gating — applies regardless of all_in_layer. Layer-less tensors
    // (layer == -1) are only captured when `all_layers` is set, since the
    // diagnostic target is intra-layer ops.
    if (!st->all_layers) {
        if (layer < 0)                                 return false;
        if (st->layers.find(layer) == st->layers.end()) return false;
    }

    // Prefix gating — bypassed by `--all-in-layer`.
    if (!st->all_in_layer && st->name_prefixes.find(prefix) == st->name_prefixes.end()) {
        return false;
    }

    if (ask) return true;
    if (t->buffer == nullptr) return true;

    const size_t   nbytes    = ggml_nbytes(t);
    const int64_t  n_elements = (int64_t) ggml_nelements(t);

    // Auto-convert to f32 for portability of downstream binders.
    std::vector<float> f32(n_elements);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, f32.data(), 0, nbytes);
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> stage(n_elements);
        ggml_backend_tensor_get(t, stage.data(), 0, nbytes);
        ggml_fp16_to_fp32_row(stage.data(), f32.data(), n_elements);
    } else if (t->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> stage(n_elements);
        ggml_backend_tensor_get(t, stage.data(), 0, nbytes);
        ggml_bf16_to_fp32_row(stage.data(), f32.data(), n_elements);
    } else {
        // Quantized / other types: skip. Audit's first cycle (FA, residual,
        // norm outputs) is fp32/fp16/bf16. Cache-write audits will need a
        // separate path that reads pre-quant tensors instead of the cache.
        st->n_skipped_type++;
        return true;
    }

    // Generate unique filename. Multi-ubatch prefill fires cb multiple times
    // per name; use ubatch counter to disambiguate. Counter is per-phase so
    // prefill ubatches and decode steps don't collide.
    std::string base = t->name;
    int ub = st->ubatch_counter[base]++;
    int oi = st->order_idx++;
    char fname[384];
    if (layer >= 0) {
        std::snprintf(fname, sizeof(fname), "%s/layer%02d/%s.ub%d.bin",
                      st->phase_label.c_str(), layer, base.c_str(), ub);
    } else {
        std::snprintf(fname, sizeof(fname), "%s/no-layer/%s.ub%d.bin",
                      st->phase_label.c_str(), base.c_str(), ub);
    }
    std::string path = st->out_dir + "/" + fname;
    {
        std::string dir = path.substr(0, path.find_last_of('/'));
        std::string mkcmd = "mkdir -p '" + dir + "'";
        if (std::system(mkcmd.c_str()) != 0) {
            fprintf(stderr, "[capture] mkdir failed for %s\n", dir.c_str());
            return true;
        }
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[capture] failed to open %s\n", path.c_str());
        return true;
    }
    f.write((const char *) f32.data(), f32.size() * sizeof(float));

    // Manifest entry — one JSON object per capture, comma-separated below.
    // `order` is the fire-order index within `phase`; manifest readers can
    // sort on (phase, order) to recover the actual op execution sequence.
    char rec[1280];
    std::snprintf(rec, sizeof(rec),
        "{\"prompt_id\":\"%s\",\"phase\":\"%s\",\"order\":%d,"
        "\"name\":\"%s\",\"prefix\":\"%s\",\"layer\":%d,"
        "\"shape\":[%lld,%lld,%lld,%lld],\"orig_dtype\":\"%s\","
        "\"n_seq_max\":%d,\"ubatch_idx\":%d,\"file\":\"%s\"}",
        st->prompt_id.c_str(), st->phase_label.c_str(), oi,
        base.c_str(), prefix.c_str(), layer,
        (long long)t->ne[0], (long long)t->ne[1],
        (long long)t->ne[2], (long long)t->ne[3],
        ggml_type_name(t->type), st->n_seq_max, ub, fname);
    st->manifest_records.emplace_back(rec);
    st->n_captured++;
    return true;
}

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "Usage: %s -m MODEL --prompt-file PROMPT --layers LAYERS --out-dir OUT_DIR\n"
        "       [--tensors PREFIXES | --all-in-layer] [--decode-only]\n"
        "       [--np N] [--prompt-id ID]\n"
        "  PREFIXES:      comma-separated tensor-name prefixes (e.g. q,k,v,l_out)\n"
        "  --all-in-layer bypass prefix filter; capture every named tensor at the\n"
        "                 listed layers (use with --decode-only to keep size sane).\n"
        "  --decode-only  skip prefill ubatches; only capture during the synthetic\n"
        "                 decode steps gated by LLAMA_CAPTURE_DECODE_STEPS=N.\n"
        "  LAYERS:        comma-separated layer indices, or 'all' for any layer\n"
        "  OUT_DIR:       output directory for .bin files + manifest.json. Files\n"
        "                 land under {OUT_DIR}/{phase}/layer{LL}/{name}.ub{N}.bin\n"
        "                 where phase is 'prefill' or 'decode-{step}'.\n"
        "  N:             n_seq_max (default 1); prompt is duplicated to all slots\n"
        "  ID:            short prompt identifier for manifest entries\n"
        "\nProduction-config pass-through args supported (-ngl, --device, etc.).\n",
        argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::string prompt_file_arg, tensors_arg, layers_arg, out_dir, prompt_id = "p0";
    int np = 1;
    bool all_in_layer = false;
    bool decode_only  = false;

    // Manually parse our own flags + filter argv for gpt_params_parse.
    std::vector<char *> filtered = { argv[0] };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char * flag) {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s requires a value\n", flag);
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (a == "--prompt-file") prompt_file_arg = need("--prompt-file");
        else if (a == "--tensors") tensors_arg = need("--tensors");
        else if (a == "--layers") layers_arg = need("--layers");
        else if (a == "--out-dir") out_dir = need("--out-dir");
        else if (a == "--np") np = std::atoi(need("--np").c_str());
        else if (a == "--prompt-id") prompt_id = need("--prompt-id");
        else if (a == "--all-in-layer") all_in_layer = true;
        else if (a == "--decode-only")  decode_only  = true;
        else filtered.push_back(argv[i]);
    }

    // With --all-in-layer the tensors arg is optional; otherwise required.
    if (prompt_file_arg.empty() || layers_arg.empty() || out_dir.empty()
        || (tensors_arg.empty() && !all_in_layer)) {
        print_usage(argv[0]);
        return 2;
    }

    // Load prompt text from file.
    std::string prompt_text;
    {
        std::ifstream f(prompt_file_arg);
        if (!f) {
            fprintf(stderr, "failed to open prompt file: %s\n", prompt_file_arg.c_str());
            return 2;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        prompt_text = ss.str();
    }

    // Parse gpt_params for model + GPU args.
    gpt_params params;
    int filt_argc = (int) filtered.size();
    if (!gpt_params_parse(filt_argc, filtered.data(), params)) {
        fprintf(stderr, "gpt_params_parse failed\n");
        return 2;
    }
    if (params.model.empty()) {
        fprintf(stderr, "model path required (-m)\n");
        return 2;
    }

    llama_backend_init();

    // Build capture state.
    capture_state st;
    st.out_dir      = out_dir;
    st.n_seq_max    = np;
    st.prompt_id    = prompt_id;
    st.all_in_layer = all_in_layer;
    st.decode_only  = decode_only;
    for (auto & p : split_csv(tensors_arg)) st.name_prefixes.insert(p);
    {
        std::string trimmed = layers_arg;
        if (trimmed == "all") st.all_layers = true;
        else for (auto & lstr : split_csv(layers_arg)) st.layers.insert(std::atoi(lstr.c_str()));
    }
    capture_set_phase(&st, "prefill", /*decode=*/false, /*step=*/-1);

    // Set cb_eval on gpt_params so llama_init_from_gpt_params propagates it
    // into the new context's cparams.
    params.cb_eval = capture_cb_eval;
    params.cb_eval_user_data = &st;
    params.n_parallel = np;

    llama_init_result init = llama_init_from_gpt_params(params);
    llama_model   * model = init.model;
    llama_context * ctx   = init.context;
    if (!model || !ctx) { fprintf(stderr, "failed to load model/ctx\n"); return 2; }

    // Tokenise + run single prefill across n_seq_max slots (same prompt in each).
    std::vector<llama_token> tokens = common_tokenize(ctx, prompt_text, true, true);
    fprintf(stderr, "[capture] prompt_id=%s n_tokens=%zu n_seq_max=%d tensors=%s layers=%s out=%s\n",
            prompt_id.c_str(), tokens.size(), np, tensors_arg.c_str(), layers_arg.c_str(),
            out_dir.c_str());

    llama_batch batch = llama_batch_init(np * (int) tokens.size(), 0, np);
    for (int sid = 0; sid < np; ++sid) {
        for (size_t i = 0; i < tokens.size(); ++i) {
            const bool last = (i + 1 == tokens.size());
            common_batch_add(batch, tokens[i], (llama_pos) i, { (llama_seq_id) sid }, last);
        }
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n");
        return 3;
    }
    llama_batch_free(batch);

    // Optional second decode step: --decode-steps N runs N additional
    // single-token decode steps (one greedy-sampled token per slot per
    // step). Captures fire for each step's named tensors; ubatch_counter
    // disambiguates per-call output files.
    const char * decode_steps_env = std::getenv("LLAMA_CAPTURE_DECODE_STEPS");
    int decode_steps = decode_steps_env ? std::atoi(decode_steps_env) : 0;
    if (decode_steps > 0) {
        // For determinism testing: use a fixed token (e.g., last prompt token)
        // for the synthetic decode, so the input is independent of any
        // sampling-step nondeterminism. We just want to compare kernel
        // outputs for the same input across NP. The position is
        // tokens.size() + step for each slot.
        const llama_token decode_tok = tokens.back();
        for (int step = 0; step < decode_steps; ++step) {
            char step_label[32];
            std::snprintf(step_label, sizeof(step_label), "decode-%d", step);
            capture_set_phase(&st, step_label, /*decode=*/true, /*step=*/step);
            llama_batch dbatch = llama_batch_init(np, 0, np);
            for (int sid = 0; sid < np; ++sid) {
                common_batch_add(dbatch, decode_tok,
                    (llama_pos)(tokens.size() + step),
                    { (llama_seq_id) sid }, true);
            }
            if (llama_decode(ctx, dbatch) != 0) {
                fprintf(stderr, "decode-step %d failed\n", step);
                llama_batch_free(dbatch);
                return 3;
            }
            llama_batch_free(dbatch);
        }
        fprintf(stderr, "[capture] ran %d decode steps after prefill\n",
                decode_steps);
    }

    // Write manifest.json.
    {
        std::string mkcmd = "mkdir -p '" + out_dir + "'";
        std::system(mkcmd.c_str());
        std::ofstream m(out_dir + "/manifest.json");
        m << "[\n";
        for (size_t i = 0; i < st.manifest_records.size(); ++i) {
            m << "  " << st.manifest_records[i];
            if (i + 1 < st.manifest_records.size()) m << ",";
            m << "\n";
        }
        m << "]\n";
    }
    fprintf(stderr, "[capture] done: %d tensors captured, %d skipped (unsupported dtype)\n",
            st.n_captured, st.n_skipped_type);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
