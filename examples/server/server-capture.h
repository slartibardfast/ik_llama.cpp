// server-capture.h — tensor-dump cb_eval hook for llama-server, used to
// localise NP-determinism divergences that surface only in the server's
// continuous-batching path (R5 task #95, 2026-05-19). Output format
// mirrors examples/llama-state-capture/ so scripts/compare-intra-layer.py
// consumes it directly.
//
// Activation:
//   LLAMA_SERVER_CAPTURE_DIR=/path/to/out  — required, opt-in
//   LLAMA_SERVER_CAPTURE_TENSORS=q,k,v,... — comma-separated name prefixes
//   LLAMA_SERVER_CAPTURE_LAYERS=0,1,2,3    — comma-separated layer ids or "all"
//
// Wiring (server-context.cpp::load_model):
//   server_capture::install_from_env(params_base);   // before llama_init_from_gpt_params
//
// Per-tick label (server-context.cpp::update_slots):
//   server_capture::set_phase("tick-NNNN-bN-tT");    // before each llama_decode
//
// Lifecycle: state lives in a file-static, freed by an atexit hook that
// flushes manifest.json. cb_eval runs single-threaded on the eval thread
// so no synchronisation is needed inside the hook.
//
// DIAGNOSTIC ONLY — delete after the R5 bug is closed
// (per feedback_bake_measurement_env_gates spirit).

#pragma once

#include "common.h"
#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace server_capture {

struct state {
    std::unordered_set<std::string> name_prefixes;
    std::set<int>                   layers;
    bool                            all_layers = false;
    std::string                     out_dir;
    std::string                     phase_label = "init";
    std::unordered_map<std::string, int> ubatch_counter;
    int                             order_idx = 0;
    std::vector<std::string>        manifest_records;
    int                             n_captured = 0;
    int                             n_skipped_type = 0;
};

inline state & singleton() {
    static state s;
    return s;
}

inline std::vector<std::string> split_csv(const std::string & s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t a = tok.find_first_not_of(" \t");
        size_t b = tok.find_last_not_of(" \t");
        if (a == std::string::npos) continue;
        out.push_back(tok.substr(a, b - a + 1));
    }
    return out;
}

// Parse "{prefix}-{N}" into (prefix, layer). Layerless tensors return -1.
inline bool parse_name(const std::string & name, std::string & prefix, int & layer) {
    size_t dash = name.find_last_of('-');
    if (dash == std::string::npos) { prefix = name; layer = -1; return true; }
    const char * tail = name.c_str() + dash + 1;
    if (!*tail) { prefix = name; layer = -1; return true; }
    for (const char * c = tail; *c; ++c) {
        if (*c < '0' || *c > '9') { prefix = name; layer = -1; return true; }
    }
    prefix = name.substr(0, dash);
    layer  = std::atoi(tail);
    return true;
}

inline void finalize() {
    state & st = singleton();
    if (st.out_dir.empty() || st.manifest_records.empty()) return;
    const std::string mfpath = st.out_dir + "/manifest.json";
    std::ofstream m(mfpath);
    if (!m) {
        std::fprintf(stderr, "[server-capture] failed to open %s\n", mfpath.c_str());
        return;
    }
    m << "[\n";
    for (size_t i = 0; i < st.manifest_records.size(); ++i) {
        m << "  " << st.manifest_records[i];
        if (i + 1 < st.manifest_records.size()) m << ",";
        m << "\n";
    }
    m << "]\n";
    std::fprintf(stderr, "[server-capture] wrote %s: %d records, %d captured, %d skipped\n",
                 mfpath.c_str(), (int) st.manifest_records.size(),
                 st.n_captured, st.n_skipped_type);
    st.manifest_records.clear();
}

inline bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    state & st = *(state *) user_data;
    if (!t || !t->name[0]) return false;

    std::string prefix;
    int layer = -1;
    parse_name(t->name, prefix, layer);

    if (!st.all_layers) {
        if (layer < 0) return false;
        if (st.layers.find(layer) == st.layers.end()) return false;
    }
    if (!st.name_prefixes.empty() &&
        st.name_prefixes.find(prefix) == st.name_prefixes.end()) {
        return false;
    }

    if (ask) return true;
    if (t->buffer == nullptr) return true;

    const size_t   nbytes     = ggml_nbytes(t);
    const int64_t  n_elements = (int64_t) ggml_nelements(t);

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
        st.n_skipped_type++;
        return true;
    }

    std::string base = t->name;
    int ub = st.ubatch_counter[base]++;
    int oi = st.order_idx++;
    char fname[512];
    if (layer >= 0) {
        std::snprintf(fname, sizeof(fname), "%s/layer%02d/%s.ub%d.bin",
                      st.phase_label.c_str(), layer, base.c_str(), ub);
    } else {
        std::snprintf(fname, sizeof(fname), "%s/no-layer/%s.ub%d.bin",
                      st.phase_label.c_str(), base.c_str(), ub);
    }
    std::string path = st.out_dir + "/" + fname;
    {
        std::string dir = path.substr(0, path.find_last_of('/'));
        std::string mkcmd = "mkdir -p '" + dir + "'";
        (void) std::system(mkcmd.c_str());
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[server-capture] failed to open %s\n", path.c_str());
        return true;
    }
    f.write((const char *) f32.data(), f32.size() * sizeof(float));

    char rec[1536];
    std::snprintf(rec, sizeof(rec),
        "{\"prompt_id\":\"server\",\"phase\":\"%s\",\"order\":%d,"
        "\"name\":\"%s\",\"prefix\":\"%s\",\"layer\":%d,"
        "\"shape\":[%lld,%lld,%lld,%lld],\"orig_dtype\":\"%s\","
        "\"n_seq_max\":1,\"ubatch_idx\":%d,\"file\":\"%s\"}",
        st.phase_label.c_str(), oi,
        base.c_str(), prefix.c_str(), layer,
        (long long) t->ne[0], (long long) t->ne[1],
        (long long) t->ne[2], (long long) t->ne[3],
        ggml_type_name(t->type), ub, fname);
    st.manifest_records.emplace_back(rec);
    st.n_captured++;
    return true;
}

// Update the phase label and reset per-phase counters. Call before each
// llama_decode in the server's tick loop.
inline void set_phase(const std::string & label) {
    state & st = singleton();
    if (st.out_dir.empty()) return;
    st.phase_label = label;
    st.ubatch_counter.clear();
    st.order_idx = 0;
}

// Returns true if capture is active (LLAMA_SERVER_CAPTURE_DIR is set).
inline bool active() {
    return !singleton().out_dir.empty();
}

// Install the hook into gpt_params before llama_init_from_gpt_params.
// Reads LLAMA_SERVER_CAPTURE_{DIR,TENSORS,LAYERS}. Idempotent.
inline void install_from_env(gpt_params & params) {
    state & st = singleton();
    if (!st.out_dir.empty()) return;  // already installed
    const char * dir_env = std::getenv("LLAMA_SERVER_CAPTURE_DIR");
    if (!dir_env || !*dir_env) return;
    st.out_dir = dir_env;
    {
        std::string mkcmd = "mkdir -p '" + st.out_dir + "'";
        (void) std::system(mkcmd.c_str());
    }
    if (const char * te = std::getenv("LLAMA_SERVER_CAPTURE_TENSORS")) {
        for (auto & p : split_csv(te)) st.name_prefixes.insert(p);
    }
    if (const char * le = std::getenv("LLAMA_SERVER_CAPTURE_LAYERS")) {
        std::string s = le;
        if (s == "all") st.all_layers = true;
        else for (auto & x : split_csv(s)) st.layers.insert(std::atoi(x.c_str()));
    }
    params.cb_eval           = cb_eval;
    params.cb_eval_user_data = &st;
    std::atexit([] { finalize(); });
    std::fprintf(stderr, "[server-capture] active: out_dir=%s tensors=%zu layers=%s\n",
                 st.out_dir.c_str(), st.name_prefixes.size(),
                 st.all_layers ? "all" : ("count=" + std::to_string(st.layers.size())).c_str());
}

}  // namespace server_capture
