// test-cy-layer0-capture.cpp
//
// Phase CY.B.1 — capture ALL named intermediates inside layer 0 (DeltaNet) at
// NP=1 vs NP=4 to find the first divergent tag.
//
// d1-capture already shows l_out-0 differs by max|Δ|≈1.118e-07 (single fp32
// ULP at exp 0) between NP=1 and NP=4 slot-0. This test drills inside layer 0
// to pinpoint the op that introduces the gap.
//
// At NP=1 (all_same_seq=true) the DeltaNet fast path takes ONE call to
// build_layer_attn_linear_core with ne[1]=1. At NP=4 (all_same_seq=false) the
// slow path splits the batch into 4 per-seq blocks, each ne[1]=1, concatenated.
// The slot-0 sub-graph at NP=4 SHOULD produce the same output as the NP=1
// single call — but the d1 data says it doesn't.
//
// Captures every cb-tagged tensor at layer 0 (il=0). For the multi-device-split
// path, il_cb = 1000*il+id, so tags like "qkv_mixed-0" (dev 0) and
// "qkv_mixed-1" (dev 1) both apply. We also capture layer-0 ffn_* tags and
// l_out-0.
//
// Env:
//   LLAMA_TEST_TARGET     — target GGUF
//   LLAMA_TEST_PROMPT_DIR — prompts dir with p0.txt..pN-1.txt (same content)
//   LLAMA_TEST_NP         — slot count (1..8)
//   LLAMA_TEST_OUT_DIR    — output directory (must exist)

#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct captured {
    std::string name;
    std::vector<int64_t> ne;
    ggml_type type;
    std::vector<uint8_t> data;
};

struct cap_state {
    // Layer-0 tags. We accept device-0 (il_cb=0 == il) and device-1 (il_cb=1)
    // suffixes since split-mode=GRAPH gives both. We also accept just "-0" for
    // tags that use il (not il_cb).
    std::vector<std::string> match_names;
    std::vector<captured> store;
};

static bool name_matches(const std::vector<std::string> & needles, const char * name) {
    for (const auto & n : needles) {
        if (std::strcmp(n.c_str(), name) == 0) return true;
    }
    return false;
}

static bool my_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (cap_state *) user_data;
    if (!st) return false;
    if (!name_matches(st->match_names, t->name)) return false;
    if (ask) return true;
    if (!t->buffer) return true;
    captured c;
    c.name = t->name;
    c.ne   = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    // Match d1-capture mechanism's dtype branching (libllama/llama.cpp:9682):
    // residual / FFN tensors land as F32 for single-token decodes and F16 for
    // multi-token ubatches. Without dtype branching, ggml_nbytes-based copies
    // of F16 tensors are silently misinterpreted as F32 in downstream reads.
    const int64_t n_elements = (int64_t) ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        c.type = GGML_TYPE_F32;
        c.data.resize((size_t) n_elements * sizeof(float));
        ggml_backend_tensor_get(t, c.data.data(), 0, c.data.size());
    } else if (t->type == GGML_TYPE_F16) {
        c.type = GGML_TYPE_F32;  // store as F32 after conversion (matches d1)
        std::vector<ggml_fp16_t> h_stage((size_t) n_elements);
        ggml_backend_tensor_get(t, h_stage.data(), 0, (size_t) n_elements * sizeof(ggml_fp16_t));
        c.data.resize((size_t) n_elements * sizeof(float));
        ggml_fp16_to_fp32_row(h_stage.data(), (float *) c.data.data(), n_elements);
    } else {
        // Other types — write raw bytes; downstream must interpret.
        c.type = t->type;
        const size_t nb = ggml_nbytes(t);
        c.data.resize(nb);
        ggml_backend_tensor_get(t, c.data.data(), 0, nb);
    }
    st->store.push_back(std::move(c));
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

static void write_bin(const std::string & path,
                      const captured & c,
                      const std::string & ctype) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char *) c.data.data(), (std::streamsize) c.data.size());
    f.close();
    std::ofstream m(path + ".json");
    m << "{\"name\":\"" << c.name << "\",\"type\":\"" << ctype
      << "\",\"ne\":[" << c.ne[0] << "," << c.ne[1] << ","
      << c.ne[2] << "," << c.ne[3] << "]}";
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

    cap_state st;
    // For layer 0 (DeltaNet), build_layer_attn_linear_core dispatches per
    // device. cb() uses both il (non-device-aware) and il_cb=1000*il+id (per
    // device). For il=0: il_cb=0 (dev 0) and il_cb=1 (dev 1). Match both.
    // Also capture l_out-0 (final layer-0 residual).
    // Capture layers 0 and 6 (first divergent layer per d1 sweep).
    // For multi-device-split DeltaNet: il_cb = 1000*il + dev_id.
    // For il=0: il_cb ∈ {0, 1}. For il=6: il_cb ∈ {6000, 6001}.
    auto add_layer = [&](int il) {
        // build_qkv (and build_fused_delta_net inside it) receives `il` from
        // its callers (build_layer_attn_linear_core line 569 passes `il` not
        // il_cb). So DeltaNet kernel internal tags use plain "-<il>" suffix.
        std::vector<std::string> per_il_bases = {
            "q_in","k_in","v_in","beta_in","g_in","state_in",
            "q_fused","k_fused","v_fused","g_fused","beta_fused","state_fused",
            "delta_net_fused_raw","output_tokens","new_state",
            "q","k","v","query_flat","key","value",
            "q_conv_normed","k_conv_normed","attn_output",
        };
        char ilstr[16];
        snprintf(ilstr, sizeof(ilstr), "-%d", il);
        for (auto & b : per_il_bases) st.match_names.push_back(b + ilstr);

        // build_qkvz (line 510) and FFN-inside-per-device (line 779) pass
        // il_cb = 1000*il + dev to cb. These tags use "-<il_cb>" suffix.
        auto suff_cb = [il](int dev) {
            char buf[32];
            int il_cb = 1000*il + dev;
            snprintf(buf, sizeof(buf), "-%d", il_cb);
            return std::string(buf);
        };
        std::vector<std::string> per_ilcb_bases = {
            "qkv_mixed","z","linear_attn_qkv_mixed","linear_attn_mixed_qkvz",
            "ffn_norm","ffn_up","ffn_gate","ffn_silu","ffn_down","ffn_up_gate",
            "ffn_with_extra","inp_normed","norm",
        };
        for (auto & b : per_ilcb_bases) {
            st.match_names.push_back(b + suff_cb(0));
            st.match_names.push_back(b + suff_cb(1));
        }
        // Per-layer (non-device-aware) residual tags use just "-<il>"
        std::vector<std::string> per_layer_bases = {
            "l_out","ffn_combined","ffn_with_inp","ffn_out_with_inp","attn_combined",
        };
        for (auto & b : per_layer_bases) st.match_names.push_back(b + ilstr);
    };
    add_layer(0);
    add_layer(5);   // last byte-identical layer (NP=1 vs NP=N≥2)
    add_layer(6);   // first divergent layer (NP=1 vs NP=N≥2)
    add_layer(7);   // FA layer that amplifies
    add_layer(19);  // FA layer immediately before NP=8 boundary (NP≤4 vs NP=8 byte-identical here)
    add_layer(20);  // first divergent layer for NP≤4 vs NP=8
    add_layer(21);  // amplification check
    st.match_names.push_back("inp_embd");

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = 4096;
    cparams.n_batch     = 2048;
    cparams.n_ubatch    = 512;
    cparams.n_seq_max   = (uint32_t) N;
    cparams.type_k      = GGML_TYPE_Q4_0;
    cparams.type_v      = GGML_TYPE_Q4_0;
    cparams.flash_attn  = true;
    cparams.mla_attn    = 3;
    cparams.k_cache_hadamard = true;
    cparams.v_cache_hadamard = true;
    cparams.cb_eval     = my_cb_eval;
    cparams.cb_eval_user_data = (void *) &st;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { llama_free_model(model); fprintf(stderr, "ctx init failed\n"); return 1; }

    const int n_vocab = llama_n_vocab(model);

    std::vector<llama_token> base_tokens;
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/p0.txt", pd);
        std::ifstream f(path);
        std::stringstream ss; ss << f.rdbuf();
        base_tokens = common_tokenize(ctx, ss.str(), true, true);
        if (base_tokens.empty()) {
            fprintf(stderr, "empty prompt at %s\n", path);
            return 1;
        }
    }
    const int n_prompt = (int) base_tokens.size();
    fprintf(stderr, "[CY.B.1] N=%d  n_prompt=%d\n", N, n_prompt);

    std::vector<llama_token> id_last(N);
    std::vector<llama_pos>   next_pos(N);

    for (int s = 0; s < N; s++) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++) {
            bool last = (i == n_prompt - 1);
            common_batch_add(batch, base_tokens[i], i, {(llama_seq_id) s}, last);
        }
        st.store.clear();
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[CY.B.1] slot %d: prefill failed\n", s);
            llama_batch_free(batch);
            return 1;
        }
        float * logits = llama_get_logits_ith(ctx, n_prompt - 1);
        if (!logits) { return 1; }
        id_last[s]  = greedy_argmax(logits, n_vocab);
        next_pos[s] = (llama_pos) n_prompt;
        llama_batch_free(batch);
    }

    st.store.clear();

    llama_batch batch = llama_batch_init(N, 0, 1);
    for (int s = 0; s < N; s++) {
        common_batch_add(batch, id_last[s], next_pos[s], {(llama_seq_id) s}, true);
    }
    fprintf(stderr, "[CY.B.1] decode N=%d tokens (one per slot, same content)\n", N);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "[CY.B.1] decode failed\n");
        llama_batch_free(batch);
        return 1;
    }
    llama_batch_free(batch);

    fprintf(stderr, "[CY.B.1] captured %zu tensors\n", st.store.size());

    // Write each capture with an instance index (occurrence within st.store)
    // since the multi-block DeltaNet path at all_same_seq=false emits the same
    // tag name multiple times (once per block). Order in st.store == order of
    // graph evaluation, so instance 0 corresponds to block 0 / slot 0.
    {
        std::vector<int> seen_count; // parallel to store, count of prior identical names
        std::vector<int> name_idx;   // for each entry, its occurrence index
        name_idx.resize(st.store.size(), 0);
        for (size_t i = 0; i < st.store.size(); ++i) {
            int cnt = 0;
            for (size_t j = 0; j < i; ++j) {
                if (st.store[j].name == st.store[i].name) cnt++;
            }
            name_idx[i] = cnt;
        }
        for (size_t i = 0; i < st.store.size(); ++i) {
            const auto & c = st.store[i];
            char path[1024];
            snprintf(path, sizeof(path), "%s/cy-np%d-%s.inst%d.bin",
                     out_dir, N, c.name.c_str(), name_idx[i]);
            write_bin(path, c, ggml_type_name(c.type));
            fprintf(stderr, "  %s inst%d [%lld,%lld,%lld,%lld] type=%s nbytes=%zu\n",
                    c.name.c_str(), name_idx[i],
                    (long long)c.ne[0], (long long)c.ne[1],
                    (long long)c.ne[2], (long long)c.ne[3],
                    ggml_type_name(c.type),
                    c.data.size());
        }
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
