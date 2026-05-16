// test-cy-layer0-serial-vs-batched.cpp
//
// Phase CY.F.15 — drill into layer 0 to find which DeltaNet/FFN intermediate
// first differs between serial (1 seq, 12 tokens) and batched (8 seqs × 12
// tokens) processing for seq=0's row.
//
// CY.F.14 confirmed layer 0 is the first divergent layer (max|Δ|=2.4e-3).
// CY.F.9 confirmed serial-across-seqs is bit-identical (no slot-id bug).
// So the bug enters at layer 0 specifically when n_seqs > 1 in the batch.
//
// Strategy: install a cb_eval matched-name capture for layer-0 internal tags
// (use plain `il=0` since both build paths emit "q_in-0", "state_in-0",
// "delta_net_fused_raw-0", "qkv_mixed-0", etc.). Capture twice:
//   Phase A: serial — prefill seq=0 alone, 12 tokens
//   Phase B: batched — prefill seqs 0..7, 96 tokens; seq=0 rows are 0..11
//
// Compare each tag's seq=0-region row-by-row between serial and batched.
//
// Env:
//   LLAMA_TEST_TARGET    target GGUF
//   LLAMA_TEST_NP_BATCH  seqs in batched phase (default 8)
//   LLAMA_TEST_OUT_DIR   output dir for per-tag bins

#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

struct captured {
    std::string          name;
    std::vector<int64_t> ne;
    ggml_type            type;
    std::vector<float>   data;  // converted to f32 if F16
};

struct cap_state {
    std::vector<std::string> match_names;
    std::vector<captured>    store;
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
    c.type = t->type;
    const int64_t n_elements = (int64_t) ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        c.data.resize((size_t) n_elements);
        ggml_backend_tensor_get(t, c.data.data(), 0, (size_t) n_elements * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> stage((size_t) n_elements);
        ggml_backend_tensor_get(t, stage.data(), 0, (size_t) n_elements * sizeof(ggml_fp16_t));
        c.data.resize((size_t) n_elements);
        ggml_fp16_to_fp32_row(stage.data(), c.data.data(), n_elements);
    } else {
        return true;
    }
    st->store.push_back(std::move(c));
    return true;
}

static void write_per_tag(const cap_state & st, const std::string & out_dir,
                          const std::string & phase) {
    std::vector<int> counts;
    counts.reserve(st.store.size());
    for (size_t i = 0; i < st.store.size(); ++i) {
        int cnt = 0;
        for (size_t j = 0; j < i; ++j)
            if (st.store[j].name == st.store[i].name) cnt++;
        counts.push_back(cnt);
    }
    for (size_t i = 0; i < st.store.size(); ++i) {
        const auto & c = st.store[i];
        char path[1024];
        snprintf(path, sizeof(path), "%s/%s-%s.inst%d.bin",
                 out_dir.c_str(), phase.c_str(), c.name.c_str(), counts[i]);
        std::ofstream f(path, std::ios::binary);
        f.write((const char *) c.data.data(),
                (std::streamsize)(c.data.size() * sizeof(float)));
    }
    fprintf(stderr, "[%s] wrote %zu tags to %s\n",
            phase.c_str(), st.store.size(), out_dir.c_str());
}

int main() {
    const char * target  = std::getenv("LLAMA_TEST_TARGET");
    const char * out_dir = std::getenv("LLAMA_TEST_OUT_DIR");
    if (!target || !out_dir) {
        fprintf(stderr, "SKIP: set LLAMA_TEST_TARGET, LLAMA_TEST_OUT_DIR\n");
        return 77;
    }
    const int np_batch = std::getenv("LLAMA_TEST_NP_BATCH")
        ? std::atoi(std::getenv("LLAMA_TEST_NP_BATCH")) : 8;
    const std::string prompt =
        "The history of artificial intelligence began in earnest with the work of";

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    mparams.split_mode   = LLAMA_SPLIT_MODE_GRAPH;
    static const char * dev_csv = "CUDA0,CUDA1";
    mparams.devices = dev_csv;
    llama_model * model = llama_model_load_from_file(target, mparams);
    if (!model) { fprintf(stderr, "load failed\n"); return 77; }

    // Layer-0 capture tags. DeltaNet kernel internals use plain "-0" suffix
    // (per build_qkv/build_fused_delta_net which receive `il` not il_cb).
    // Projections (build_qkvz) use il_cb = 1000*0 + dev = 0 or 1.
    // l_out and ffn_with_inp use plain "-0".
    cap_state st;
    // NOTE on il_cb formulas (DIFFERENT between DeltaNet and FFN!):
    //   DeltaNet (build_qkv/build_qkvz):  il_cb = 1000*il + id
    //     For il=0:  il_cb ∈ {0, 1} (dev 0/1)
    //   FFN (llm_build_ffn split path):   il_cb = 1000*(id+1) + il
    //     For il=0:  il_cb ∈ {1000, 2000} (dev 0/1)
    st.match_names = {
        // DeltaNet kernel internals (use il, plain "-0")
        "q_in-0", "k_in-0", "v_in-0", "beta_in-0", "g_in-0", "state_in-0",
        "q_fused-0", "k_fused-0", "v_fused-0", "g_fused-0", "beta_fused-0", "state_fused-0",
        "delta_net_fused_raw-0", "output_tokens-0", "new_state-0",
        "attn_output-0", "q_conv_normed-0", "k_conv_normed-0",
        "conv_states-0", "state_predelta-0", "conv_output_raw-0", "conv_output_silu-0",
        // DeltaNet projections (il_cb = 1000*il + id = 0, 1)
        "qkv_mixed-0", "z-0", "linear_attn_qkv_mixed-0",
        "qkv_mixed-1", "z-1", "linear_attn_qkv_mixed-1",
        // build_gated_output internal tags (il_cb = 1000*il + id = 0, 1)
        "attn_rms_norm-0", "attn_out_norm-0", "final_output-0", "linear_attn_out-0",
        "attn_rms_norm-1", "attn_out_norm-1", "final_output-1", "linear_attn_out-1",
        // FFN per-device tags (il_cb = 1000*(id+1) + il = 1000, 2000)
        "ffn_up_gate-1000", "ffn_up_gate-2000",
        "ffn_down-1000",    "ffn_down-2000",
        "ffn_with_extra-1000", "ffn_with_extra-2000",
        // norm tags inside do_split_norm (likely uses il_cb too)
        "norm-1000", "norm-2000", "ffn_norm-1000", "ffn_norm-2000",
        "inp_normed-1000", "inp_normed-2000",
        // Per-layer (uses plain il)
        "l_out-0", "ffn_with_inp-0", "ffn_combined-0",
        "norm-0", "ffn_norm-0",
        "inp_embd",
    };

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = 4096 * 8;
    cparams.n_batch     = 2048;
    cparams.n_ubatch    = 2048;
    cparams.n_seq_max   = 8;
    cparams.type_k      = GGML_TYPE_Q4_0;
    cparams.type_v      = GGML_TYPE_Q4_0;
    cparams.type_reduce = GGML_TYPE_F32;  // CY.F.16: disable F32→F16 cast at ne[1]>32
    cparams.flash_attn  = true;
    cparams.mla_attn    = 3;
    cparams.k_cache_hadamard = true;
    cparams.v_cache_hadamard = true;
    cparams.cb_eval     = my_cb_eval;
    cparams.cb_eval_user_data = (void *) &st;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { llama_free_model(model); return 77; }

    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true, true);
    const int n_prompt = (int) tokens.size();
    fprintf(stderr, "[CY.F.15] n_prompt=%d np_batch=%d\n", n_prompt, np_batch);

    // ---- Phase A: serial prefill seq=0 ----
    llama_kv_cache_seq_rm(ctx, 0, -1, -1);
    st.store.clear();
    {
        llama_batch b = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++) {
            common_batch_add(b, tokens[i], i, {0}, i == n_prompt - 1);
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "serial fail\n"); return 1; }
        llama_batch_free(b);
    }
    fprintf(stderr, "[serial] captured %zu tensors\n", st.store.size());
    write_per_tag(st, out_dir, "serial");

    // ---- Phase B: batched np_batch seqs ----
    for (int sid = 0; sid < np_batch; ++sid)
        llama_kv_cache_seq_rm(ctx, (llama_seq_id) sid, -1, -1);
    st.store.clear();
    {
        const int total = np_batch * n_prompt;
        llama_batch b = llama_batch_init(total, 0, 1);
        for (int sid = 0; sid < np_batch; ++sid) {
            for (int i = 0; i < n_prompt; i++) {
                common_batch_add(b, tokens[i], i, {(llama_seq_id) sid}, i == n_prompt - 1);
            }
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "batched fail\n"); return 1; }
        llama_batch_free(b);
    }
    fprintf(stderr, "[batched] captured %zu tensors\n", st.store.size());
    write_per_tag(st, out_dir, "batched");

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
