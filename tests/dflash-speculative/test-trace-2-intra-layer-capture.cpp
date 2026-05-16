// test-trace-2-intra-layer-capture.cpp
//
// TRACE-2 — drills inside layer 3 (first full-attention layer in Qwen 3.6 hybrid)
// to localize the slot-parity divergence found in TRACE-1
// (data/trace-1-2026-05-16/findings.md).
//
// Captures every named intermediate around the layer-3 attention block via
// a custom cb_eval that matches a list of name prefixes, then dumps each
// per-slot row to .bin files.
//
// What we capture per device (production runs split across CUDA0 + CUDA1
// with split_mode=graph + tensor-split 1,1; per build_std_attention,
// `il_cb = 1000*(id+1) + il` so layer 3 on device 0 → il_cb=1003, device 1 → 2003):
//
//   l_out-2                       — input to layer 3 (= layer 2's output)
//   Qcur-1003 / Qcur-2003         — post-RoPE Q (one per device)
//   Kcur-1003 / Kcur-2003         — post-RoPE K
//   Qcur_hadamard-1003 / -2003    — post-K-hadamard Q
//   Kcur_hadamard-1003 / -2003    — post-K-hadamard K
//   Vcur_hadamard-1003 / -2003    — post-V-hadamard V
//   l_out-3                       — full layer-3 output
//
// At NP=2 with same prompt across both slots: if slot 0 and slot 1 differ at
// `l_out-2` already, the divergence is upstream of layer 3 (TRACE-1 didn't
// see this but is checked here). If they're equal at `l_out-2` and `Qcur*` /
// `Kcur*` / `Vcur*` are all equal but `l_out-3` differs, the divergence is
// inside the FA call or the cache read.
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
    std::vector<std::string> match_prefixes;  // exact-name matches we want
    std::vector<captured> store;
};

static bool name_matches(const std::string & needle, const char * name) {
    return std::strcmp(needle.c_str(), name) == 0;
}

static bool my_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (cap_state *) user_data;
    if (!st) return false;
    bool wanted = false;
    for (const auto & pat : st->match_prefixes) {
        if (name_matches(pat, t->name)) { wanted = true; break; }
    }
    if (!wanted) return false;
    if (ask) return true;
    if (!t->buffer) return true;
    captured c;
    c.name = t->name;
    c.ne   = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    c.type = t->type;
    const size_t nb = ggml_nbytes(t);
    c.data.resize(nb);
    ggml_backend_tensor_get(t, c.data.data(), 0, nb);
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
    // Write data as-is plus a small JSON sidecar with shape + dtype.
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
    // Layer 3 (first full-attention layer in the qwen3.6 hybrid).
    // build_std_attention tags with il_cb = 1000*(id+1) + il.
    st.match_prefixes = {
        "l_out-2",
        "l_out-3",
        "Qcur-1003", "Qcur-2003",
        "Kcur-1003", "Kcur-2003",
        "Qcur_hadamard-1003", "Qcur_hadamard-2003",
        "Kcur_hadamard-1003", "Kcur_hadamard-2003",
        "Vcur_hadamard-1003", "Vcur_hadamard-2003",
        // TRACE-3: capture the actual FA inputs — views into the K and
        // V caches. These reveal the post-quantize Q4_0 cache contents
        // that FA will read.
        "q-1003", "q-2003",
        "k-1003", "k-2003",
        "v-1003", "v-2003",
        "v_cache_view-1003", "v_cache_view-2003",
        // TRACE-6 (this run): capture EVERY downstream FA + output-
        // projection intermediate to pinpoint where slot-parity first
        // diverges within the layer-3 block.
        "flash_attn_per_slot_kv-1003", "flash_attn_per_slot_kv-2003",
        "flash_attn_h-1003", "flash_attn_h-2003",
        "flash_attn_reshaped-1003", "flash_attn_reshaped-2003",
        "kqv_wo-1003", "kqv_wo-2003",
        "kqv_wo_biased-1003", "kqv_wo_biased-2003",
        "attn_combined-3", "attn_out_with_input-3",
    };

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

    // Tokenize one prompt; copy to all N slots.
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
    fprintf(stderr, "[T2] N=%d  n_prompt=%d\n", N, n_prompt);

    std::vector<llama_token> id_last(N);
    std::vector<llama_pos>   next_pos(N);

    // Sequential prefill per slot, each with seq_id = slot index.
    for (int s = 0; s < N; s++) {
        llama_batch batch = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++) {
            bool last = (i == n_prompt - 1);
            common_batch_add(batch, base_tokens[i], i, {(llama_seq_id) s}, last);
        }
        // Drop captures from prefill — only want decode-step captures.
        st.store.clear();
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[T2] slot %d: prefill failed\n", s);
            llama_batch_free(batch);
            return 1;
        }
        float * logits = llama_get_logits_ith(ctx, n_prompt - 1);
        if (!logits) { return 1; }
        id_last[s]  = greedy_argmax(logits, n_vocab);
        next_pos[s] = (llama_pos) n_prompt;
        llama_batch_free(batch);
    }

    // Discard prefill captures.
    st.store.clear();

    // ONE decode step at N tokens (one per slot, same content).
    llama_batch batch = llama_batch_init(N, 0, 1);
    for (int s = 0; s < N; s++) {
        common_batch_add(batch, id_last[s], next_pos[s], {(llama_seq_id) s}, true);
    }
    fprintf(stderr, "[T2] decode N=%d tokens (next_token of same prompt)\n", N);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "[T2] decode failed\n");
        llama_batch_free(batch);
        return 1;
    }
    llama_batch_free(batch);

    fprintf(stderr, "[T2] captured %zu tensors\n", st.store.size());

    // Dump each captured tensor to .bin + .json.
    // Tensor shape comments from build_std_attention:
    //   Qcur, Kcur post-RoPE: [n_embd_head_k, n_heads_q/n_kv_heads, n_tokens, n_seqs]
    //   l_out-il:              [n_embd, n_tokens]  (single seq for our test)
    for (size_t i = 0; i < st.store.size(); ++i) {
        const auto & c = st.store[i];
        char path[1024];
        snprintf(path, sizeof(path), "%s/t2-np%d-%s.bin", out_dir, N, c.name.c_str());
        // Replace '/' in name if any (shouldn't happen for these tags).
        for (char * p = path; *p; ++p) if (*p == '/') {
            // leave it — only output dir has slashes
        }
        write_bin(path, c, ggml_type_name(c.type));
        fprintf(stderr, "  %s [%lld,%lld,%lld,%lld] type=%s nbytes=%zu\n",
                c.name.c_str(),
                (long long)c.ne[0], (long long)c.ne[1],
                (long long)c.ne[2], (long long)c.ne[3],
                ggml_type_name(c.type),
                c.data.size());
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
