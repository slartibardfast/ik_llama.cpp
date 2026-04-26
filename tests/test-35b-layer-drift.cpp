// Layer-by-layer drift locator for the fused_moe BI failure on 35B-A3B.
//
// Runs the model twice on the same prompt: once decoding a single token (N=1)
// and once decoding two tokens (N=2), capturing the pos-0 slice of every
// named intermediate tensor via ggml_backend_sched_eval_callback. Reports
// the first tensor whose values differ byte-for-byte between the two runs.
//
// The first divergent tensor pinpoints the op that actually introduces the
// drift. If it is one of the fused_moe output tensors, the bug is in the
// shader or its dispatch. If it is upstream of the MoE (attention, router,
// norm), the fused path is an innocent bystander and the real drift is in a
// previously-verified-BI op whose determinism breaks under the graph
// structure used when fused_moe is active.

#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct captured {
    std::string name;
    std::string op;
    int64_t ne[4];
    std::vector<float> pos0;
};

struct capture_ctx {
    std::vector<captured>                    in_order;
    std::unordered_map<std::string, size_t>  by_name;
    int n_hits = 0;
    bool enabled = false;
};

static void read_pos0(const ggml_tensor * t, std::vector<float> & out) {
    const size_t n = (size_t)t->ne[0];
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        std::vector<uint8_t> buf(n * sizeof(float));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
        memcpy(out.data(), buf.data(), buf.size());
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<uint8_t> buf(n * sizeof(ggml_fp16_t));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
        for (size_t i = 0; i < n; i++) {
            out[i] = ggml_fp16_to_fp32(((ggml_fp16_t *)buf.data())[i]);
        }
    } else {
        out.clear();
    }
}

static bool capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    capture_ctx * ctx = (capture_ctx *)user_data;
    if (ask) {
        return ctx->enabled && t->name[0] != '\0';
    }
    if (!ctx->enabled) return true;
    captured c;
    c.name = t->name;
    c.op   = ggml_op_name(t->op);
    for (int i = 0; i < 4; i++) c.ne[i] = t->ne[i];
    read_pos0(t, c.pos0);
    if (!c.pos0.empty()) {
        // Deduplicate: later occurrences of the same name (e.g. re-run cb nodes)
        // overwrite only if not already captured this run.
        if (ctx->by_name.find(c.name) == ctx->by_name.end()) {
            ctx->by_name[c.name] = ctx->in_order.size();
            ctx->in_order.push_back(std::move(c));
            ctx->n_hits++;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    gpt_params params;
    params.n_ctx = 256;
    params.n_batch = 32;
    params.n_ubatch = 32;
    params.n_gpu_layers = 99;
    if (!gpt_params_parse(argc, argv, params)) return 1;
    if (params.model.empty()) { fprintf(stderr, "usage: %s -m <gguf>\n", argv[0]); return 1; }

    llama_backend_init();

    // Two separate contexts — each with its own capture_ctx attached via cb_eval.
    capture_ctx cap1;
    capture_ctx capN;

    auto run_once = [&](int n_tokens, capture_ctx & cap) -> bool {
        gpt_params p = params;
        p.cb_eval = capture_cb;
        p.cb_eval_user_data = &cap;
        llama_init_result init = llama_init_from_gpt_params(p);
        if (!init.model || !init.context) {
            fprintf(stderr, "init failed\n");
            return false;
        }

        const std::string prompt = "The capital of France is";
        std::vector<llama_token> toks = llama_tokenize(llama_model_get_vocab(init.model), prompt, true);
        if ((int)toks.size() < n_tokens) { fprintf(stderr, "prompt too short for n=%d\n", n_tokens); return false; }

        cap.enabled = true;
        std::vector<llama_token> use(toks.begin(), toks.begin() + n_tokens);
        llama_batch batch = llama_batch_get_one(use.data(), (int32_t)use.size(), 0, 0);
        int rc = llama_decode(init.context, batch);
        cap.enabled = false;
        llama_free(init.context);
        llama_free_model(init.model);
        if (rc != 0) { fprintf(stderr, "decode failed rc=%d\n", rc); return false; }
        return true;
    };

    if (!run_once(1, cap1)) return 1;
    if (!run_once(2, capN)) return 1;

    printf("captured: run1=%d tensors, runN=%d tensors\n", cap1.n_hits, capN.n_hits);

    int checked = 0, matched = 0, first_printed = 0;
    int pos0_diverged = 0, tail_only_diverged = 0;
    for (const auto & c1 : cap1.in_order) {
        auto it = capN.by_name.find(c1.name);
        if (it == capN.by_name.end()) continue;
        const captured & cN = capN.in_order[it->second];
        const size_t n = std::min(c1.pos0.size(), cN.pos0.size());
        if (n == 0) continue;
        checked++;

        // pos_width heuristic: at N=1 only positions 0..0 are computed; at
        // N=2 positions 0..1. For most intermediate tensors the position axis
        // is the stride of c1.ne[0] / (something small). We don't know which
        // axis is position from the tensor metadata alone. Instead, split the
        // tensor into "likely pos-0 portion = first n/2 elements" and "tail"
        // and report both separately. If the drift is entirely in the tail,
        // it's uninit-slot noise. If it's in the first half too, it's real
        // pos-0 drift.
        const size_t half = n / 2;
        size_t first_diff_any  = n;
        size_t first_diff_head = n;  // within first half
        float  max_abs_head    = 0.0f;
        float  max_abs_tail    = 0.0f;
        size_t head_diff_count = 0;
        for (size_t i = 0; i < n; i++) {
            if (c1.pos0[i] != cN.pos0[i]) {
                if (first_diff_any == n) first_diff_any = i;
                float d = std::fabs(c1.pos0[i] - cN.pos0[i]);
                if (i < half) {
                    head_diff_count++;
                    if (first_diff_head == n) first_diff_head = i;
                    if (d > max_abs_head) max_abs_head = d;
                } else {
                    if (d > max_abs_tail) max_abs_tail = d;
                }
            }
        }
        if (first_diff_any == n) { matched++; continue; }

        const bool head_diverged = (head_diff_count > 0);
        if (head_diverged) pos0_diverged++;
        else               tail_only_diverged++;

        // Only print the first 20 pos-0-head divergences — those are the
        // real BI-relevant drift sites.
        if (head_diverged && first_printed < 20) {
            printf("DIVERGE-HEAD [%3d]: %-42s  op=%-12s  shape=[%lld,%lld,%lld,%lld]  head_diffs=%zu  first_idx=%zu  max|Δ|_head=%.3g  (run1=%.6f runN=%.6f)\n",
                   first_printed, c1.name.c_str(), c1.op.c_str(),
                   (long long)c1.ne[0], (long long)c1.ne[1], (long long)c1.ne[2], (long long)c1.ne[3],
                   head_diff_count, first_diff_head, max_abs_head,
                   c1.pos0[first_diff_head], cN.pos0[first_diff_head]);
            first_printed++;
        }
    }
    printf("summary: checked=%d identical=%d diverged=%d (head/pos-0-like=%d, tail-only=%d)\n",
           checked, matched, checked - matched, pos0_diverged, tail_only_diverged);

    llama_backend_free();
    return first_printed == 0 ? 0 : 2;  // exit 2 if drift detected
}
