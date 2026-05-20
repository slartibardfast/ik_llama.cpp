// llama-dflash.cpp
//
// DFlash speculative decoding orchestration (sm_75).
//
// The drafter is a sidecar weight bundle loaded via
// llama_dflash_drafter_load (gguf_init_from_file + per-tensor cudaMalloc
// upload, mirroring tests/dflash-speculative/dflash-drafter-loader.h),
// bound to a target context via llama_set_dflash, and driven by
// llama_dflash_draft which runs the fused kernel pipeline:
//
//   combine_features -> inject_kv_fused x L_d -> drafter_forward
//                    -> drafter_lm_head -> per-row argmax
//
// Target hiddens at the 5 source-layer indices are captured by the T2
// cb_eval hook into ctx->default_decoder.dflash_extract_buf[il] and
// uploaded per cycle.
//
// Shared embed / lm_head: drafter does NOT carry token_embd or
// output.weight in its GGUF; we point at the target model's tensors
// via llama_get_model_tensor. Satisfies @SharedEmbedAndLMHead.
//
// When GGML_CUDA_DFLASH is OFF every entry returns
// LLAMA_DFLASH_NOT_IMPLEMENTED.

#include "llama.h"

// Internal helper: called from llama_context's destructor in llama.cpp to
// release per-context DFlash scratch (KV cache + scratch buffers). The
// drafter itself is caller-owned and freed via llama_dflash_drafter_free.
extern void llama_dflash_release_ctx_state(struct llama_context * ctx);

#ifdef GGML_CUDA_DFLASH

#include "llama-context.h"
#include "llama-model.h"

#include "ggml.h"

#include "ggml-cuda/dflash/dflash-combine-features.cuh"
#include "ggml-cuda/dflash/dflash-inject-kv.cuh"
#include "ggml-cuda/dflash/dflash-drafter-forward.cuh"
#include "ggml-cuda/dflash/dflash-drafter-lm-head.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char * MODULE = "dflash";

inline bool cuda_ok(cudaError_t e, const char * what) {
    if (e == cudaSuccess) return true;
    std::fprintf(stderr, "[%s] CUDA error in %s: %s\n", MODULE, what, cudaGetErrorString(e));
    return false;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────
// Opaque types
// ─────────────────────────────────────────────────────────────────────

struct llama_dflash_drafter {
    // ── Dimensions / metadata (from drafter GGUF) ──
    int n_layers          = 0;
    int hidden_size       = 0;
    int intermediate_size = 0;
    int n_q_heads         = 0;
    int n_kv_heads        = 0;
    int head_dim          = 0;
    int sliding_window    = 0;
    int block_size        = 0;
    int mask_token_id     = 0;
    float rope_theta      = 0.0f;
    float rms_norm_eps    = 0.0f;
    std::vector<int> target_layer_ids;   // [1, 16, 31, 46, 61]
    std::vector<int> layer_types;        // 0 = SWA, 1 = full

    // ── Per-layer device pointers ──
    std::vector<const __half *> attn_norm, attn_q, attn_q_norm;
    std::vector<const __half *> attn_k,    attn_k_norm;
    std::vector<const __half *> attn_v,    attn_output;
    std::vector<const __half *> ffn_norm,  ffn_gate, ffn_up, ffn_down;

    // ── Non-layer device pointers ──
    const __half * dflash_fc          = nullptr;
    const __half * dflash_hidden_norm = nullptr;
    const __half * output_norm        = nullptr;

    // ── Backing storage ──
    std::vector<void *> gpu_buffers;
    struct gguf_context * gguf_ctx = nullptr;
    struct ggml_context * ggml_ctx = nullptr;
};

struct llama_dflash_ctx_state {
    // Drafter reference (caller owns; do not free here).
    struct llama_dflash_drafter * drafter = nullptr;

    // Shared tensor pointers (target-owned; do not free).
    const __half        * target_token_embd = nullptr;   // F16
    const __half        * target_lm_head    = nullptr;   // F16 (target must use lm_head-f16 recast — see scripts/recast_bf16_to_fp16.py T1)
    int                   target_vocab_size = 0;

    // Drafter KV cache: L_d * N_slots_cap * SeqLen * H_kv * D_h * 2 (K and V).
    __half * d_k_cache = nullptr;
    __half * d_v_cache = nullptr;
    int seq_len_cap = 0;

    // Scratch buffers, sized at allocate-once time.
    // Multi-slot layouts (slot-major) baked in at allocation; the
    // single-slot dispatch only writes/reads slot-0 offsets. Phase 4
    // turns on full multi-slot use of these buffers.
    __half * d_ctx_states     = nullptr;     // [N_slots_cap, MAL_anchors, D_emb]
    __half * d_target_hiddens = nullptr;     // [N_slots_cap, MAL_anchors, L_src, D_emb]
    __half * d_input_emb      = nullptr;     // [N_slots_cap, (1+BS), D_emb]
    __half * d_drafter_hidden = nullptr;     // [N_slots_cap, BS, D_emb]
    float  * d_drafter_logits = nullptr;     // [N_slots_cap, BS, V]
    int    * d_anchor_pos     = nullptr;     // [N_slots_cap, MAL_anchors]
    int    * d_slot_positions = nullptr;     // [N_slots_cap]
    int    * d_layer_types    = nullptr;     // [L_d]
    int      mal_cap          = 0;           // largest MAL allocated for so far
    int      n_slots_cap      = 1;           // captured from ctx_tgt->cparams.n_seq_max at bind

    // Host scratch for logit readback + argmax (since we do CPU argmax for T5).
    std::vector<float> h_logits;

    // Cycle counter for diagnostics. Single CPU int increment per
    // cycle — no measurable perf impact (GPU work dwarfs by 6+ orders).
    int n_cycles = 0;
};

// ─────────────────────────────────────────────────────────────────────
// Helpers — modelled after tests/dflash-speculative/dflash-drafter-loader.h
// ─────────────────────────────────────────────────────────────────────

namespace {

inline void * upload(llama_dflash_drafter & w, const void * src, std::size_t n_bytes) {
    void * dev = nullptr;
    if (!cuda_ok(cudaMalloc(&dev, n_bytes), "cudaMalloc(drafter weight)")) return nullptr;
    if (!cuda_ok(cudaMemcpy(dev, src, n_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(drafter weight)")) {
        cudaFree(dev);
        return nullptr;
    }
    w.gpu_buffers.push_back(dev);
    return dev;
}

inline float kv_f32(struct gguf_context * g, const char * key) {
    int idx = gguf_find_key(g, key);
    return idx < 0 ? 0.0f : gguf_get_val_f32(g, idx);
}
inline uint32_t kv_u32(struct gguf_context * g, const char * key) {
    int idx = gguf_find_key(g, key);
    return idx < 0 ? 0u : gguf_get_val_u32(g, idx);
}
inline std::vector<int> kv_array_i32(struct gguf_context * g, const char * key) {
    std::vector<int> out;
    int idx = gguf_find_key(g, key);
    if (idx < 0) return out;
    int n = gguf_get_arr_n(g, idx);
    const int32_t * data = static_cast<const int32_t *>(gguf_get_arr_data(g, idx));
    out.assign(data, data + n);
    return out;
}
inline std::vector<int> kv_layer_types_enum(struct gguf_context * g, const char * key) {
    std::vector<int> out;
    int idx = gguf_find_key(g, key);
    if (idx < 0) return out;
    int n = gguf_get_arr_n(g, idx);
    out.reserve(n);
    for (int i = 0; i < n; ++i) {
        const char * s = gguf_get_arr_str(g, idx, i);
        if (!s) { out.push_back(-1); continue; }
        if (std::strcmp(s, "sliding_attention") == 0)   out.push_back(0);
        else if (std::strcmp(s, "full_attention") == 0) out.push_back(1);
        else                                            out.push_back(-1);
    }
    return out;
}

// F32 norm weights live in the drafter GGUF as F32; the kernels read
// them as __half *. Cast at upload time per
// feedback_validate_gguf_dtype_at_load.md. Non-F32 tensors uploaded raw.
const __half * upload_f32_as_f16(llama_dflash_drafter & w, const char * name) {
    struct ggml_tensor * tn = ggml_get_tensor(w.ggml_ctx, name);
    if (!tn) {
        std::fprintf(stderr, "[%s] tensor not found: %s\n", MODULE, name);
        return nullptr;
    }
    if (tn->type != GGML_TYPE_F32) {
        return static_cast<const __half *>(upload(w, tn->data, ggml_nbytes(tn)));
    }
    const std::size_t n_elems = ggml_nelements(tn);
    std::vector<__half> tmp(n_elems);
    const float * src = static_cast<const float *>(tn->data);
    for (std::size_t i = 0; i < n_elems; ++i) tmp[i] = __float2half(src[i]);
    return static_cast<const __half *>(upload(w, tmp.data(), n_elems * sizeof(__half)));
}

const __half * upload_raw_f16(llama_dflash_drafter & w, const char * name) {
    struct ggml_tensor * tn = ggml_get_tensor(w.ggml_ctx, name);
    if (!tn) {
        std::fprintf(stderr, "[%s] tensor not found: %s\n", MODULE, name);
        return nullptr;
    }
    return static_cast<const __half *>(upload(w, tn->data, ggml_nbytes(tn)));
}

} // namespace

// ─────────────────────────────────────────────────────────────────────
// Drafter load / free
// ─────────────────────────────────────────────────────────────────────

struct llama_dflash_drafter * llama_dflash_drafter_load(const char * path) {
    if (!path) return nullptr;
    auto * w = new llama_dflash_drafter();

    struct gguf_init_params gparams{};
    gparams.no_alloc = false;
    gparams.ctx      = &w->ggml_ctx;
    w->gguf_ctx = gguf_init_from_file(path, gparams);
    if (!w->gguf_ctx) {
        std::fprintf(stderr, "[%s] gguf_init_from_file failed: %s\n", MODULE, path);
        delete w;
        return nullptr;
    }

    w->n_layers          = (int) kv_u32(w->gguf_ctx, "dflash.block_count");
    w->hidden_size       = (int) kv_u32(w->gguf_ctx, "dflash.embedding_length");
    w->intermediate_size = (int) kv_u32(w->gguf_ctx, "dflash.feed_forward_length");
    w->n_q_heads         = (int) kv_u32(w->gguf_ctx, "dflash.attention.head_count");
    w->n_kv_heads        = (int) kv_u32(w->gguf_ctx, "dflash.attention.head_count_kv");
    w->sliding_window    = (int) kv_u32(w->gguf_ctx, "dflash.attention.sliding_window");
    w->block_size        = (int) kv_u32(w->gguf_ctx, "dflash.block_size");
    w->mask_token_id     = (int) kv_u32(w->gguf_ctx, "dflash.mask_token_id");
    w->rope_theta        =       kv_f32(w->gguf_ctx, "dflash.rope.freq_base");
    w->rms_norm_eps      =       kv_f32(w->gguf_ctx, "dflash.attention.layer_norm_rms_epsilon");
    w->head_dim          = (int) kv_u32(w->gguf_ctx, "dflash.attention.key_length");
    if (w->head_dim == 0 && w->n_q_heads > 0 && w->hidden_size > 0) {
        w->head_dim = w->hidden_size / w->n_q_heads;
    }
    w->target_layer_ids = kv_array_i32(w->gguf_ctx, "dflash.target_layer_ids");
    w->layer_types      = kv_layer_types_enum(w->gguf_ctx, "dflash.layer_types");

    if (w->n_layers <= 0 || w->hidden_size <= 0 || w->intermediate_size <= 0) {
        std::fprintf(stderr, "[%s] drafter metadata incomplete: n_layers=%d hidden=%d ffn=%d\n",
                     MODULE, w->n_layers, w->hidden_size, w->intermediate_size);
        llama_dflash_drafter_free(w);
        return nullptr;
    }

    w->attn_norm.resize(w->n_layers);
    w->attn_q.resize(w->n_layers);
    w->attn_q_norm.resize(w->n_layers);
    w->attn_k.resize(w->n_layers);
    w->attn_k_norm.resize(w->n_layers);
    w->attn_v.resize(w->n_layers);
    w->attn_output.resize(w->n_layers);
    w->ffn_norm.resize(w->n_layers);
    w->ffn_gate.resize(w->n_layers);
    w->ffn_up.resize(w->n_layers);
    w->ffn_down.resize(w->n_layers);

    char nb[64];
    for (int l = 0; l < w->n_layers; ++l) {
        std::snprintf(nb, sizeof(nb), "blk.%d.attn_norm.weight",   l); w->attn_norm[l]   = upload_f32_as_f16(*w, nb);
        std::snprintf(nb, sizeof(nb), "blk.%d.attn_q.weight",      l); w->attn_q[l]      = upload_raw_f16(*w, nb);
        std::snprintf(nb, sizeof(nb), "blk.%d.attn_q_norm.weight", l); w->attn_q_norm[l] = upload_f32_as_f16(*w, nb);
        std::snprintf(nb, sizeof(nb), "blk.%d.attn_k.weight",      l); w->attn_k[l]      = upload_raw_f16(*w, nb);
        std::snprintf(nb, sizeof(nb), "blk.%d.attn_k_norm.weight", l); w->attn_k_norm[l] = upload_f32_as_f16(*w, nb);
        std::snprintf(nb, sizeof(nb), "blk.%d.attn_v.weight",      l); w->attn_v[l]      = upload_raw_f16(*w, nb);
        std::snprintf(nb, sizeof(nb), "blk.%d.attn_output.weight", l); w->attn_output[l] = upload_raw_f16(*w, nb);
        std::snprintf(nb, sizeof(nb), "blk.%d.ffn_norm.weight",    l); w->ffn_norm[l]    = upload_f32_as_f16(*w, nb);
        std::snprintf(nb, sizeof(nb), "blk.%d.ffn_gate.weight",    l); w->ffn_gate[l]    = upload_raw_f16(*w, nb);
        std::snprintf(nb, sizeof(nb), "blk.%d.ffn_up.weight",      l); w->ffn_up[l]      = upload_raw_f16(*w, nb);
        std::snprintf(nb, sizeof(nb), "blk.%d.ffn_down.weight",    l); w->ffn_down[l]    = upload_raw_f16(*w, nb);

        if (!w->attn_norm[l] || !w->attn_q[l] || !w->attn_q_norm[l] || !w->attn_k[l] ||
            !w->attn_k_norm[l] || !w->attn_v[l] || !w->attn_output[l] || !w->ffn_norm[l] ||
            !w->ffn_gate[l] || !w->ffn_up[l] || !w->ffn_down[l]) {
            std::fprintf(stderr, "[%s] missing tensor at layer %d\n", MODULE, l);
            llama_dflash_drafter_free(w);
            return nullptr;
        }
    }

    w->dflash_fc          = upload_raw_f16(*w, "dflash_fc.weight");
    w->dflash_hidden_norm = upload_f32_as_f16(*w, "dflash_hidden_norm.weight");
    w->output_norm        = upload_f32_as_f16(*w, "output_norm.weight");
    if (!w->dflash_fc || !w->dflash_hidden_norm || !w->output_norm) {
        llama_dflash_drafter_free(w);
        return nullptr;
    }

    std::fprintf(stderr, "[%s] drafter loaded: %d layers, hidden=%d, ffn=%d, vocab(target)=?, BS=%d, swa=%d, mask=%d\n",
                 MODULE, w->n_layers, w->hidden_size, w->intermediate_size,
                 w->block_size, w->sliding_window, w->mask_token_id);
    return w;
}

void llama_dflash_drafter_free(struct llama_dflash_drafter * drafter) {
    if (!drafter) return;
    for (void * p : drafter->gpu_buffers) if (p) cudaFree(p);
    if (drafter->gguf_ctx) gguf_free(drafter->gguf_ctx);
    if (drafter->ggml_ctx) ggml_free(drafter->ggml_ctx);
    delete drafter;
}

// ─────────────────────────────────────────────────────────────────────
// Per-context bind
// ─────────────────────────────────────────────────────────────────────

namespace {

// Find a target tensor by name, return its data pointer (must be on GPU
// after the CUDA backend has uploaded it).
const void * find_target_tensor_data(struct llama_model * model, const char * name) {
    struct ggml_tensor * t = llama_get_model_tensor(model, name);
    if (!t || !t->data) {
        std::fprintf(stderr, "[%s] target tensor missing: %s\n", MODULE, name);
        return nullptr;
    }
    return t->data;
}

// Allocate per-context DFlash scratch + drafter KV cache.
// seq_len_cap  = sequence-length budget for the drafter cache.
// mal_cap      = maximum MAL_anchors (= context positions to combine over).
// n_slots_cap  = max NP slots the bound context may serve. Captured from
//                ctx_tgt->cparams.n_seq_max at llama_set_dflash time. The
//                slot-major layouts in llama_dflash_ctx_state are sized
//                for the full N_slots_cap; the single-slot dispatch
//                writes/reads slot-0 offsets only until Phase 4 turns on
//                multi-slot use.
bool allocate_ctx_scratch(llama_dflash_ctx_state & st, const llama_dflash_drafter & dw,
                          int seq_len_cap, int mal_cap, int n_slots_cap) {
    const int L_d   = dw.n_layers;
    const int H_kv  = dw.n_kv_heads;
    const int D_h   = dw.head_dim;
    const int D_emb = dw.hidden_size;
    const int BS    = dw.block_size;
    const int Q     = 1 + BS;
    const int V     = st.target_vocab_size;
    const int L_src = (int) dw.target_layer_ids.size();

    if (n_slots_cap < 1) n_slots_cap = 1;

    // KV cache layout per kernel cuh: [L_d, N_slots, SeqLen, H_kv, D_h].
    // The single-slot dispatch passes a per-layer pointer that starts at
    // [l, 0, 0, 0, 0]; with N_slots=1 in the dispatch the kernel only
    // writes/reads the slot-0 sub-block.
    const std::size_t n_kv_per_layer = (std::size_t) n_slots_cap * seq_len_cap * H_kv * D_h;
    const std::size_t kv_bytes = (std::size_t) L_d * n_kv_per_layer * sizeof(__half);

    if (!cuda_ok(cudaMalloc(&st.d_k_cache, kv_bytes), "alloc K cache")) return false;
    if (!cuda_ok(cudaMalloc(&st.d_v_cache, kv_bytes), "alloc V cache")) return false;
    cudaMemset(st.d_k_cache, 0, kv_bytes);
    cudaMemset(st.d_v_cache, 0, kv_bytes);

    const std::size_t N = (std::size_t) n_slots_cap;

    if (!cuda_ok(cudaMalloc(&st.d_ctx_states, N * (std::size_t) mal_cap * D_emb * sizeof(__half)),
                 "alloc ctx_states")) return false;
    if (!cuda_ok(cudaMalloc(&st.d_target_hiddens, N * (std::size_t) mal_cap * L_src * D_emb * sizeof(__half)),
                 "alloc target_hiddens stage")) return false;
    if (!cuda_ok(cudaMalloc(&st.d_input_emb, N * (std::size_t) Q * D_emb * sizeof(__half)),
                 "alloc input_emb")) return false;
    if (!cuda_ok(cudaMalloc(&st.d_drafter_hidden, N * (std::size_t) BS * D_emb * sizeof(__half)),
                 "alloc drafter_hidden")) return false;
    if (!cuda_ok(cudaMalloc(&st.d_drafter_logits, N * (std::size_t) BS * V * sizeof(float)),
                 "alloc drafter_logits")) return false;
    if (!cuda_ok(cudaMalloc(&st.d_anchor_pos, N * (std::size_t) mal_cap * sizeof(int)),
                 "alloc anchor_pos")) return false;
    if (!cuda_ok(cudaMalloc(&st.d_slot_positions, N * sizeof(int)),
                 "alloc slot_positions")) return false;
    if (!cuda_ok(cudaMalloc(&st.d_layer_types, (std::size_t) L_d * sizeof(int)),
                 "alloc layer_types")) return false;
    cudaMemcpy(st.d_layer_types, dw.layer_types.data(),
               (std::size_t) L_d * sizeof(int), cudaMemcpyHostToDevice);

    st.h_logits.assign(N * (std::size_t) BS * V, 0.0f);

    st.seq_len_cap = seq_len_cap;
    st.mal_cap     = mal_cap;
    st.n_slots_cap = n_slots_cap;
    return true;
}

void free_ctx_scratch(llama_dflash_ctx_state & st) {
    auto F = [](void *& p){ if (p) { cudaFree(p); p = nullptr; } };
    F((void *&) st.d_k_cache);
    F((void *&) st.d_v_cache);
    F((void *&) st.d_ctx_states);
    F((void *&) st.d_target_hiddens);
    F((void *&) st.d_input_emb);
    F((void *&) st.d_drafter_hidden);
    F((void *&) st.d_drafter_logits);
    F((void *&) st.d_anchor_pos);
    F((void *&) st.d_slot_positions);
    F((void *&) st.d_layer_types);
}

// (T6.A DeltaNet state ping-pong removed 2026-05-13 — superseded by
// `llama_spec_ckpt_*` PER_STEP mode which is the canonical mechanism
// for DeltaNet state restore. See kernel-design.md §6.4.)

} // namespace

int32_t llama_set_dflash(
        struct llama_context        * ctx_tgt,
        struct llama_dflash_drafter * drafter) {
    if (!ctx_tgt || !drafter) return LLAMA_DFLASH_INVALID_DRAFTER;

    if (ctx_tgt->dflash_state) {
        std::fprintf(stderr, "[%s] llama_set_dflash called twice on same context — freeing previous binding\n", MODULE);
        free_ctx_scratch(*ctx_tgt->dflash_state);
        delete ctx_tgt->dflash_state;
        ctx_tgt->dflash_state = nullptr;
    }

    struct llama_model * model = const_cast<struct llama_model *>(&ctx_tgt->model);

    auto * st = new llama_dflash_ctx_state();
    st->drafter = drafter;

    // Resolve shared embed + lm_head from target.
    // Both must be F16 — if the target's output.weight is still BF16, reject
    // loudly. Use the T1-recast target variant (V-F1.T1.lm_head-f16.gguf).
    {
        struct ggml_tensor * te = llama_get_model_tensor(model, "token_embd.weight");
        struct ggml_tensor * oe = llama_get_model_tensor(model, "output.weight");
        if (!te || !oe) { delete st; return LLAMA_DFLASH_INVALID_DRAFTER; }
        if (te->type != GGML_TYPE_F16 || oe->type != GGML_TYPE_F16) {
            std::fprintf(stderr,
                "[%s] target token_embd / output.weight must both be F16 — got %s / %s. "
                "Use the lm_head-f16 recast target variant.\n",
                MODULE,
                ggml_type_name(te->type), ggml_type_name(oe->type));
            delete st;
            return LLAMA_DFLASH_INVALID_DRAFTER;
        }
    }
    st->target_token_embd = static_cast<const __half *>(
        find_target_tensor_data(model, "token_embd.weight"));
    st->target_lm_head = static_cast<const __half *>(
        find_target_tensor_data(model, "output.weight"));
    if (!st->target_token_embd || !st->target_lm_head) {
        delete st;
        return LLAMA_DFLASH_INVALID_DRAFTER;
    }

    // Read target vocab size from model.
    {
        struct ggml_tensor * te = llama_get_model_tensor(model, "token_embd.weight");
        if (!te) { delete st; return LLAMA_DFLASH_INVALID_DRAFTER; }
        st->target_vocab_size = (int) te->ne[1];
    }

    // Allocate scratch. seq_len_cap bounds the drafter's per-slot K/V
    // cache depth AND the MAL (max anchor positions combined per cycle).
    // Bound = swa_window + block_size + headroom: SWA layers only attend
    // within swa_window of the current position, and the drafter writes
    // K/V for Q = 1 + block_size query positions in cache_write_kv during
    // forward. With MAL capped to this, the full-attention layer also
    // operates within the same window (an explicit consequence of the
    // cap; revisit if drafter quality at production-scale context shows
    // measurable regression). Caller capping anchor_pos at this bound
    // (common/speculative.cpp) prevents the validation reject at draft
    // time; stage_target_hiddens reads from the extract-buf tail so the
    // drafter sees the most recent MAL_max target positions.
    const int swa_window  = std::max(1, drafter->sliding_window);
    const int block_size  = std::max(1, drafter->block_size);
    const int MAL_max     = swa_window + block_size + 16;
    const int seq_len_cap = MAL_max;
    const int mal_cap     = MAL_max;
    const int n_slots_cap = (int) std::max<uint32_t>(ctx_tgt->cparams.n_seq_max, 1u);
    if (!allocate_ctx_scratch(*st, *drafter, seq_len_cap, mal_cap, n_slots_cap)) {
        free_ctx_scratch(*st);
        delete st;
        return LLAMA_DFLASH_LOAD_FAILED;
    }
    // (T6.A DeltaNet state ping-pong allocation removed 2026-05-13.
    // The canonical state-restore path is llama_spec_ckpt_* — callers
    // invoke llama_spec_ckpt_init themselves after binding DFlash.)

    // Arrange the extract hook on target's 5 source layers automatically.
    std::vector<int32_t> ids(drafter->target_layer_ids.begin(),
                              drafter->target_layer_ids.end());
    llama_set_dflash_extract_layers(ctx_tgt, ids.data(), (int32_t) ids.size());

    ctx_tgt->dflash_state = st;

    std::fprintf(stderr, "[%s] bound: seq_len_cap=%d mal_cap=%d V=%d  source_layers=[",
                 MODULE, seq_len_cap, mal_cap, st->target_vocab_size);
    for (size_t i = 0; i < ids.size(); ++i) {
        std::fprintf(stderr, "%s%d", i?",":"", ids[i]);
    }
    std::fprintf(stderr, "]\n");
    return LLAMA_DFLASH_OK;
}

// ─────────────────────────────────────────────────────────────────────
// Query API
// ─────────────────────────────────────────────────────────────────────

struct llama_dflash_drafter * llama_get_dflash_drafter(struct llama_context * ctx) {
    if (!ctx || !ctx->dflash_state) return nullptr;
    return ctx->dflash_state->drafter;
}

int32_t llama_dflash_n_source_layers(const struct llama_dflash_drafter * drafter) {
    return drafter ? (int32_t) drafter->target_layer_ids.size() : -1;
}
int32_t llama_dflash_block_size(const struct llama_dflash_drafter * drafter) {
    return drafter ? drafter->block_size : -1;
}
llama_token llama_dflash_mask_token_id(const struct llama_dflash_drafter * drafter) {
    return drafter ? (llama_token) drafter->mask_token_id : -1;
}
int32_t llama_dflash_swa_window(const struct llama_dflash_drafter * drafter, int32_t layer_idx) {
    if (!drafter) return -2;
    if (layer_idx < 0 || layer_idx >= drafter->n_layers) return -2;
    if (drafter->layer_types.empty()) return drafter->sliding_window;
    return drafter->layer_types[layer_idx] == 0 ? drafter->sliding_window : 0;
}
enum llama_dflash_layer_type llama_dflash_layer_type_at(
        const struct llama_dflash_drafter * drafter, int32_t layer_idx) {
    if (!drafter || layer_idx < 0 || layer_idx >= drafter->n_layers || drafter->layer_types.empty())
        return LLAMA_DFLASH_LAYER_FULL_ATTENTION;
    return drafter->layer_types[layer_idx] == 0
        ? LLAMA_DFLASH_LAYER_SLIDING_ATTENTION
        : LLAMA_DFLASH_LAYER_FULL_ATTENTION;
}

// ─────────────────────────────────────────────────────────────────────
// Cycle: produce BLOCK_SIZE candidate tokens
// ─────────────────────────────────────────────────────────────────────

namespace {

// Stage target hiddens from the cb_eval host buffer to the per-cycle
// GPU staging buffer at shape [MAL_anchors, L_src, D_emb] row-major.
// T2's cb_eval hook is append-only (one ubatch's hiddens per cb_eval
// fire). After a verify decode in cycle N adds BS+1 rows, the buffer
// grows to anchor_pos_prev + BS+1 rows. For cycle N+1, anchor_pos is
// at most that — the slot at index n_accepted in the verify-added rows
// was decoded with the REJECTED input rather than the bonus, but we
// accept that one-row degradation rather than re-decoding (T6
// state-save/restore would clean this up; out of T5 scope).
//
// After staging, we trim the buffer to anchor_pos rows so subsequent
// verify decode appends extend a clean slate.
//
// Spec: specs/dflash/dflash.allium @ AtomicityPerCycle / NewAnchorIsBonus
bool stage_target_hiddens(llama_dflash_ctx_state & st,
                          const llama_dflash_drafter & dw,
                          struct llama_context * ctx,
                          int mal_anchors,
                          llama_seq_id seq_id,
                          int slot_idx = 0) {
    const int D_emb = dw.hidden_size;
    const int L_src = (int) dw.target_layer_ids.size();

    std::vector<__half> h_stage((std::size_t) mal_anchors * L_src * D_emb);

    for (int i = 0; i < L_src; ++i) {
        // dflash_extract_buf is indexed by source-layer SLOT (0..L_src-1),
        // NOT by target layer id. The cb_eval hook stores at the slot
        // index it finds for the matched layer; this matches the order
        // of llama_set_dflash_extract_layers' input array. Inner vector
        // is per-seq_id (sized to cparams.n_seq_max at hook install).
        auto & buf_per_seq = ctx->default_decoder.dflash_extract_buf[i];
        if (seq_id < 0 || (size_t) seq_id >= buf_per_seq.size()) {
            std::fprintf(stderr, "[%s] stage_target_hiddens: seq_id %d out of range (n_seq_max=%zu)\n",
                         MODULE, (int) seq_id, buf_per_seq.size());
            return false;
        }
        std::vector<float> & buf = buf_per_seq[seq_id];
        const int il = dw.target_layer_ids[i];
        if (i == 0 && std::getenv("DFLASH_DIAG")) {
            std::fprintf(stderr, "[dflash-diag stage] slot 0 layer %d seq_id %d: buf_rows=%zu MAL=%d\n",
                         il, (int) seq_id, buf.size() / (std::size_t) D_emb, mal_anchors);
        }
        if ((int) (buf.size() / D_emb) < mal_anchors) {
            std::fprintf(stderr, "[%s] extract buffer too short for slot %d (layer %d) seq_id %d: have %zu rows, need %d\n",
                         MODULE, i, il, (int) seq_id, buf.size() / (std::size_t) D_emb, mal_anchors);
            return false;
        }
        for (int a = 0; a < mal_anchors; ++a) {
            const float * row = buf.data() + (std::size_t) a * D_emb;
            for (int d = 0; d < D_emb; ++d) {
                h_stage[((std::size_t) a * L_src + i) * D_emb + d] = __float2half(row[d]);
            }
        }
        // Trim to mal_anchors rows for next cycle. This removes the
        // rejected-draft hiddens that the previous cycle's verify decode
        // appended past the (capped) anchor_pos count — the next cycle's
        // mal_anchors equals the post-accept cache_tokens.size() (capped
        // at MAL_max in common/speculative.cpp), so any rows beyond that
        // are stale and must not be fed to the drafter. Without this trim
        // the drafter sees stale rejected hiddens and produces wrong
        // predictions, visible as token duplication in the output stream.
        buf.resize((std::size_t) mal_anchors * D_emb);
        ctx->default_decoder.dflash_extract_n[i][seq_id] = buf.size();
    }

    // Multi-slot layout: d_target_hiddens is [N_slots_cap, mal_cap, L_src, D_emb].
    // Write this slot's MAL anchors at the slot's base offset; the rectangular
    // stride uses mal_cap (not mal_anchors) so adjacent slots don't overlap.
    const std::size_t slot_stride = (std::size_t) st.mal_cap * L_src * D_emb;
    __half * d_slot = st.d_target_hiddens + (std::size_t) slot_idx * slot_stride;
    return cuda_ok(cudaMemcpy(d_slot, h_stage.data(),
                              h_stage.size() * sizeof(__half), cudaMemcpyHostToDevice),
                   "upload staged target_hiddens");
}

} // namespace

// Single-slot trampoline. Phase 4 collapsed the bulk single-slot
// dispatch into llama_dflash_draft_batch with n_slots=1 — preserving
// byte-identical output (slot-0 pointer offsets are zero so the
// multi-slot path degenerates to the prior layout exactly).
int32_t llama_dflash_draft(
        struct llama_context * ctx_tgt,
        llama_token            anchor_token_id,
        int32_t                anchor_pos,
        llama_token          * out_candidates,
        int32_t                max_candidates) {
    const llama_seq_id sid0 = 0;
    return llama_dflash_draft_batch(
            ctx_tgt,
            /*n_slots*/ 1,
            &anchor_token_id,
            &anchor_pos,
            &sid0,
            out_candidates,
            max_candidates);
}

// Multi-slot batched draft. Phase 4: full per-slot dispatch.
//
// Memory layout invariants (set up by Phase 2 alloc_ctx_scratch sized
// to st.n_slots_cap, populated by Phase 3 cb_eval per-seq demux):
//   d_target_hiddens : [n_slots_cap, mal_cap, L_src, D_emb]
//   d_ctx_states    : [n_slots_cap, mal_cap, D_emb]
//   d_anchor_pos    : [n_slots_cap, mal_cap]
//   d_input_emb     : [n_slots_cap, Q,   D_emb]
//   d_slot_positions: [n_slots_cap]
//   d_drafter_hidden: [n_slots_cap, BS,  D_emb]
//   d_drafter_logits: [n_slots_cap, BS,  V]
//   d_k_cache       : [L_d, n_slots_cap, SeqLen, H_kv, D_h]
//   d_v_cache       : [L_d, n_slots_cap, SeqLen, H_kv, D_h]
//
// combine_features and inject_kv_fused take a single MAL_anchors stride
// across all slots; production slots have variable MAL (anchor_pos), so
// those two kernels are dispatched serially per slot with N_slots=1 at
// the slot's pointer offset. drafter_forward and lm_head are shape-
// uniform across slots and dispatch ONCE with N_slots=n_slots —
// recovering the in-flight parallelism on those two stages.
int32_t llama_dflash_draft_batch(
        struct llama_context * ctx_tgt,
        int32_t                n_slots,
        const llama_token    * anchor_token_ids,
        const int32_t        * anchor_positions,
        const llama_seq_id   * seq_ids,
        llama_token          * out_candidates,
        int32_t                max_total_candidates) {
    if (!ctx_tgt || !ctx_tgt->dflash_state) return LLAMA_DFLASH_NOT_IMPLEMENTED;
    auto & st = *ctx_tgt->dflash_state;
    if (!st.drafter)                        return LLAMA_DFLASH_INVALID_DRAFTER;
    auto & dw = *st.drafter;

    if (n_slots < 1)                        return LLAMA_DFLASH_INVALID_DRAFTER;
    if (n_slots > st.n_slots_cap)           return LLAMA_DFLASH_NP_GT_1;
    if (!anchor_token_ids)                  return LLAMA_DFLASH_INVALID_DRAFTER;
    if (!anchor_positions)                  return LLAMA_DFLASH_INVALID_DRAFTER;
    if (!out_candidates)                    return LLAMA_DFLASH_INVALID_DRAFTER;

    const int BS = 4;
    if (max_total_candidates < BS * n_slots) {
        std::fprintf(stderr, "[%s] out_candidates size %d < BS*n_slots = %d\n",
                     MODULE, max_total_candidates, BS * n_slots);
        return LLAMA_DFLASH_INVALID_DRAFTER;
    }

    const int D_emb        = dw.hidden_size;
    const int L_d          = dw.n_layers;
    const int H_kv         = dw.n_kv_heads;
    const int D_h          = dw.head_dim;
    const int H_q          = dw.n_q_heads;
    const int intermediate = dw.intermediate_size;
    const int swa_window   = dw.sliding_window;
    const float rope_base  = dw.rope_theta;
    const float norm_eps   = dw.rms_norm_eps;
    const int L_src        = (int) dw.target_layer_ids.size();
    const int Q            = 1 + BS;
    const int SeqLen       = st.seq_len_cap;
    const int V            = st.target_vocab_size;

    // Per-slot anchor_pos = each slot's MAL. Validate range and pack
    // host-side anchor_positions array we'll upload per slot.
    for (int s = 0; s < n_slots; ++s) {
        const int ap_s = anchor_positions[s];
        if (ap_s <= 0 || ap_s >= st.seq_len_cap) {
            std::fprintf(stderr, "[%s] slot %d anchor_pos %d out of range (0, %d)\n",
                         MODULE, s, ap_s, st.seq_len_cap);
            return LLAMA_DFLASH_INVALID_DRAFTER;
        }
        if (ap_s > st.mal_cap) {
            std::fprintf(stderr, "[%s] slot %d MAL %d exceeds capacity %d\n",
                         MODULE, s, ap_s, st.mal_cap);
            return LLAMA_DFLASH_INVALID_DRAFTER;
        }
    }

    // ── 1+2. Per-slot stage + combine_features + inject_kv × L_d ──
    const std::size_t n_kv_per_layer = (std::size_t) st.n_slots_cap * SeqLen * H_kv * D_h;
    const std::size_t hiddens_slot_stride = (std::size_t) st.mal_cap * L_src * D_emb;
    const std::size_t ctx_slot_stride     = (std::size_t) st.mal_cap * D_emb;
    const std::size_t anchor_slot_stride  = (std::size_t) st.mal_cap;
    const std::size_t kv_slot_stride      = (std::size_t) SeqLen * H_kv * D_h;

    for (int s = 0; s < n_slots; ++s) {
        const int MAL_s = anchor_positions[s];
        const llama_seq_id sid = seq_ids ? seq_ids[s] : (llama_seq_id) s;

        // Stage target hiddens for slot s at its slot-major offset.
        if (!stage_target_hiddens(st, dw, ctx_tgt, MAL_s, sid, s)) {
            return LLAMA_DFLASH_MISSING_METADATA;
        }

        // anchor_positions[a] = a for slot s, packed into d_anchor_pos
        // at slot s's offset.
        {
            std::vector<int> ap(MAL_s);
            for (int a = 0; a < MAL_s; ++a) ap[a] = a;
            cudaMemcpy(st.d_anchor_pos + s * anchor_slot_stride,
                       ap.data(), (std::size_t) MAL_s * sizeof(int),
                       cudaMemcpyHostToDevice);
        }

        // combine_features for slot s — N_slots=1 at slot offsets.
        dflash_combine_features_launch(
            st.d_target_hiddens + s * hiddens_slot_stride,
            dw.dflash_fc, dw.dflash_hidden_norm, norm_eps,
            st.d_ctx_states    + s * ctx_slot_stride,
            /*N_slots*/ 1, MAL_s, L_src, D_emb, 0);

        // inject_kv_fused × L_d for slot s — N_slots=1 at slot offsets.
        // Per-layer base pointer must include slot offset within the
        // [L_d, n_slots_cap, SeqLen, ...] cache.
        for (int l = 0; l < L_d; ++l) {
            dflash_inject_kv_fused_launch(
                st.d_ctx_states + s * ctx_slot_stride,
                dw.attn_k[l], dw.attn_v[l], dw.attn_k_norm[l],
                rope_base, norm_eps,
                st.d_k_cache + (std::size_t) l * n_kv_per_layer + s * kv_slot_stride,
                st.d_v_cache + (std::size_t) l * n_kv_per_layer + s * kv_slot_stride,
                st.d_anchor_pos + s * anchor_slot_stride,
                /*N_slots*/ 1, MAL_s, H_kv, D_h, D_emb, SeqLen, 0);
        }
    }

    // ── 3. Compose input embeddings slot-major: [N_slots, Q, D_emb] ──
    for (int s = 0; s < n_slots; ++s) {
        const std::size_t s_off = (std::size_t) s * Q * D_emb;
        if (!cuda_ok(cudaMemcpy(
                st.d_input_emb + s_off + (std::size_t) 0 * D_emb,
                st.target_token_embd + (std::size_t) anchor_token_ids[s] * D_emb,
                (std::size_t) D_emb * sizeof(__half), cudaMemcpyDeviceToDevice),
                "copy anchor emb")) return LLAMA_DFLASH_LOAD_FAILED;
        for (int i = 1; i < Q; ++i) {
            if (!cuda_ok(cudaMemcpy(
                    st.d_input_emb + s_off + (std::size_t) i * D_emb,
                    st.target_token_embd + (std::size_t) dw.mask_token_id * D_emb,
                    (std::size_t) D_emb * sizeof(__half), cudaMemcpyDeviceToDevice),
                    "copy mask emb")) return LLAMA_DFLASH_LOAD_FAILED;
        }
    }

    // ── 4. slot_positions[s] = anchor_positions[s] ──
    {
        std::vector<int> sp(n_slots);
        for (int s = 0; s < n_slots; ++s) sp[s] = anchor_positions[s];
        cudaMemcpy(st.d_slot_positions, sp.data(),
                   (std::size_t) n_slots * sizeof(int), cudaMemcpyHostToDevice);
    }

    // ── 5. drafter_forward — multi-slot in ONE launch ──
    std::vector<const __half *> p_attn_norm(L_d), p_q_w(L_d), p_q_norm(L_d);
    std::vector<const __half *> p_k_w(L_d), p_k_norm(L_d), p_v_w(L_d), p_o_w(L_d);
    std::vector<const __half *> p_ffn_norm(L_d), p_gate(L_d), p_up(L_d), p_down(L_d);
    for (int l = 0; l < L_d; ++l) {
        p_attn_norm[l] = dw.attn_norm[l];
        p_q_w[l]       = dw.attn_q[l];
        p_q_norm[l]    = dw.attn_q_norm[l];
        p_k_w[l]       = dw.attn_k[l];
        p_k_norm[l]    = dw.attn_k_norm[l];
        p_v_w[l]       = dw.attn_v[l];
        p_o_w[l]       = dw.attn_output[l];
        p_ffn_norm[l]  = dw.ffn_norm[l];
        p_gate[l]      = dw.ffn_gate[l];
        p_up[l]        = dw.ffn_up[l];
        p_down[l]      = dw.ffn_down[l];
    }

    // N_slots = n_slots (dispatch count) is decoupled from n_slots_cap
    // (storage stride) — kernel uses N_slots for grid iteration and
    // n_slots_cap for the per-layer K/V cache base offset. This
    // matches Phase 2's [L_d, n_slots_cap, SeqLen, H_kv, D_h] cache
    // allocation regardless of how many slots the call actually wants.
    dflash_drafter_forward_launch(
        st.d_input_emb, st.d_k_cache, st.d_v_cache, st.d_slot_positions,
        p_attn_norm.data(), p_q_w.data(), p_q_norm.data(),
        p_k_w.data(), p_k_norm.data(), p_v_w.data(), p_o_w.data(),
        p_ffn_norm.data(), p_gate.data(), p_up.data(), p_down.data(),
        dw.output_norm,
        st.d_layer_types,
        swa_window, rope_base, norm_eps,
        BS, n_slots, st.n_slots_cap, SeqLen, L_d,
        D_emb, H_q, H_kv, D_h, intermediate,
        st.d_drafter_hidden, 0);

    // ── 6. drafter_lm_head — flat n_slots*BS rows ──
    dflash_drafter_lm_head_launch(
        st.d_drafter_hidden, st.target_lm_head, st.d_drafter_logits,
        n_slots * BS, D_emb, V, 0);

    cudaDeviceSynchronize();

    // ── 7. Per-row argmax on CPU; slot-major out_candidates ──
    cudaMemcpy(st.h_logits.data(), st.d_drafter_logits,
               (std::size_t) n_slots * BS * V * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int s = 0; s < n_slots; ++s) {
        for (int row = 0; row < BS; ++row) {
            const float * r = st.h_logits.data() + ((std::size_t) s * BS + row) * V;
            int argmax = 0;
            float maxv = r[0];
            for (int v = 1; v < V; ++v) {
                if (r[v] > maxv) { maxv = r[v]; argmax = v; }
            }
            out_candidates[s * BS + row] = (llama_token) argmax;
        }
    }

    st.n_cycles += 1;
    return BS * n_slots;
}

// Internal helper called from llama.cpp's llama_context destructor.
// Releases per-context DFlash scratch (KV cache + scratch buffers)
// but NOT the drafter — drafter is caller-owned, freed by
// llama_dflash_drafter_free.
void llama_dflash_release_ctx_state(struct llama_context * ctx) {
    if (!ctx || !ctx->dflash_state) return;
    free_ctx_scratch(*ctx->dflash_state);
    delete ctx->dflash_state;
    ctx->dflash_state = nullptr;
}

// (T6.A `llama_dflash_state_snapshot` / `_restore` removed 2026-05-13 —
//  see kernel-design.md §6.4. Canonical path is llama_spec_ckpt_*.)

// T6.C: trim the cb_eval extract buffer per slot to match a seq_rm
// rollback. Walks the source-layer slots configured at
// llama_set_dflash time; for each, resizes the float buffer.
//
// Semantics match llama_kv_cache_seq_rm: p_end == -1 means "remove
// to end"; p_end > p_start means "remove the slice [p_start, p_end)".
//
// Spec: kernel-design.md §6.8.
int32_t llama_dflash_trim_extract(
        struct llama_context * ctx_tgt,
        int32_t                p_start,
        int32_t                p_end) {
    if (!ctx_tgt) return LLAMA_DFLASH_INVALID_DRAFTER;
    if (p_start < 0) return LLAMA_DFLASH_INVALID_DRAFTER;

    const int n_slots = ctx_tgt->cparams.dflash_extract_count;
    if (n_slots <= 0) return LLAMA_DFLASH_OK;  // hook not armed; no-op

    // Determine D_emb from the drafter binding so we can convert
    // position counts to float-element counts.
    int D_emb = 0;
    if (ctx_tgt->dflash_state && ctx_tgt->dflash_state->drafter) {
        D_emb = ctx_tgt->dflash_state->drafter->hidden_size;
    }
    if (D_emb <= 0) {
        // Fall back to the model's n_embd if drafter not bound.
        D_emb = ctx_tgt->model.hparams.n_embd;
    }
    if (D_emb <= 0) return LLAMA_DFLASH_MISSING_METADATA;

    // Apply trim semantics to every populated per-seq buffer. At
    // n_seq_max=1 only seq_id=0 is populated (Phase 3 default); at
    // n_seq_max>1 with Phase 3 single-slot dispatch this still only
    // affects seq_id=0 (others remain empty no-ops). Phase 4's
    // multi-slot draft path drives separate seq_id populations.
    for (int i = 0; i < n_slots; ++i) {
        auto & buf_per_seq = ctx_tgt->default_decoder.dflash_extract_buf[i];
        auto & n_per_seq   = ctx_tgt->default_decoder.dflash_extract_n[i];
        for (size_t sid = 0; sid < buf_per_seq.size(); ++sid) {
            std::vector<float> & buf = buf_per_seq[sid];
            if (buf.empty()) continue;
            const std::size_t row = (std::size_t) D_emb;
            const std::size_t n_rows_have = buf.size() / row;
            const std::size_t n_rows_before = n_rows_have;
            if (p_end < 0 || (std::size_t) p_end >= n_rows_have) {
                // Truncate to first p_start rows.
                if ((std::size_t) p_start < n_rows_have) {
                    buf.resize((std::size_t) p_start * row);
                }
            } else {
                // Remove slice [p_start, p_end).
                if ((std::size_t) p_start >= n_rows_have) continue;
                const std::size_t b_start = (std::size_t) p_start * row;
                const std::size_t b_end   = (std::size_t) p_end   * row;
                buf.erase(buf.begin() + b_start, buf.begin() + b_end);
            }
            n_per_seq[sid] = buf.size();
            if (i == 0 && sid == 0 && std::getenv("DFLASH_DIAG")) {
                std::fprintf(stderr, "[dflash-diag trim] slot 0 seq_id 0: before=%zu p_start=%d p_end=%d after=%zu\n",
                             n_rows_before, p_start, p_end, buf.size() / row);
            }
        }
    }
    return LLAMA_DFLASH_OK;
}

void llama_dflash_get_cycle_stats(
        const struct llama_context * ctx_tgt,
        int32_t * out_n_cycles) {
    if (!ctx_tgt || !ctx_tgt->dflash_state) {
        if (out_n_cycles) *out_n_cycles = 0;
        return;
    }
    const auto & st = *ctx_tgt->dflash_state;
    if (out_n_cycles) *out_n_cycles = st.n_cycles;
}

#else // GGML_CUDA_DFLASH not defined — stubs

void llama_dflash_release_ctx_state(struct llama_context * /*ctx*/) {
    // Field doesn't exist when DFlash compile-disabled.
}


struct llama_dflash_drafter * llama_dflash_drafter_load(const char * /*path*/) {
    return nullptr;
}
void llama_dflash_drafter_free(struct llama_dflash_drafter * /*drafter*/) {
}
int32_t llama_set_dflash(struct llama_context * /*ctx*/, struct llama_dflash_drafter * /*drafter*/) {
    return LLAMA_DFLASH_NOT_IMPLEMENTED;
}
int32_t llama_dflash_n_source_layers(const struct llama_dflash_drafter * /*drafter*/) { return -1; }
int32_t llama_dflash_block_size     (const struct llama_dflash_drafter * /*drafter*/) { return -1; }
llama_token llama_dflash_mask_token_id(const struct llama_dflash_drafter * /*drafter*/) { return -1; }
int32_t llama_dflash_swa_window     (const struct llama_dflash_drafter * /*drafter*/, int32_t /*layer*/) { return -2; }
enum llama_dflash_layer_type llama_dflash_layer_type_at(
        const struct llama_dflash_drafter * /*drafter*/, int32_t /*layer*/) {
    return LLAMA_DFLASH_LAYER_FULL_ATTENTION;
}
int32_t llama_dflash_draft(struct llama_context * /*ctx*/, llama_token /*anchor*/, int32_t /*pos*/,
                            llama_token * /*out*/, int32_t /*max*/) {
    return LLAMA_DFLASH_NOT_IMPLEMENTED;
}
int32_t llama_dflash_draft_batch(struct llama_context * /*ctx*/, int32_t /*n_slots*/,
                                  const llama_token * /*anchor_token_ids*/,
                                  const int32_t * /*anchor_positions*/,
                                  const llama_seq_id * /*seq_ids*/,
                                  llama_token * /*out*/, int32_t /*max*/) {
    return LLAMA_DFLASH_NOT_IMPLEMENTED;
}
int32_t llama_dflash_trim_extract(struct llama_context * /*ctx*/, int32_t /*ps*/, int32_t /*pe*/) {
    return LLAMA_DFLASH_NOT_IMPLEMENTED;
}
void llama_dflash_get_cycle_stats(
        const struct llama_context * /*ctx*/,
        int32_t * out_n_cycles) {
    if (out_n_cycles) *out_n_cycles = 0;
}

#endif // GGML_CUDA_DFLASH
