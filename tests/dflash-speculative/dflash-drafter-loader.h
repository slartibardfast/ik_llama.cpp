// dflash-drafter-loader.h
//
// Minimal drafter weight loader for T4 closure binding. Opens a DFlash
// drafter GGUF via gguf_init_from_file, allocates GPU buffers for each
// per-layer + global tensor, and returns an organised struct ready for
// the kernel pipeline.
//
// This is the test-side plumbing path. Full server-side integration
// (llama-model.cpp DFlash arch dispatch + llama_set_dflash) is T5 work.
//
// Spec: specs/dflash/kernel-design.md §10 T4.
//
// Allium witnesses:
//   - SharedEmbedAndLMHead         (caller provides target's token_embd
//                                    and output.weight as device pointers
//                                    — drafter handle == target handle by
//                                    construction)
//   - FeatureSourceFixedPerDeployment  (target_layer_ids read from
//                                    drafter GGUF metadata at load time)

#pragma once

#include "ggml.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace dflash_reference {

struct DrafterWeights {
    // Drafter dims (read from drafter GGUF kv).
    int n_layers          = 0;   // = 5
    int hidden_size       = 0;   // = 5120
    int intermediate_size = 0;   // = 17408
    int n_q_heads         = 0;   // = 32
    int n_kv_heads        = 0;   // = 8
    int head_dim          = 0;   // = 128
    int vocab_size        = 0;   // = 248320
    int sliding_window    = 0;   // = 2048
    int block_size        = 0;   // = 16 (model's max — operating point is 4)
    int mask_token_id     = 0;   // = 248070
    float rope_theta      = 0.0f; // = 10000000.0
    float rms_norm_eps    = 0.0f; // = 1e-6

    std::vector<int> target_layer_ids;     // = [1, 16, 31, 46, 61]
    std::vector<int> layer_types;          // 0 = SWA, 1 = full

    // Per-layer device pointers (n_layers each).
    std::vector<const __half *> attn_norm;     // F32 stored, cast to __half on load? No — keep F32.
    std::vector<const __half *> attn_q;        // F16 [n_q*head_dim, hidden_size]
    std::vector<const __half *> attn_q_norm;   // F32 [head_dim]
    std::vector<const __half *> attn_k;        // F16 [n_kv*head_dim, hidden_size]
    std::vector<const __half *> attn_k_norm;   // F32 [head_dim]
    std::vector<const __half *> attn_v;        // F16 [n_kv*head_dim, hidden_size]
    std::vector<const __half *> attn_output;   // F16 [hidden_size, n_q*head_dim]
    std::vector<const __half *> ffn_norm;      // F32 [hidden_size]
    std::vector<const __half *> ffn_gate;      // F16 [intermediate, hidden_size]
    std::vector<const __half *> ffn_up;        // F16 [intermediate, hidden_size]
    std::vector<const __half *> ffn_down;      // F16 [hidden_size, intermediate]

    // DFlash-specific tensors.
    const __half * dflash_fc          = nullptr;  // F16 [n_layers*hidden_size, hidden_size]
    const __half * dflash_hidden_norm = nullptr;  // F32 in GGUF, cast to F16 at load
    const __half * output_norm        = nullptr;  // F32 in GGUF, cast to F16 at load — used by lm_head pre-step

    // Backing allocations to free on destroy.
    std::vector<void *> gpu_buffers;
    struct gguf_context * gguf_ctx  = nullptr;
    struct ggml_context * ggml_ctx  = nullptr;
};

// Upload a host tensor at `src` of `n_bytes` bytes to a fresh GPU buffer.
// Records the pointer in `w.gpu_buffers` for later free. Returns the
// device pointer.
inline void * _upload(DrafterWeights & w, const void * src, std::size_t n_bytes) {
    void * dev = nullptr;
    cudaError_t err = cudaMalloc(&dev, n_bytes);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(%zu) failed: %s\n", n_bytes, cudaGetErrorString(err));
        return nullptr;
    }
    cudaMemcpy(dev, src, n_bytes, cudaMemcpyHostToDevice);
    w.gpu_buffers.push_back(dev);
    return dev;
}

// Locate a tensor by name in the loaded ggml context, return its data
// pointer and byte size. Returns nullptr if not found.
inline const void * _find_tensor(struct ggml_context * ctx, const char * name, std::size_t * out_bytes) {
    struct ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (t == nullptr) {
        if (out_bytes) *out_bytes = 0;
        return nullptr;
    }
    if (out_bytes) *out_bytes = ggml_nbytes(t);
    return t->data;
}

// Read a fp32 metadata key (returns 0.0 if not found).
inline float _kv_f32(struct gguf_context * g, const char * key) {
    const int idx = gguf_find_key(g, key);
    if (idx < 0) return 0.0f;
    return gguf_get_val_f32(g, idx);
}
// Read a uint32 metadata key (returns 0 if not found).
inline uint32_t _kv_u32(struct gguf_context * g, const char * key) {
    const int idx = gguf_find_key(g, key);
    if (idx < 0) return 0;
    return gguf_get_val_u32(g, idx);
}
// Read an array of int32 (e.g., dflash.target_layer_ids).
inline std::vector<int> _kv_array_i32(struct gguf_context * g, const char * key) {
    std::vector<int> out;
    const int idx = gguf_find_key(g, key);
    if (idx < 0) return out;
    const int n = gguf_get_arr_n(g, idx);
    const int32_t * data = static_cast<const int32_t *>(gguf_get_arr_data(g, idx));
    out.assign(data, data + n);
    return out;
}
// Read an array of strings into a numeric encoding for layer_types:
// "sliding_attention" → 0, "full_attention" → 1, anything else → -1.
inline std::vector<int> _kv_layer_types_enum(struct gguf_context * g, const char * key) {
    std::vector<int> out;
    const int idx = gguf_find_key(g, key);
    if (idx < 0) return out;
    const int n = gguf_get_arr_n(g, idx);
    out.reserve(n);
    for (int i = 0; i < n; ++i) {
        const char * s = gguf_get_arr_str(g, idx, i);
        if (s == nullptr) { out.push_back(-1); continue; }
        if (std::strcmp(s, "sliding_attention") == 0)      out.push_back(0);
        else if (std::strcmp(s, "full_attention") == 0)    out.push_back(1);
        else                                                out.push_back(-1);
    }
    return out;
}

// Load drafter GGUF and upload all tensors to GPU. Returns true on
// success, false on any IO / shape error.
inline bool load_drafter(const char * gguf_path, DrafterWeights & w) {
    struct gguf_init_params params{};
    params.no_alloc = false;
    params.ctx      = &w.ggml_ctx;
    w.gguf_ctx = gguf_init_from_file(gguf_path, params);
    if (w.gguf_ctx == nullptr) {
        std::fprintf(stderr, "load_drafter: gguf_init_from_file failed for %s\n", gguf_path);
        return false;
    }

    // Read metadata from the drafter GGUF. All scalar dflash.* keys are
    // UINT32; rope.freq_base and layer_norm_rms_epsilon are FLOAT32;
    // target_layer_ids is an INT32 array; layer_types is a STRING array
    // ("sliding_attention" / "full_attention"). The drafter does NOT
    // store its own vocab_size — it shares vocab with the target.
    w.n_layers          = static_cast<int>(_kv_u32(w.gguf_ctx, "dflash.block_count"));
    w.hidden_size       = static_cast<int>(_kv_u32(w.gguf_ctx, "dflash.embedding_length"));
    w.intermediate_size = static_cast<int>(_kv_u32(w.gguf_ctx, "dflash.feed_forward_length"));
    w.n_q_heads         = static_cast<int>(_kv_u32(w.gguf_ctx, "dflash.attention.head_count"));
    w.n_kv_heads        = static_cast<int>(_kv_u32(w.gguf_ctx, "dflash.attention.head_count_kv"));
    w.sliding_window    = static_cast<int>(_kv_u32(w.gguf_ctx, "dflash.attention.sliding_window"));
    w.block_size        = static_cast<int>(_kv_u32(w.gguf_ctx, "dflash.block_size"));
    w.mask_token_id     = static_cast<int>(_kv_u32(w.gguf_ctx, "dflash.mask_token_id"));
    w.rope_theta        = _kv_f32(w.gguf_ctx, "dflash.rope.freq_base");
    w.rms_norm_eps      = _kv_f32(w.gguf_ctx, "dflash.attention.layer_norm_rms_epsilon");

    // head_dim from explicit dflash.attention.key_length (UINT32).
    w.head_dim = static_cast<int>(_kv_u32(w.gguf_ctx, "dflash.attention.key_length"));
    if (w.head_dim == 0 && w.n_q_heads > 0 && w.hidden_size > 0) {
        w.head_dim = w.hidden_size / w.n_q_heads;
    }

    // vocab_size: not stored in drafter GGUF (shared with target). Caller
    // sets this from the target side or leaves it at 0 if not needed.
    w.vocab_size = 0;

    w.target_layer_ids = _kv_array_i32(w.gguf_ctx, "dflash.target_layer_ids");
    w.layer_types      = _kv_layer_types_enum(w.gguf_ctx, "dflash.layer_types");

    if (w.n_layers == 0 || w.hidden_size == 0 || w.intermediate_size == 0) {
        std::fprintf(stderr, "load_drafter: missing metadata. n_layers=%d hidden=%d ffn=%d\n",
                     w.n_layers, w.hidden_size, w.intermediate_size);
        return false;
    }

    auto upload_tensor = [&](const char * name) -> const __half * {
        std::size_t nb = 0;
        const void * src = _find_tensor(w.ggml_ctx, name, &nb);
        if (src == nullptr) {
            std::fprintf(stderr, "load_drafter: tensor not found: %s\n", name);
            return nullptr;
        }
        return static_cast<const __half *>(_upload(w, src, nb));
    };

    // Cast F32 → F16 then upload. For norm-weight tensors which the
    // drafter GGUF stores as F32 but the kernels consume as __half*.
    // Norm weights are in [~0.5, ~2.0] range — fp16 has plenty of
    // precision; F32→F16 cast is essentially lossless here.
    auto upload_f32_as_f16 = [&](const char * name) -> const __half * {
        struct ggml_tensor * tn = ggml_get_tensor(w.ggml_ctx, name);
        if (tn == nullptr) {
            std::fprintf(stderr, "load_drafter: tensor not found: %s\n", name);
            return nullptr;
        }
        if (tn->type != GGML_TYPE_F32) {
            // Already not F32 — fall through to raw upload.
            return static_cast<const __half *>(_upload(w, tn->data, ggml_nbytes(tn)));
        }
        const std::size_t n_elems = ggml_nelements(tn);
        std::vector<__half> tmp(n_elems);
        const float * src = static_cast<const float *>(tn->data);
        for (std::size_t i = 0; i < n_elems; ++i) {
            tmp[i] = __float2half(src[i]);
        }
        return static_cast<const __half *>(_upload(w, tmp.data(), n_elems * sizeof(__half)));
    };

    w.attn_norm.resize(w.n_layers);
    w.attn_q.resize(w.n_layers);
    w.attn_q_norm.resize(w.n_layers);
    w.attn_k.resize(w.n_layers);
    w.attn_k_norm.resize(w.n_layers);
    w.attn_v.resize(w.n_layers);
    w.attn_output.resize(w.n_layers);
    w.ffn_norm.resize(w.n_layers);
    w.ffn_gate.resize(w.n_layers);
    w.ffn_up.resize(w.n_layers);
    w.ffn_down.resize(w.n_layers);

    char namebuf[64];
    for (int l = 0; l < w.n_layers; ++l) {
        // Norm weights are F32 in the drafter GGUF — must cast to F16
        // before the kernels consume them as __half*. The other weights
        // (q/k/v/o/gate/up/down) are F16 already.
        std::snprintf(namebuf, sizeof(namebuf), "blk.%d.attn_norm.weight",      l); w.attn_norm[l]    = upload_f32_as_f16(namebuf);
        std::snprintf(namebuf, sizeof(namebuf), "blk.%d.attn_q.weight",         l); w.attn_q[l]       = upload_tensor(namebuf);
        std::snprintf(namebuf, sizeof(namebuf), "blk.%d.attn_q_norm.weight",    l); w.attn_q_norm[l]  = upload_f32_as_f16(namebuf);
        std::snprintf(namebuf, sizeof(namebuf), "blk.%d.attn_k.weight",         l); w.attn_k[l]       = upload_tensor(namebuf);
        std::snprintf(namebuf, sizeof(namebuf), "blk.%d.attn_k_norm.weight",    l); w.attn_k_norm[l]  = upload_f32_as_f16(namebuf);
        std::snprintf(namebuf, sizeof(namebuf), "blk.%d.attn_v.weight",         l); w.attn_v[l]       = upload_tensor(namebuf);
        std::snprintf(namebuf, sizeof(namebuf), "blk.%d.attn_output.weight",    l); w.attn_output[l]  = upload_tensor(namebuf);
        std::snprintf(namebuf, sizeof(namebuf), "blk.%d.ffn_norm.weight",       l); w.ffn_norm[l]     = upload_f32_as_f16(namebuf);
        std::snprintf(namebuf, sizeof(namebuf), "blk.%d.ffn_gate.weight",       l); w.ffn_gate[l]     = upload_tensor(namebuf);
        std::snprintf(namebuf, sizeof(namebuf), "blk.%d.ffn_up.weight",         l); w.ffn_up[l]       = upload_tensor(namebuf);
        std::snprintf(namebuf, sizeof(namebuf), "blk.%d.ffn_down.weight",       l); w.ffn_down[l]     = upload_tensor(namebuf);

        if (!w.attn_norm[l] || !w.attn_q[l] || !w.attn_q_norm[l] || !w.attn_k[l] ||
            !w.attn_k_norm[l] || !w.attn_v[l] || !w.attn_output[l] || !w.ffn_norm[l] ||
            !w.ffn_gate[l] || !w.ffn_up[l] || !w.ffn_down[l]) {
            std::fprintf(stderr, "load_drafter: missing tensor at layer %d\n", l);
            return false;
        }
    }
    w.dflash_fc          = upload_tensor("dflash_fc.weight");
    // dflash_hidden_norm is F32 in the GGUF; cast to F16 for the
    // combine_features kernel which expects __half * weight.
    w.dflash_hidden_norm = upload_f32_as_f16("dflash_hidden_norm.weight");
    w.output_norm = upload_f32_as_f16("output_norm.weight");
    return true;
}

inline void free_drafter(DrafterWeights & w) {
    for (void * p : w.gpu_buffers) {
        if (p) cudaFree(p);
    }
    w.gpu_buffers.clear();
    if (w.gguf_ctx) { gguf_free(w.gguf_ctx); w.gguf_ctx = nullptr; }
    if (w.ggml_ctx) { ggml_free(w.ggml_ctx); w.ggml_ctx = nullptr; }
}

} // namespace dflash_reference
