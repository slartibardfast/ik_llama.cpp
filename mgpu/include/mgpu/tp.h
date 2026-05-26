// libmgpu — Megatron-style tensor-parallel graph builders
//
// These are the GRAPH-CONSTRUCTION primitives for per-device sub-graphs
// (FFN, attention, MoE FFN). They consume ggml tensors decorated with
// per-device split state (ggml_split_tensor_t in tensor->extra, populated
// upstream by ggml_mgpu_alloc_split_tensors) and emit a complete
// per-device sub-graph terminated by ggml_reduce.
//
// Extracted (Phase 46 B.5e) from the private LM helpers at
// src/llama-build-context.cpp:1183-1303 (FFN), 1900-1985 (MoE FFN),
// 3186-3220+ (attention). The implementation in libmgpu is intended
// to be byte-identical to those originals so the LM determinism gates
// (B.6) PASS by construction.
#pragma once

#include "ggml.h"
#include "mgpu/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Per-device input fetcher.
//
// If `input` was produced by GGML_OP_REDUCE, returns input->src[id]
// (the per-device partial that fed the reduce — see ggml_reduce
// semantics at ggml/src/ggml.c). Lazily creates the per-device dst
// tensor on first read if it isn't already populated. Otherwise
// returns input as-is (single-device path).
//
// Mirrors src/llama-build-context.cpp:1183-1200.
struct ggml_tensor * mgpu_get_input_split(
        struct ggml_context * ctx,
        struct ggml_tensor  * input,
        int                   id);

// Per-device norm.
//
// If `norm` has split extras (typically REPLICATE so each device has the
// full norm weight), applies the appropriate norm using norm->extra
// ->splits[id]. Otherwise applies the single-device norm. Casts the
// result to F32 (matches LM convention at src/llama-build-context.cpp:1216).
//
// Mirrors src/llama-build-context.cpp:1202-1219.
struct ggml_tensor * mgpu_norm_split(
        struct ggml_context  * ctx,
        struct ggml_tensor   * cur,
        struct ggml_tensor   * norm,            // nullable — no norm
        float                  eps,
        enum mgpu_norm_type    type,
        int                    id);

// Build the per-device FFN sub-graph in Megatron-TP layout.
//
// Expects `up` and `gate` to be COL_PAR (split_dim=1; each device has
// full reduction axis, partial output features) and `down` to be ROW_PAR
// (split_dim=0; each device has partial reduction axis, full output
// features). Norm is typically REPLICATE.
//
// Emits per device: norm → up/gate matmuls (col-parallel; per-device
// output-feature slices) → activation (per-device) → down matmul
// (row-parallel; per-device partial full-shape output) → collect into
// ffn[id]. After the loop, reduces ffn[0..N-1] with GGML_OP_ADD and
// returns the reduce node.
//
// Mirrors src/llama-build-context.cpp:1240-1303.
struct ggml_tensor * mgpu_build_ffn_megatron(
        struct ggml_context * ctx,
        struct ggml_tensor  * input,
        struct ggml_tensor  * ffn_norm,         // nullable
        struct ggml_tensor  * up,
        struct ggml_tensor  * up_b,             // nullable
        struct ggml_tensor  * gate,             // nullable for gateless FFN
        struct ggml_tensor  * gate_b,           // nullable
        struct ggml_tensor  * down,
        struct ggml_tensor  * down_b,           // nullable
        float                 norm_eps,
        enum mgpu_norm_type   norm_type,
        enum mgpu_act         act,
        int                   il);              // layer index for naming/debug

// Config for mgpu_build_attn_megatron. Holds shape and arch-variant
// parameters that vary per call site. Grouped as a struct so the
// function signature stays manageable.
//
// RoPE handling:
//   rope_mode == -1            → no RoPE applied
//   rope_mode == 0 (NORMAL)    → standard RoPE (uses rope_n_dims, freqs)
//   rope_mode == 1 (NEOX)      → NeoX-style RoPE
//   rope_mode == GGML_ROPE_TYPE_VISION → M-RoPE with rope_sections
//                                       (qwen3vl, qwen2vl)
// (Values match GGML_ROPE_TYPE_* enum at ggml/include/ggml.h.)
struct mgpu_attn_config {
    // Attention shape.
    int n_head;
    int n_head_kv;
    int n_embd_head_k;
    int n_embd_head_v;
    int n_pos;                  // sequence length / image patch count

    // Norm.
    float                  norm_eps;
    enum mgpu_norm_type    norm_type;

    // Attention math.
    float kq_scale;
    int   fa_enabled;           // bool: prefer ggml_flash_attn_ext

    // RoPE. Set rope_mode = -1 to disable RoPE entirely.
    int   rope_mode;
    int   rope_n_dims;
    int   rope_sections[4];     // m-rope only
    int   rope_n_ctx_orig;
    float rope_freq_base;
    float rope_freq_scale;
    float rope_ext_factor;
    float rope_attn_factor;
    float rope_beta_fast;
    float rope_beta_slow;

    // For debug naming.
    int   il;
};

// Build the per-device attention sub-graph in Megatron-TP layout.
//
// Supports fused-QKV (qwen3vl style) where a single `wqkv` weight is
// col-parallel split and view-3d-sliced per-device into Q/K/V slices.
// The non-fused (separate wq/wk/wv) path is in a sibling function added
// in Phase 4b when needed by non-qwen3vl architectures.
//
// Per-device flow:
//   1. mgpu_get_input_split(input, id)
//   2. mgpu_norm_split(attn_norm)
//   3. 0xff marker (mirrors LM at line 2444)
//   4. QKV col-parallel matmul: ggml_mul_mat(wqkv->splits[id], cur)
//   5. View-3d-slice into Q_id, K_id, V_id at per-device offsets
//   6. Optional RoPE on Q_id and K_id (configured per cfg->rope_*)
//   7. Permute Q, K, V for attention math
//   8. Attention math: FA (if cfg->fa_enabled) or KQ matmul + softmax + KQV
//   9. wo row-parallel matmul: ggml_mul_mat(wo->splits[id], kqv_out)
//   10. Collect into attn[id]
// After loop: ggml_reduce(attn, n_device, GGML_OP_ADD).
//
// Mirrors the structural pattern at src/llama-build-context.cpp:3186-3220
// adapted for fused QKV. (LM uses separate wq/wk/wv; that variant is the
// sibling function deferred to Phase 4b / Session 2.)
struct ggml_tensor * mgpu_build_attn_megatron_fused_qkv(
        struct ggml_context             * ctx,
        const struct mgpu_attn_config   * cfg,
        struct ggml_tensor              * attn_norm,    // nullable
        struct ggml_tensor              * input,
        struct ggml_tensor              * wqkv,         // col-parallel
        struct ggml_tensor              * wqkv_b,       // nullable
        struct ggml_tensor              * wo,           // row-parallel
        struct ggml_tensor              * wo_b,         // nullable
        struct ggml_tensor              * positions,    // nullable when rope_mode == -1
        struct ggml_tensor              * kq_mask);     // nullable

// (Phase 4b — deferred) Separate-Q/K/V variant for non-fused-QKV CLIP
// architectures and LM. Header extended when implemented.
//
// (Phase 13) LM-style attention with KV cache. Deferred to Session 2.
//
// (Phase 14) LM-style MoE FFN. Deferred to Session 2.

#ifdef __cplusplus
}
#endif
