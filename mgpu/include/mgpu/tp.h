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

// Build the per-device attention sub-graph for CLIP (and any model that
// has no KV cache and no RoPE).
//
// Expects wq/wk/wv to be COL_PAR, wo to be ROW_PAR. The per-device loop
// handles: norm → Q/K/V matmuls (col-parallel) → permute → attention
// math (FlashAttention if fa_enabled, else KQ-matmul + softmax + KQV-
// matmul) → wo matmul (row-parallel) → collect. Reduces at the end.
//
// Mirrors the structural pattern at src/llama-build-context.cpp:3186-3220
// minus KV cache / RoPE / q-norm / k-norm.
struct ggml_tensor * mgpu_build_attn_megatron_clip(
        struct ggml_context * ctx,
        struct ggml_tensor  * attn_norm,        // nullable
        struct ggml_tensor  * input,
        struct ggml_tensor  * wq,
        struct ggml_tensor  * wq_b,             // nullable
        struct ggml_tensor  * wk,
        struct ggml_tensor  * wk_b,             // nullable
        struct ggml_tensor  * wv,
        struct ggml_tensor  * wv_b,             // nullable
        struct ggml_tensor  * wo,
        struct ggml_tensor  * wo_b,             // nullable
        struct ggml_tensor  * kq_mask,
        float                 kq_scale,
        float                 norm_eps,
        enum mgpu_norm_type   norm_type,
        int                   n_head,
        int                   n_head_kv,
        int                   fa_enabled,       // bool: use ggml_flash_attn_ext
        int                   il);

// (Phase 13) LM-style attention with KV cache, RoPE, q/k norms, and
// arch-variant config. Signature deferred until LM port begins; the
// header will be extended at that time.
//
// (Phase 14) LM-style MoE FFN. Same deferral.

#ifdef __cplusplus
}
#endif
