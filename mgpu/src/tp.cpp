// libmgpu — Megatron-style tensor-parallel graph builders
//
// Phase 0 skeleton — function bodies are added in Phase 2 (helpers),
// Phase 3 (FFN builder), Phase 4 (CLIP-attn builder), Phase 13 (LM-attn),
// Phase 14 (MoE FFN).
//
// See mgpu/include/mgpu/tp.h for the contract.
#include "mgpu/tp.h"

#include "ggml.h"
#include "ggml-mgpu-split.h"

#include <cassert>
#include <cstdio>

// ---------------------------------------------------------------------------
// Phase 2 — helpers
// ---------------------------------------------------------------------------

// mgpu_get_input_split
//
// Byte-identical mirror of src/llama-build-context.cpp:1183-1200
// (llm_build_context::get_input_tensor_sm_graph).
//
// When `input` was produced by GGML_OP_REDUCE, its src[0..n_device-1]
// slots hold the per-device partial tensors that fed the reduce. This
// helper returns the slot for device `id`, lazily allocating a fresh
// dup tensor if the slot is still nullptr (which happens when the
// caller is constructing the per-device sub-graph that WILL feed the
// reduce — `input` here is the reduce node and we're populating its
// inputs). The op_params[4] bitmask tracks which device slots have
// been initialised.
//
// When `input` is not a REDUCE, returns input unchanged — both the
// single-device fallback and the first-of-chain case (where the input
// hasn't gone through a reduce yet, so all devices read from the same
// source via peer access).
extern "C" struct ggml_tensor * mgpu_get_input_split(
        struct ggml_context * ctx,
        struct ggml_tensor  * input,
        int                   id) {
    struct ggml_tensor * cur = input;
    if (input->op == GGML_OP_REDUCE) {
        struct ggml_tensor * view_src = input->view_src;
        GGML_ASSERT(view_src);
        cur = input->src[id];
        if (!cur) {
            GGML_ASSERT((input->op_params[4] & (1u << id)) == 0);
            cur = ggml_dup_tensor(ctx, input);
            input->src[id] = cur;
            input->op_params[4] |= (1u << id);
        }
        else if (cur == view_src) {
            cur = input;
        }
    }
    return cur;
}

// mgpu_norm_split
//
// Byte-identical mirror of src/llama-build-context.cpp:1202-1219
// (llm_build_context::do_split_norm) in the common case where
// `norm->extra` is set. Provides a safe fallback when extras are
// absent (applies the same fused norm op against the full weight
// tensor — equivalent to the single-device norm path).
//
// `eps` is passed through to ggml_fused_rms_norm / ggml_fused_norm
// directly; the caller is responsible for any hparams-side eps
// multipliers (LM does `scale_eps * hparams.f_norm_rms_eps` in
// llm_build_norm; the caller of mgpu_norm_split passes the
// already-multiplied value).
extern "C" struct ggml_tensor * mgpu_norm_split(
        struct ggml_context  * ctx,
        struct ggml_tensor   * cur,
        struct ggml_tensor   * norm,
        float                  eps,
        enum mgpu_norm_type    type,
        int                    id) {
    if (norm && norm->extra) {
        ggml_split_tensor_t * ext = (ggml_split_tensor_t *) norm->extra;
        GGML_ASSERT(ext->splits[id]);
        if (type == MGPU_NORM_LAYER) {
            cur = ggml_fused_norm(ctx, cur, ext->splits[id], eps);
        } else {
            cur = ggml_fused_rms_norm(ctx, cur, ext->splits[id], eps);
        }
    } else if (norm) {
        // No per-device extras — fall back to the same fused op against
        // the full norm weight. The matmul that follows will run on
        // whichever backend the sched picked for `cur`; this branch is
        // exercised when the caller passes a single-device norm into a
        // per-device loop (e.g. early-graph cases).
        if (type == MGPU_NORM_LAYER) {
            cur = ggml_fused_norm(ctx, cur, norm, eps);
        } else {
            cur = ggml_fused_rms_norm(ctx, cur, norm, eps);
        }
    }
    // If norm is nullptr, no normalization applied — the caller is
    // responsible for handling the no-norm path.

    // Cast to F32 — mirrors LM at line 1216-1218.
    if (cur->type != GGML_TYPE_F32) {
        cur = ggml_cast(ctx, cur, GGML_TYPE_F32);
    }
    return cur;
}

// ---------------------------------------------------------------------------
// Phase 3 — mgpu_build_ffn_megatron (stub)
// ---------------------------------------------------------------------------

extern "C" struct ggml_tensor * mgpu_build_ffn_megatron(
        struct ggml_context * /*ctx*/,
        struct ggml_tensor  * input,
        struct ggml_tensor  * /*ffn_norm*/,
        struct ggml_tensor  * /*up*/,
        struct ggml_tensor  * /*up_b*/,
        struct ggml_tensor  * /*gate*/,
        struct ggml_tensor  * /*gate_b*/,
        struct ggml_tensor  * /*down*/,
        struct ggml_tensor  * /*down_b*/,
        float                 /*norm_eps*/,
        enum mgpu_norm_type   /*norm_type*/,
        enum mgpu_act         /*act*/,
        int                   /*il*/) {
    fprintf(stderr, "mgpu_build_ffn_megatron: not yet implemented (Phase 0 stub)\n");
    GGML_ABORT("mgpu_build_ffn_megatron stub called — implement in Phase 3");
    return input; // unreachable
}

// ---------------------------------------------------------------------------
// Phase 4 — mgpu_build_attn_megatron_clip (stub)
// ---------------------------------------------------------------------------

extern "C" struct ggml_tensor * mgpu_build_attn_megatron_clip(
        struct ggml_context * /*ctx*/,
        struct ggml_tensor  * /*attn_norm*/,
        struct ggml_tensor  * input,
        struct ggml_tensor  * /*wq*/,
        struct ggml_tensor  * /*wq_b*/,
        struct ggml_tensor  * /*wk*/,
        struct ggml_tensor  * /*wk_b*/,
        struct ggml_tensor  * /*wv*/,
        struct ggml_tensor  * /*wv_b*/,
        struct ggml_tensor  * /*wo*/,
        struct ggml_tensor  * /*wo_b*/,
        struct ggml_tensor  * /*kq_mask*/,
        float                 /*kq_scale*/,
        float                 /*norm_eps*/,
        enum mgpu_norm_type   /*norm_type*/,
        int                   /*n_head*/,
        int                   /*n_head_kv*/,
        int                   /*fa_enabled*/,
        int                   /*il*/) {
    fprintf(stderr, "mgpu_build_attn_megatron_clip: not yet implemented (Phase 0 stub)\n");
    GGML_ABORT("mgpu_build_attn_megatron_clip stub called — implement in Phase 4");
    return input; // unreachable
}
