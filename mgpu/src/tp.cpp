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
#include <vector>

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
// Phase 3 — mgpu_build_ffn_megatron
// ---------------------------------------------------------------------------

// Internal: apply the gate-fused activation against (gate_out, up_out) when
// both up and gate matmuls are present.
static ggml_tensor * mgpu_activation_gated(
        ggml_context * ctx,
        ggml_tensor  * gate_out,
        ggml_tensor  * up_out,
        mgpu_act       act) {
    switch (act) {
        case MGPU_ACT_SILU:       return ggml_swiglu_split    (ctx, gate_out, up_out);
        case MGPU_ACT_GELU:       return ggml_geglu_split     (ctx, gate_out, up_out);
        case MGPU_ACT_GELU_ERF:   return ggml_geglu_erf_split (ctx, gate_out, up_out);
        case MGPU_ACT_GELU_QUICK: return ggml_geglu_quick_split(ctx, gate_out, up_out);
        case MGPU_ACT_RELU:       return ggml_mul(ctx, up_out, ggml_relu(ctx, gate_out));
    }
    GGML_ABORT("mgpu_activation_gated: unhandled act");
    return up_out;
}

// Internal: apply the unary activation against up_out when there's no gate.
static ggml_tensor * mgpu_activation_unary(
        ggml_context * ctx,
        ggml_tensor  * up_out,
        mgpu_act       act) {
    switch (act) {
        case MGPU_ACT_SILU:       return ggml_silu      (ctx, up_out);
        case MGPU_ACT_GELU:       return ggml_gelu      (ctx, up_out);
        case MGPU_ACT_GELU_ERF:   return ggml_gelu_erf  (ctx, up_out);
        case MGPU_ACT_GELU_QUICK: return ggml_gelu_quick(ctx, up_out);
        case MGPU_ACT_RELU:       return ggml_relu      (ctx, up_out);
    }
    GGML_ABORT("mgpu_activation_unary: unhandled act");
    return up_out;
}

// mgpu_build_ffn_megatron
//
// Mirrors the per-device FFN body at src/llama-build-context.cpp:1240-1303.
//
// Two execution paths:
//   PATH A (fused, byte-identical to LM line 1240-1263 when conditions match):
//     ggml_fused_up_gate(split_u, split_g, cur, unary_op). Requires:
//       gate present; no biases; activation ∈ {SILU, RELU, GELU}.
//   PATH B (general): separate up/gate matmuls (with optional biases), then
//     ggml_*glu_split for gated activations or ggml_act unary for ungated.
//     Used by CLIP (qwen3vl has biases) and any LM path that has biases or
//     a non-fused activation.
//
// In both paths the post-activation matmul against `down->splits[id]` is the
// row-parallel half — each device produces a PARTIAL full-shape output. The
// final ggml_reduce sums across devices.
extern "C" struct ggml_tensor * mgpu_build_ffn_megatron(
        struct ggml_context * ctx,
        struct ggml_tensor  * input,
        struct ggml_tensor  * ffn_norm,
        struct ggml_tensor  * up,
        struct ggml_tensor  * up_b,
        struct ggml_tensor  * gate,
        struct ggml_tensor  * gate_b,
        struct ggml_tensor  * down,
        struct ggml_tensor  * down_b,
        float                 norm_eps,
        enum mgpu_norm_type   norm_type,
        enum mgpu_act         act,
        int                   /*il*/) {

    GGML_ASSERT(up   && up->extra);
    GGML_ASSERT(down && down->extra);

    ggml_split_tensor_t * u = (ggml_split_tensor_t *) up->extra;
    ggml_split_tensor_t * d = (ggml_split_tensor_t *) down->extra;
    ggml_split_tensor_t * g = (gate && gate->extra) ? (ggml_split_tensor_t *) gate->extra : nullptr;
    GGML_ASSERT(u->n_device == d->n_device);
    if (g) GGML_ASSERT(u->n_device == g->n_device);
    const int n_device = u->n_device;

    // PATH A eligibility: matches LM line 1240-1242 fast-path conditions.
    const bool can_fuse =
        g != nullptr &&
        up_b == nullptr && gate_b == nullptr && down_b == nullptr &&
        (act == MGPU_ACT_SILU || act == MGPU_ACT_RELU || act == MGPU_ACT_GELU);

    ggml_unary_op fused_unary = GGML_UNARY_OP_SILU;
    if (can_fuse) {
        switch (act) {
            case MGPU_ACT_SILU: fused_unary = GGML_UNARY_OP_SILU; break;
            case MGPU_ACT_RELU: fused_unary = GGML_UNARY_OP_RELU; break;
            case MGPU_ACT_GELU: fused_unary = GGML_UNARY_OP_GELU; break;
            default: break;
        }
    }

    std::vector<ggml_tensor *> ffn(n_device, nullptr);
    int id_last = -1;

    for (int id = 0; id < n_device; ++id) {
        ggml_tensor * split_u = u->splits[id];
        ggml_tensor * split_d = d->splits[id];
        ggml_tensor * split_g = g ? g->splits[id] : nullptr;
        // Either all per-device slices are present or none are (mirrors LM
        // line 1256 invariant). When none, this device skips entirely.
        GGML_ASSERT((!split_u && !split_d && (!g || !split_g)) ||
                    ( split_u &&  split_d && (!g ||  split_g)));
        if (!split_u) continue;

        ggml_tensor * cur = mgpu_get_input_split(ctx, input, id);
        cur = mgpu_norm_split(ctx, cur, ffn_norm, norm_eps, norm_type, id);
        // sched-partition marker (mirrors LM line 1260-1262).
        if (input->op != GGML_OP_REDUCE) {
            cur->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] = 0xff;
        }

        if (can_fuse) {
            // PATH A — byte-identical to LM line 1263.
            cur = ggml_fused_up_gate(ctx, split_u, split_g, cur, fused_unary);
        } else {
            // PATH B — separate matmuls, with optional biases.
            ggml_tensor * up_out = ggml_mul_mat(ctx, split_u, cur);
            if (up_b) up_out = ggml_add(ctx, up_out, up_b);

            if (g) {
                ggml_tensor * gate_out = ggml_mul_mat(ctx, split_g, cur);
                if (gate_b) gate_out = ggml_add(ctx, gate_out, gate_b);
                cur = mgpu_activation_gated(ctx, gate_out, up_out, act);
            } else {
                cur = mgpu_activation_unary(ctx, up_out, act);
            }
        }

        // Down matmul (row-parallel) + optional bias.
        cur = ggml_mul_mat(ctx, split_d, cur);
        if (down_b) cur = ggml_add(ctx, cur, down_b);

        ffn[id] = cur;
        id_last = id;
    }
    GGML_ASSERT(id_last >= 0);

    // Reduce across devices (mirrors LM line 1299).
    return ggml_reduce(ctx, ffn.data(), n_device, GGML_OP_ADD);
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
