/**
 * ggml-fusion.h — Systematic op fusion registry for CPU + Vulkan.
 *
 * Single GGML_OP_FUSED op with fusion_id in op_params[0] dispatching
 * to per-backend kernels. Adding a new fusion:
 *   1. Add enum value here
 *   2. Write graph builder (ggml.c)
 *   3. Write CPU kernel (ggml.c) + Vulkan shader (optional)
 *   4. Add to backend dispatch switches
 *   5. Update model code to emit the fused op
 *
 * Ported from slartibardfast/llama.cpp phase25-decode-perf branch
 * (commits cc7612e92, 82ecb2c7a by David Connolly).
 */
#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_fusion_id {
    GGML_FUSION_NONE = 0,
    GGML_FUSION_GATE_PREP,      /* softplus(alpha + dt_bias) * ssm_a           */
    GGML_FUSION_SILU_MUL,       /* silu(x) * y                                */
    GGML_FUSION_COUNT,
};

/**
 * Fused gate preparation for SSM layers (Qwen3.5 DeltaNet).
 *
 * Combines: add(alpha, dt_bias) + softplus + mul(ssm_a) into one
 * dispatch, eliminating 2 intermediate tensor materializations.
 *
 * softplus(x) = x > 20 ? x : log(1 + exp(x))
 *
 * @param ctx     ggml context
 * @param alpha   [num_v_heads, n_seq_tokens, n_seqs] from matmul
 * @param dt_bias [num_v_heads] per-head bias (model weight)
 * @param ssm_a   [num_v_heads] per-head scale (model weight, typically negative)
 * @return        [num_v_heads, n_seq_tokens, n_seqs] = softplus(alpha + dt_bias) * ssm_a
 */
GGML_API struct ggml_tensor * ggml_fused_gate_prep(
        struct ggml_context * ctx,
        struct ggml_tensor  * alpha,
        struct ggml_tensor  * dt_bias,
        struct ggml_tensor  * ssm_a);

/**
 * Fused SiLU-gated multiply.
 *
 * Combines: silu(x) * y into one dispatch.
 * silu(x) = x / (1 + exp(-x))
 *
 * @param ctx  ggml context
 * @param x    input to SiLU (same shape as y)
 * @param y    multiplicand
 * @return     silu(x) * y (same shape as x and y)
 */
GGML_API struct ggml_tensor * ggml_fused_silu_mul(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * y);

#ifdef __cplusplus
}
#endif
