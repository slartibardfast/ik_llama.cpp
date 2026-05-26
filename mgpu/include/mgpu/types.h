// libmgpu — common enums for graph-mode tensor parallelism
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// How a weight tensor is partitioned across devices in graph-mode TP.
//
//   NONE      — not partitioned; tensor stays single-device (in ctx_data,
//               default cuda buft for backend_ptrs[0]).
//   REPLICATE — full tensor allocated on every device (split_dim=-1 in
//               ggml_mgpu_alloc_split_tensors). Used for norms, biases,
//               embeddings, anything that flows into non-matmul ops or
//               must be locally addressable.
//   COL_PAR   — column-parallel: split output features (split_dim=1).
//               Each device gets [k_full, n/N]. Input is replicated;
//               output is per-device (no reduce needed after).
//   ROW_PAR   — row-parallel: split reduction axis (split_dim=0).
//               Each device gets [k/N, n_full]. Input must be sliced
//               to k/N rows; output is partial full-shape; REDUCE
//               sums across devices.
enum mgpu_split_kind {
    MGPU_SPLIT_NONE      = 0,
    MGPU_SPLIT_REPLICATE = 1,
    MGPU_SPLIT_COL_PAR   = 2,
    MGPU_SPLIT_ROW_PAR   = 3,
};

// Norm variant for mgpu_norm_split and the per-device norm leg of
// mgpu_build_ffn_megatron / mgpu_build_attn_megatron_*.
enum mgpu_norm_type {
    MGPU_NORM_RMS   = 0,
    MGPU_NORM_LAYER = 1,
};

// Activation in the FFN inner stage. Matches LM's LLM_FFN_* enum but
// kept independent so libmgpu doesn't depend on libllama headers.
enum mgpu_act {
    MGPU_ACT_SILU       = 0,
    MGPU_ACT_GELU       = 1,
    MGPU_ACT_GELU_ERF   = 2,
    MGPU_ACT_GELU_QUICK = 3,
    MGPU_ACT_RELU       = 4,
};

#ifdef __cplusplus
}
#endif
