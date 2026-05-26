// libmgpu — weight-name → split-kind classifier
//
// Centralises the per-weight tensor-parallel layout choice that was
// previously spread across src/llama-load-tensors.cpp (LM) and the
// ad-hoc is_splittable in examples/mtmd/clip.cpp (CLIP). Both modalities
// now classify weights through this single function so the LM and the
// vision encoder agree on what's column-parallel, row-parallel,
// replicated, or unsplit.
#pragma once

#include "mgpu/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the split kind appropriate for a weight tensor identified by
// its GGUF name (e.g. "blk.0.attn_qkv.weight", "v.blk.5.ffn_down.weight",
// "v.position_embd.weight"). Matches by name SUFFIX so the same function
// works for LM weights (no prefix), CLIP vision weights ("v." prefix),
// audio weights, and any future modality that follows GGUF conventions.
//
// The caller is responsible for size cutoffs (B.5b's 1-MiB floor) and
// for honouring the returned kind via ggml_mgpu_alloc_split_tensors
// with the appropriate split_dim (-1 / 1 / 0) or by skipping allocation.
enum mgpu_split_kind mgpu_classify_weight(const char * name);

#ifdef __cplusplus
}
#endif
