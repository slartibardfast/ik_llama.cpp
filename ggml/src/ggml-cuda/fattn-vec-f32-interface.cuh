#pragma once

#include "common.cuh"

void ggml_cuda_flash_attn_ext_vec_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_fattn_vec_f32_is_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);

// NP-invariant FA entry point for the GGML_OP_FLASH_ATTN_EXT_PER_SLOT_KV
// route. Forces cols_per_block=1 (per-row CTA) + parallel_blocks=1
// (no K-loop split) so the kernel partitioning is identical across
// all NP values. Routes through flash_attn_vec_ext_f32<256,256,1,F16,F16>
// (fp32 throughout). K/V are auto-dequanted from Q4_0 / Q8_0 / etc. to
// F16 by launch_fattn. See specs/deltanet/fattn-per-slot-kv-sm75.md §15.10.
void ggml_cuda_flash_attn_ext_vec_f32_strict_np_invariant(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
