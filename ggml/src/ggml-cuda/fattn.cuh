#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Per-row K-loop bound variant (sm_75-only): consumes src[5]=per_row_k_bound
// (i32, length q->ne[1]) to bound the K-loop per query row, restoring
// np>1 determinism. Per spec §15.6 dispatch routes to the wmma_f16+bound
// variant; the bespoke fattn-per-slot-kv-sm75.cu kernels remain compiled
// for unit-test reference only.
void ggml_cuda_flash_attn_ext_per_slot_kv_sm75(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_fattn_is_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
