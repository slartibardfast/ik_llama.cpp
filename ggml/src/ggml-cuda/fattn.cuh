#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Per-slot KV occupancy variant (sm_75-only): consumes src[5]=slot_seq_lens
// to bound the per-slot K-loop in the FA kernel, restoring np>1 determinism.
// Implementation in fattn-per-slot-kv-sm75.cu.
void ggml_cuda_flash_attn_ext_per_slot_kv_sm75(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_fattn_is_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
