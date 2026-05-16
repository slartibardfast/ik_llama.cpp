// Manually authored for FIX-C v4 (per-slot-kv batch-invariance fix).
// Required to route ggml_cuda_flash_attn_ext_per_slot_kv_sm75 to the
// per-row CTA F32 vec kernel at the production Q4_0 KV cache shape
// (Dq = Dv = 256). See yarn-agentic/PHASE_MMQ_Q4_0_AR16.md §6b CX.D.

#include "../fattn-vec-f32.cuh"

DECL_FATTN_VEC_F32_CASE(256, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
