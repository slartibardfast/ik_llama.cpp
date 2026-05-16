// Manually authored for FIX-C v4 (per-slot-kv batch-invariance fix).
// Required for the vec_f16 perf comparison against vec_f32 at the
// production Q4_0 KV cache shape (Dq = Dv = 256). See yarn-agentic/
// PHASE_MMQ_Q4_0_AR16.md §6b CX.D.

#include "../fattn-vec-f16.cuh"

DECL_FATTN_VEC_F16_CASE(256, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0);
