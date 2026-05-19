// dflash-drafter-lm-head.cu
//
// F16 GEMV against target's shared `output.weight`. Produces drafter
// logits for argmax_match (next kernel in the pipeline).
//
// Spec: kernel-design.md §6.1 "Kernel boundary — lm_head" + §6.1.A.
//
// Design (revised 2026-05-19 per spec §6.1.A):
//   - Forwards to `ggml_cuda_mul_mat_f16_pinned` via `dflash_gemm_npc`,
//     same NPC-safe HMMA m16n8k16 dispatch used by the 35 drafter forward
//     GEMMs. F16 act × F16 weight → F32 logits, fp32 accumulator in
//     HMMA fragments. Byte-identity-by-construction across batch
//     composition.
//
// Shape (Qwen 3.6 27B): n_rows = N_slots × BLOCK_SIZE (4 per slot in
// production), D_emb = 5120, V = 248320. M tiny, N huge → tall-skinny;
// pinned handles this trivially (~20-50 ms/call expected vs the
// 1648 ms/call of the old scalar fp32 GEMV which only utilized
// n_rows SMs out of 72).
//
// Numerics:
//   - F16 weight × F16 hidden → fp32 accumulator inside m16n8k16 fragments.
//   - Output fp32 (logits — preserves precision for closure binding vs vLLM).
//
// Allium witness: SharedEmbedAndLMHead (load-time pointer equality
// asserted in the loader; kernel just reads the F16 buffer).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "dflash-drafter-lm-head.cuh"
#include "dflash-gemm.cuh"

extern "C" void dflash_drafter_lm_head_launch(
    const __half * d_hidden,
    const __half * d_lm_head_w,
    float        * d_logits,
    int            n_rows,
    int            D_emb,
    int            V,
    cudaStream_t   stream)
{
    if (d_hidden == nullptr || d_lm_head_w == nullptr || n_rows <= 0) {
        const std::size_t n_bytes =
            static_cast<std::size_t>(n_rows) * V * sizeof(float);
        if (n_rows > 0 && d_logits != nullptr) {
            cudaMemsetAsync(d_logits, 0, n_bytes, stream);
        }
        return;
    }

    // Same NPC-safe dispatch as drafter forward GEMMs.
    // Map: weight[V, D_emb] · act[n_rows, D_emb]^T → logits[n_rows, V].
    dflash_gemm_npc(
        /*weight=*/d_lm_head_w,
        /*act=*/d_hidden,
        /*dst_f32=*/d_logits,
        /*K=*/D_emb,
        /*N_cols=*/V,
        /*n_rows=*/n_rows,
        stream);
}
