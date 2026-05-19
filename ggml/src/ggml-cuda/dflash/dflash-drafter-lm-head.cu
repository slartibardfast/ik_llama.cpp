// dflash-drafter-lm-head.cu
//
// F16 GEMV against target's shared `output.weight`. Produces drafter
// logits for argmax_match (next kernel in the pipeline).
//
// Spec: kernel-design.md §6.1 "Kernel boundary — lm_head".
//
// Design:
//   - One CTA per (slot, position) row of the hidden state.
//   - Block: 256 threads (8 warps).
//   - Stage hidden[row, :] into SMEM as fp32 (D_emb=5120 × 4B = 20 KiB).
//   - Each thread strides across output columns; per column does a
//     scalar fp32 dot product over D_emb against the F16 weight row.
//
// Numerics (locked at Q&A 2026-05-13; lm_head recast 2026-05-18):
//   - fp32 accumulator throughout.
//   - F16 hidden → fp32 via __half2float.
//   - F16 weight → fp32 via __half2float (was BF16 pre-recast; T1 cast
//     is a precision improvement for the Band-A absmax=0.36 weight —
//     fp16 has 10 mantissa bits vs bf16's 7 at this magnitude).
//   - Output fp32 (no fp16 cast — preserves precision for the 1e-5 NMSE
//     closure binding vs vLLM).
//
// Allium witness: SharedEmbedAndLMHead (load-time pointer equality
// asserted in the loader; kernel just reads the F16 buffer).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "dflash-drafter-lm-head.cuh"

namespace {

constexpr int LM_HEAD_THREADS = 256;

__global__ void dflash_drafter_lm_head_kernel(
    const __half * __restrict__ hidden,      // [n_rows, D_emb]
    const __half * __restrict__ lm_head_w,   // [V, D_emb] — F16
    float        * __restrict__ logits,      // [n_rows, V]
    int                         D_emb,
    int                         V)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ float hidden_smem[];

    const __half * row_h = hidden + static_cast<std::size_t>(row) * D_emb;
    for (int i = tid; i < D_emb; i += blockDim.x) {
        hidden_smem[i] = __half2float(row_h[i]);
    }
    __syncthreads();

    float * row_o = logits + static_cast<std::size_t>(row) * V;
    for (int col = tid; col < V; col += blockDim.x) {
        const __half * row_w = lm_head_w + static_cast<std::size_t>(col) * D_emb;
        float acc = 0.0f;
        for (int k = 0; k < D_emb; ++k) {
            acc += hidden_smem[k] * __half2float(row_w[k]);
        }
        row_o[col] = acc;
    }
}

} // anonymous namespace

extern "C" void dflash_drafter_lm_head_launch(
    const __half * d_hidden,
    const __half * d_lm_head_w,
    float               * d_logits,
    int                   n_rows,
    int                   D_emb,
    int                   V,
    cudaStream_t          stream)
{
    if (d_hidden == nullptr || d_lm_head_w == nullptr || n_rows <= 0) {
        const std::size_t n_bytes =
            static_cast<std::size_t>(n_rows) * V * sizeof(float);
        if (n_rows > 0 && d_logits != nullptr) {
            cudaMemsetAsync(d_logits, 0, n_bytes, stream);
        }
        return;
    }

    const dim3 grid(n_rows);
    const dim3 block(LM_HEAD_THREADS);
    const std::size_t smem_bytes = static_cast<std::size_t>(D_emb) * sizeof(float);
    dflash_drafter_lm_head_kernel<<<grid, block, smem_bytes, stream>>>(
        d_hidden, d_lm_head_w, d_logits, D_emb, V);
}
