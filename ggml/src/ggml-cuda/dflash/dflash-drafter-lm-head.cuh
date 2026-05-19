// dflash-drafter-lm-head.cuh
//
// Launcher declaration for the DFlash drafter lm_head kernel — an F16
// GEMV against the target's `output.weight` (shared via @SharedEmbedAndLMHead).
//
// Spec: specs/dflash/kernel-design.md §6.1 "Kernel boundary — lm_head".
//
// Target tensor types (post-recast 2026-05-18):
//   - Target's `output.weight` is F16 [V=248320, D_emb=5120] — recast
//     BF16→F16 at T1 (absmax=0.36 Band-A; lossless / mantissa-improving).
//     Target GGUF: qwen3.6-27b-V-F1.T1.lm_head-f16.gguf
//   - Drafter's hidden state from dflash_drafter_forward is F16
//     [N_slots, BLOCK_SIZE, D_emb=5120]
//
// Kernel computes:
//   logits[row, col] = sum_k hidden[row, k] * lm_head_w[col, k]
//
// fp32 accumulator (Q-locked at T4 design Q&A) — fp16 hidden lifted to fp32,
// F16 weight lifted to fp32, accumulated in fp32, output stays fp32.
//
// Allium witnesses:
//   - SharedEmbedAndLMHead       (kernel reads target's F16 output.weight
//                                  via the same pointer as the target side)

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void dflash_drafter_lm_head_launch(
    const __half * d_hidden,        // [n_rows, D_emb]
    const __half * d_lm_head_w,     // [V, D_emb] — F16 (T1-recast)
    float        * d_logits,        // [n_rows, V] fp32 output
    int            n_rows,
    int            D_emb,
    int            V,
    cudaStream_t   stream
);

#ifdef __cplusplus
}
#endif
