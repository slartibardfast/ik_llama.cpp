// dflash-drafter-lm-head.cuh
//
// Launcher declaration for the DFlash drafter lm_head kernel — a BF16
// GEMV against the target's `output.weight` (shared via @SharedEmbedAndLMHead).
//
// Spec: specs/dflash/kernel-design.md §6.1 "Kernel boundary — lm_head".
//
// Per the production GGUF inspection (MEMORY 2026-05-13 IQ4_KS correction):
//   - Target's `output.weight` is BF16 [V=248320, D_emb=5120]
//   - Drafter's hidden state from dflash_drafter_forward is F16
//     [N_slots, BLOCK_SIZE, D_emb=5120]
//
// Kernel computes:
//   logits[row, col] = sum_k hidden[row, k] * lm_head_w[col, k]
//
// fp32 accumulator (Q-locked at T4 design Q&A) — fp16 hidden lifted to fp32,
// BF16 weight lifted to fp32, accumulated in fp32, output stays fp32.
//
// Allium witnesses:
//   - SharedEmbedAndLMHead       (kernel reads target's BF16 output.weight
//                                  via the same pointer as the target side)

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void dflash_drafter_lm_head_launch(
    const __half        * d_hidden,        // [n_rows, D_emb]
    const __nv_bfloat16 * d_lm_head_w,     // [V, D_emb]
    float               * d_logits,        // [n_rows, V] fp32 output
    int                   n_rows,
    int                   D_emb,
    int                   V,
    cudaStream_t          stream
);

#ifdef __cplusplus
}
#endif
