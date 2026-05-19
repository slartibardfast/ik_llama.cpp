// dflash-gemm.cuh
//
// NPC-safe GEMM dispatch for the DFlash drafter forward pipeline.
//
// Forwards to ggml_cuda_mul_mat_f16_pinned (HMMA m16n8k16, byte-identical
// regardless of total M — see mul-mat-f16-pinned.cuh). Canonical dispatch
// per specs/dflash/kernel-design.md §6.1.A.
//
// Layout (ggml convention, same as the legacy scalar gemm_row_x_col_kernel):
//   - weight  [N_cols, K]  row-major: data[col * K + k]
//   - act     [n_rows, K]  row-major: data[row * K + k]
//   - dst_f32 [n_rows, N_cols] row-major: data[row * N_cols + col]
//
// Math:
//   dst_f32[row, col] = sum_k act[row, k] * weight[col, k]   (fp32 accumulator)
//
// Determinism contract: byte-identity-by-construction across all n_rows in a
// single batch dispatch — pinned kernel uses one CTA per output cell, fixed
// compile-time K loop, no Split-K, no atomics. cuBLAS HGEMM is explicitly
// forbidden in this dispatch path (spec §6.1.A).

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void dflash_gemm_npc(
    const __half * weight,   // [N_cols, K]
    const __half * act,      // [n_rows, K]
    float        * dst_f32,  // [n_rows, N_cols]
    int            K,
    int            N_cols,
    int            n_rows,
    cudaStream_t   stream);

#ifdef __cplusplus
}
#endif
