#pragma once

#include "common.cuh"

// Deterministic F16 weight × F16 activation → F32 dst matrix multiply.
//
// Each output cell dst[n, m] is computed by exactly ONE CTA via a fixed
// compile-time K loop with fp32 accumulators inside an HMMA m16n8k16
// fragment (Turing fallback: 2x m8n8k8). No split-K, no cross-CTA atomics,
// no shape-dependent algo selection — so dst[n, m] is byte-identical
// regardless of the total M.
//
// Compatible with the shape produced by `ggml_cuda_op_mul_mat_cublas` when
// it converts src0 to F16 (line 1815-1853 of ggml-cuda.cu) and src1 to F16
// (line 1856-1875). Replaces the cublasGemmEx call at line 1884.
//
// Layout (ggml convention):
//   src0 weight  [K, N]: data[k + n*K], "row-major in N×K" terms
//   src1 act     [K, M]: data[k + m*K], "column-major in K×M" terms
//   dst          [N, M]: data[n + m*N_dst_stride]
//
// Per-cell accumulation order = K loop iteration order, fixed at compile
// time. Same for any M. Byte-identity by construction.

// Tile dims correspond to one mma m16n8k16 fragment. Kernel computes
// TILE_N output rows × TILE_M output cols per CTA.
#define MUL_MAT_F16_PINNED_TILE_N 16
#define MUL_MAT_F16_PINNED_TILE_M 8
#define MUL_MAT_F16_PINNED_TILE_K 16
#define MUL_MAT_F16_PINNED_NWARPS 1   // one warp per CTA

// Host-side launcher. Computes dst[N, M] = weight[K, N]^T @ act[K, M]
// (ggml semantics). K MUST be a multiple of TILE_K = 16. N_rows and M
// are rounded up to TILE_N and TILE_M respectively; out-of-range stores
// are masked.
void ggml_cuda_mul_mat_f16_pinned(
        const half  * weight,    // K * N_rows halves, weight[k + n*K]
        const half  * act,       // K * M halves,      act   [k + m*K]
        float       * dst,       // N_rows * M floats, dst[n + m*N_dst_stride]
        int           K,
        int           N_rows,
        int           M,
        int           K_stride_w,    // = K for contiguous; passed to be safe
        int           K_stride_a,    // = K for contiguous
        int           N_dst_stride,  // dst columns: stride along m
        cudaStream_t  stream);
