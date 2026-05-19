// dflash-gemm.cu
//
// Thin forwarder from dflash_gemm_npc to ggml_cuda_mul_mat_f16_pinned.
// See dflash-gemm.cuh and specs/dflash/kernel-design.md §6.1.A.

#include "dflash-gemm.cuh"
#include "../mul-mat-f16-pinned.cuh"

extern "C" void dflash_gemm_npc(
    const __half * weight,
    const __half * act,
    float        * dst_f32,
    int            K,
    int            N_cols,
    int            n_rows,
    cudaStream_t   stream)
{
    ggml_cuda_mul_mat_f16_pinned(
        weight, act, dst_f32,
        K,
        /*N_rows=*/N_cols,
        /*M=*/n_rows,
        /*K_stride_w=*/K,
        /*K_stride_a=*/K,
        /*N_dst_stride=*/N_cols,
        stream);
}
