#include "common.cuh"
#include "ggml-cuda.h"

// Device-side per-step state restore for delta-net / SSM checkpoint rollback.
//
// One block per layer. Each block reconstructs the layer's recurrent state in s_l[il]:
//
//   conv portion (offset 0, size conv_state_dim):
//     for each (col, d) in [0..d_conv-1) x [0..conv_dim):
//       src_token = step - (d_conv-2) + col
//       if src_token >= 0:
//         s_l[col + d * d_conv_m1] = qkv[d + src_token * conv_dim]
//       else:
//         old_col = (d_conv-1) + src_token
//         s_l[col + d * d_conv_m1] = shadow ? shadow[old_col + d * d_conv_m1] : 0
//
//   ssm portion (offset conv_state_dim, size ssm_state_dim):
//     copy from per_step_ssm[il] + step * ssm_state_dim
//
// Mirrors the host-side logic in llama_kv_cache::per_step_restore (src/llama.cpp).

template <int BLOCK_SIZE>
__global__ void per_step_restore_layer_kernel(
    float * const * __restrict__ dst_ptrs,
    const float * const * __restrict__ ssm_ptrs,
    const float * const * __restrict__ qkv_ptrs,
    const float * const * __restrict__ shadow_ptrs,
    int step,
    int conv_state_dim,
    int conv_dim,
    int d_conv,
    int ssm_state_dim) {

    const int il = blockIdx.x;
    const int tid = threadIdx.x;
    const int d_conv_m1 = d_conv - 1;

    float * const s_l = dst_ptrs[il];
    if (s_l == nullptr) return;

    const float * const ssm = ssm_ptrs[il];
    const float * const qkv = qkv_ptrs[il];
    const float * const shadow = shadow_ptrs[il];  // may be null

    // Phase 1: reconstruct conv state at s_l[0 .. conv_state_dim)
    // Layout: conv_buf[col + d * d_conv_m1]
    for (int idx = tid; idx < conv_state_dim; idx += BLOCK_SIZE) {
        const int col = idx % d_conv_m1;
        const int d   = idx / d_conv_m1;
        const int src_token = step - (d_conv_m1 - 1) + col;
        float val;
        if (src_token >= 0) {
            val = qkv[d + (size_t)src_token * conv_dim];
        } else {
            const int old_col = d_conv_m1 + src_token;
            if (old_col >= 0 && old_col < d_conv_m1 && shadow != nullptr) {
                val = shadow[old_col + (size_t)d * d_conv_m1];
            } else {
                val = 0.0f;
            }
        }
        s_l[idx] = val;
    }

    // Phase 2: copy SSM slice from per_step_ssm[il] + step * ssm_state_dim
    const float * ssm_src = ssm + (size_t)step * ssm_state_dim;
    float * ssm_dst = s_l + conv_state_dim;
    for (int idx = tid; idx < ssm_state_dim; idx += BLOCK_SIZE) {
        ssm_dst[idx] = ssm_src[idx];
    }
}

extern "C" GGML_CALL void ggml_backend_cuda_per_step_restore_layers(
    int n_layers,
    void * const * dst_ptrs,
    const void * const * ssm_ptrs,
    const void * const * qkv_ptrs,
    const void * const * shadow_ptrs,
    int step,
    int64_t conv_state_dim,
    int64_t conv_dim,
    int32_t d_conv,
    int64_t ssm_state_dim,
    int device) {

    if (n_layers <= 0 || conv_state_dim <= 0 || ssm_state_dim <= 0 || d_conv <= 1 || step < 0) return;
    if (conv_state_dim != (int64_t)(d_conv - 1) * conv_dim) return;
    GGML_ASSERT(conv_state_dim <= INT_MAX);
    GGML_ASSERT(ssm_state_dim  <= INT_MAX);
    GGML_ASSERT(conv_dim       <= INT_MAX);

    ggml_cuda_set_device(device);

    // Marshal the arrays of layer pointers to device. The per_step_restore call site
    // hands us host-side arrays of device pointers; the kernel needs them on-device
    // to dereference dst_ptrs[il] etc.
    cudaStream_t stream = nullptr;  // legacy default stream — synchronizes with everything
    const size_t ptr_array_bytes = (size_t)n_layers * sizeof(void *);

    void   ** d_dst    = nullptr;
    void const ** d_ssm    = nullptr;
    void const ** d_qkv    = nullptr;
    void const ** d_shadow = nullptr;
    CUDA_CHECK(cudaMallocAsync((void**)&d_dst,    ptr_array_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_ssm,    ptr_array_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_qkv,    ptr_array_bytes, stream));
    CUDA_CHECK(cudaMallocAsync((void**)&d_shadow, ptr_array_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_dst,    dst_ptrs,    ptr_array_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ssm,    ssm_ptrs,    ptr_array_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_qkv,    qkv_ptrs,    ptr_array_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_shadow, shadow_ptrs, ptr_array_bytes, cudaMemcpyHostToDevice, stream));

    constexpr int BLOCK_SIZE = 256;
    per_step_restore_layer_kernel<BLOCK_SIZE><<<n_layers, BLOCK_SIZE, 0, stream>>>(
        (float * const *)d_dst,
        (const float * const *)d_ssm,
        (const float * const *)d_qkv,
        (const float * const *)d_shadow,
        step,
        (int)conv_state_dim,
        (int)conv_dim,
        d_conv,
        (int)ssm_state_dim);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFreeAsync(d_dst, stream));
    CUDA_CHECK(cudaFreeAsync(d_ssm, stream));
    CUDA_CHECK(cudaFreeAsync(d_qkv, stream));
    CUDA_CHECK(cudaFreeAsync(d_shadow, stream));
}
