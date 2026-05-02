#include "common.cuh"
#include "ggml-cuda.h"

// Per-row argmax + softmax-prob over [n_rows, n_vocab] logits, fully on device.
//
// Replaces the host-side argmax + softmax-denom computation in
// common_sampler_sample_speculative for the MTP DRAFT_GEN path. With Qwen 3.5
// 0.8B's vocab=248,320, the MTP DRAFT_GEN forward emits 2 rows of logits
// (n_outputs=2), totalling 1.99 MB per draft event of D2H if pulled to host.
// This kernel collapses that to (n_rows × 8 bytes) of (id, prob) results.
//
// One block per row. The block does two passes:
//   pass 1: cooperative argmax (max value + index)
//   pass 2: cooperative sum_exp(logits[i] - maxval), then prob = 1/sum_exp.

template <int BLOCK_SIZE>
__global__ void mtp_argmax_with_prob_f32_kernel(
    const float * __restrict__ logits,
    int32_t *     __restrict__ dst_ids,
    float   *     __restrict__ dst_probs,
    const int n_vocab) {

    const int row = blockIdx.x;
    const float * row_logits = logits + (size_t)row * n_vocab;

    constexpr int n_warps = BLOCK_SIZE / WARP_SIZE;
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;

    // ---------- Pass 1: argmax ----------
    float t_max = -FLT_MAX;
    int   t_arg = 0;
    for (int i = tid; i < n_vocab; i += BLOCK_SIZE) {
        const float v = row_logits[i];
        if (v > t_max) { t_max = v; t_arg = i; }
    }

    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, t_max, off, WARP_SIZE);
        const int   oa = __shfl_xor_sync(0xffffffff, t_arg, off, WARP_SIZE);
        if (ov > t_max) { t_max = ov; t_arg = oa; }
    }

    __shared__ float s_warp_max[n_warps];
    __shared__ int   s_warp_arg[n_warps];
    if (lane == 0) {
        s_warp_max[warp] = t_max;
        s_warp_arg[warp] = t_arg;
    }
    __syncthreads();

    if (warp == 0) {
        if (lane < n_warps) {
            t_max = s_warp_max[lane];
            t_arg = s_warp_arg[lane];
        } else {
            t_max = -FLT_MAX;
            t_arg = 0;
        }
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffff, t_max, off, WARP_SIZE);
            const int   oa = __shfl_xor_sync(0xffffffff, t_arg, off, WARP_SIZE);
            if (ov > t_max) { t_max = ov; t_arg = oa; }
        }
        if (lane == 0) {
            s_warp_max[0] = t_max;
            s_warp_arg[0] = t_arg;
        }
    }
    __syncthreads();

    const float maxval = s_warp_max[0];
    const int   argmax = s_warp_arg[0];

    // ---------- Pass 2: sum_exp(logits[i] - maxval) ----------
    float t_sum = 0.0f;
    for (int i = tid; i < n_vocab; i += BLOCK_SIZE) {
        t_sum += expf(row_logits[i] - maxval);
    }

    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        t_sum += __shfl_xor_sync(0xffffffff, t_sum, off, WARP_SIZE);
    }

    __shared__ float s_warp_sum[n_warps];
    if (lane == 0) {
        s_warp_sum[warp] = t_sum;
    }
    __syncthreads();

    if (warp == 0) {
        t_sum = (lane < n_warps) ? s_warp_sum[lane] : 0.0f;
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            t_sum += __shfl_xor_sync(0xffffffff, t_sum, off, WARP_SIZE);
        }
        if (lane == 0) {
            dst_ids[row]   = argmax;
            dst_probs[row] = 1.0f / t_sum;
        }
    }
}

extern "C" GGML_CALL void ggml_backend_cuda_mtp_argmax_with_prob(
    const void * logits_dev,    // float*, [n_rows, n_vocab]
    int n_rows,
    int n_vocab,
    void * dst_ids_dev,         // int32_t*, [n_rows]
    void * dst_probs_dev,       // float*,   [n_rows]
    int device) {

    if (n_rows <= 0 || n_vocab <= 0) return;

    ggml_cuda_set_device(device);

    constexpr int BLOCK_SIZE = 256;  // n_warps = 8 on sm_75
    cudaStream_t stream = nullptr;   // legacy default stream — synchronizes with prior async work

    mtp_argmax_with_prob_f32_kernel<BLOCK_SIZE><<<n_rows, BLOCK_SIZE, 0, stream>>>(
        (const float *)logits_dev,
        (int32_t *)dst_ids_dev,
        (float *)dst_probs_dev,
        n_vocab);
    CUDA_CHECK(cudaGetLastError());
}

// Single-call helper that handles device scratch allocation, kernel launch, and
// D2H of the (id, prob) pair per row. Caller passes host buffers; this function
// synchronises before returning so the host buffers are valid.
//
// Per-device scratch is cached and grown on demand to avoid alloc/free per call.
extern "C" GGML_CALL void ggml_backend_cuda_mtp_argmax_with_prob_to_host(
    const void * logits_dev,
    int n_rows,
    int n_vocab,
    int32_t * host_ids_out,
    float   * host_probs_out,
    int device) {

    if (n_rows <= 0 || n_vocab <= 0) return;

    ggml_cuda_set_device(device);

    static int32_t * cached_ids_dev[GGML_CUDA_MAX_DEVICES]   = {0};
    static float   * cached_probs_dev[GGML_CUDA_MAX_DEVICES] = {0};
    static int       cached_capacity[GGML_CUDA_MAX_DEVICES]  = {0};

    if (n_rows > cached_capacity[device]) {
        if (cached_ids_dev[device])   cudaFree(cached_ids_dev[device]);
        if (cached_probs_dev[device]) cudaFree(cached_probs_dev[device]);
        CUDA_CHECK(cudaMalloc((void**)&cached_ids_dev[device],   n_rows * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc((void**)&cached_probs_dev[device], n_rows * sizeof(float)));
        cached_capacity[device] = n_rows;
    }

    constexpr int BLOCK_SIZE = 256;
    cudaStream_t stream = nullptr;

    mtp_argmax_with_prob_f32_kernel<BLOCK_SIZE><<<n_rows, BLOCK_SIZE, 0, stream>>>(
        (const float *)logits_dev,
        cached_ids_dev[device],
        cached_probs_dev[device],
        n_vocab);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(host_ids_out,   cached_ids_dev[device],   n_rows * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_probs_out, cached_probs_dev[device], n_rows * sizeof(float),   cudaMemcpyDeviceToHost));
}

extern "C" GGML_CALL void ggml_backend_cuda_memcpy_d2d(
    void * dst_dev,
    const void * src_dev,
    size_t nbytes,
    int device) {

    if (nbytes == 0 || dst_dev == nullptr || src_dev == nullptr) return;
    ggml_cuda_set_device(device);
    CUDA_CHECK(cudaMemcpyAsync(dst_dev, src_dev, nbytes, cudaMemcpyDeviceToDevice, /*stream=*/nullptr));
}

extern "C" GGML_CALL void * ggml_backend_cuda_malloc(size_t nbytes, int device) {
    if (nbytes == 0) return nullptr;
    ggml_cuda_set_device(device);
    void * p = nullptr;
    cudaError_t err = cudaMalloc(&p, nbytes);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return nullptr;
    }
    return p;
}

extern "C" GGML_CALL void ggml_backend_cuda_free(void * ptr) {
    if (ptr == nullptr) return;
    cudaFree(ptr);
}
