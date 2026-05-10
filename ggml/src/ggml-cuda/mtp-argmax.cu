#include "common.cuh"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// Optional top-2 variant: identical pass-2 (sum_exp) but pass 1 carries a
// (best, second) pair through warp/block reduction via packed (val, id) in
// __ulonglong (sign-flipped float bits in hi 32 + id in lo 32 → unsigned
// ordering matches signed-float ordering). Halves shfl ops vs naive (val,id)
// pairs separately. Default API path (top-1 only) is unchanged from HEAD —
// this kernel is only invoked when the caller supplies dst_top2_ids_dev.
__device__ __forceinline__ unsigned long long _mtp_pack_val_id(float v, int id) {
    unsigned int u = __float_as_uint(v);
    u ^= (int(u) >> 31) | 0x80000000u;
    return ((unsigned long long)u << 32) | (unsigned int)id;
}
__device__ __forceinline__ int _mtp_unpack_id(unsigned long long p) {
    return (int)(unsigned int)(p & 0xffffffffu);
}

template <int BLOCK_SIZE>
__global__ void mtp_argmax_with_prob_top2_f32_kernel(
    const float * __restrict__ logits,
    int32_t *     __restrict__ dst_ids,
    float   *     __restrict__ dst_probs,
    int32_t *     __restrict__ dst_top2_ids,
    const int n_vocab) {

    const int row = blockIdx.x;
    const float * row_logits = logits + (size_t)row * n_vocab;

    constexpr int n_warps = BLOCK_SIZE / WARP_SIZE;
    const int tid  = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;

    // ---------- Pass 1: top-2 packed argmax ----------
    unsigned long long t_best = _mtp_pack_val_id(-FLT_MAX, 0);
    unsigned long long t_scnd = _mtp_pack_val_id(-FLT_MAX, 0);
    for (int i = tid; i < n_vocab; i += BLOCK_SIZE) {
        const unsigned long long p = _mtp_pack_val_id(row_logits[i], i);
        if (p > t_best) { t_scnd = t_best; t_best = p; }
        else if (p > t_scnd) { t_scnd = p; }
    }

    auto merge_top2 = [&](unsigned long long ob, unsigned long long os) {
        const unsigned long long pairs[4] = { t_best, t_scnd, ob, os };
        int b_idx = 0;
        #pragma unroll
        for (int k = 1; k < 4; k++) if (pairs[k] > pairs[b_idx]) b_idx = k;
        int s_idx = (b_idx == 0) ? 1 : 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) if (k != b_idx && pairs[k] > pairs[s_idx]) s_idx = k;
        t_best = pairs[b_idx];
        t_scnd = pairs[s_idx];
    };

    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        const unsigned long long ob = __shfl_xor_sync(0xffffffff, t_best, off, WARP_SIZE);
        const unsigned long long os = __shfl_xor_sync(0xffffffff, t_scnd, off, WARP_SIZE);
        merge_top2(ob, os);
    }

    __shared__ unsigned long long s_warp_best[n_warps];
    __shared__ unsigned long long s_warp_scnd[n_warps];
    if (lane == 0) {
        s_warp_best[warp] = t_best;
        s_warp_scnd[warp] = t_scnd;
    }
    __syncthreads();

    if (warp == 0) {
        if (lane < n_warps) { t_best = s_warp_best[lane]; t_scnd = s_warp_scnd[lane]; }
        else { t_best = _mtp_pack_val_id(-FLT_MAX, 0); t_scnd = _mtp_pack_val_id(-FLT_MAX, 0); }
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            const unsigned long long ob = __shfl_xor_sync(0xffffffff, t_best, off, WARP_SIZE);
            const unsigned long long os = __shfl_xor_sync(0xffffffff, t_scnd, off, WARP_SIZE);
            merge_top2(ob, os);
        }
        if (lane == 0) {
            s_warp_best[0] = t_best;
            s_warp_scnd[0] = t_scnd;
        }
    }
    __syncthreads();

    // unpack maxval for pass 2
    unsigned int u_best = (unsigned int)(s_warp_best[0] >> 32);
    u_best ^= (int(u_best) >> 31) | 0x80000000u;
    const float maxval = __uint_as_float(u_best);
    const int   argmax     = _mtp_unpack_id(s_warp_best[0]);
    const int   argmax_top2 = _mtp_unpack_id(s_warp_scnd[0]);

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
    if (lane == 0) s_warp_sum[warp] = t_sum;
    __syncthreads();

    if (warp == 0) {
        t_sum = (lane < n_warps) ? s_warp_sum[lane] : 0.0f;
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            t_sum += __shfl_xor_sync(0xffffffff, t_sum, off, WARP_SIZE);
        }
        if (lane == 0) {
            dst_ids[row]      = argmax;
            dst_probs[row]    = 1.0f / t_sum;
            dst_top2_ids[row] = argmax_top2;
        }
    }
}

extern "C" GGML_CALL void ggml_backend_cuda_mtp_argmax_with_prob(
    const void * logits_dev,    // float*, [n_rows, n_vocab]
    int n_rows,
    int n_vocab,
    void * dst_ids_dev,         // int32_t*, [n_rows]
    void * dst_probs_dev,       // float*,   [n_rows]
    void * dst_top2_ids_dev,    // int32_t*, [n_rows] OR nullptr to skip top-2
    int device) {

    if (n_rows <= 0 || n_vocab <= 0) return;

    ggml_cuda_set_device(device);

    constexpr int BLOCK_SIZE = 256;  // n_warps = 8 on sm_75
    cudaStream_t stream = nullptr;   // legacy default stream — synchronizes with prior async work

    if (dst_top2_ids_dev == nullptr) {
        // Default fast path: kernel and behaviour identical to HEAD.
        mtp_argmax_with_prob_f32_kernel<BLOCK_SIZE><<<n_rows, BLOCK_SIZE, 0, stream>>>(
            (const float *)logits_dev,
            (int32_t *)dst_ids_dev,
            (float *)dst_probs_dev,
            n_vocab);
    } else {
        // Probe / tree-K=2 path: top-2 reduction adds ~1.2% kernel cost.
        mtp_argmax_with_prob_top2_f32_kernel<BLOCK_SIZE><<<n_rows, BLOCK_SIZE, 0, stream>>>(
            (const float *)logits_dev,
            (int32_t *)dst_ids_dev,
            (float *)dst_probs_dev,
            (int32_t *)dst_top2_ids_dev,
            n_vocab);
    }
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
    int32_t * host_top2_ids_out,  // optional; nullptr to skip top-2 (default fast path)
    int device) {

    if (n_rows <= 0 || n_vocab <= 0) return;

    ggml_cuda_set_device(device);

    // C.1 diagnostic: env-gated per-row INPUT-LOGITS hash dump. Hashes are
    // computed by D2H copying each row of logits_dev into a host scratch
    // buffer, then summing element-wise XOR. If row hashes differ across
    // slots in the same call, the bug is upstream of this kernel. Off by
    // default; significant cost when on (n_rows D2H copies per call).
    static const bool _diag_logits_hash = []() {
        const char * v = getenv("LLAMA_DIAG_MTP_LOGITS_HASH");
        return v && *v && *v != '0';
    }();
    if (_diag_logits_hash && n_rows > 1) {
        std::vector<float> tmp((size_t) n_vocab);
        for (int r = 0; r < n_rows; ++r) {
            cudaMemcpy(tmp.data(), (const char *)logits_dev + (size_t) r * n_vocab * sizeof(float),
                       n_vocab * sizeof(float), cudaMemcpyDeviceToHost);
            uint64_t h = 0;
            double l2 = 0.0;
            for (int j = 0; j < n_vocab; ++j) {
                h ^= (uint64_t) __builtin_bswap32(*reinterpret_cast<uint32_t*>(&tmp[j])) * (uint64_t) (j+1);
                l2 += (double) tmp[j] * tmp[j];
            }
            fprintf(stderr, "[c1.diag.mtp_argmax] r%d/%d n_vocab=%d hash=0x%016llx L2=%.6f\n",
                    r, n_rows, n_vocab, (unsigned long long) h, sqrt(l2));
        }
    }

    static int32_t * cached_ids_dev[GGML_CUDA_MAX_DEVICES]      = {0};
    static float   * cached_probs_dev[GGML_CUDA_MAX_DEVICES]    = {0};
    static int32_t * cached_top2_ids_dev[GGML_CUDA_MAX_DEVICES] = {0};
    static int       cached_capacity[GGML_CUDA_MAX_DEVICES]     = {0};

    if (n_rows > cached_capacity[device]) {
        if (cached_ids_dev[device])      cudaFree(cached_ids_dev[device]);
        if (cached_probs_dev[device])    cudaFree(cached_probs_dev[device]);
        if (cached_top2_ids_dev[device]) cudaFree(cached_top2_ids_dev[device]);
        CUDA_CHECK(cudaMalloc((void**)&cached_ids_dev[device],      n_rows * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc((void**)&cached_probs_dev[device],    n_rows * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&cached_top2_ids_dev[device], n_rows * sizeof(int32_t)));
        cached_capacity[device] = n_rows;
    }

    constexpr int BLOCK_SIZE = 256;
    cudaStream_t stream = nullptr;

    if (host_top2_ids_out == nullptr) {
        // Default fast path: HEAD-equivalent kernel.
        mtp_argmax_with_prob_f32_kernel<BLOCK_SIZE><<<n_rows, BLOCK_SIZE, 0, stream>>>(
            (const float *)logits_dev,
            cached_ids_dev[device],
            cached_probs_dev[device],
            n_vocab);
    } else {
        // Probe path: writes top-2 alongside top-1.
        mtp_argmax_with_prob_top2_f32_kernel<BLOCK_SIZE><<<n_rows, BLOCK_SIZE, 0, stream>>>(
            (const float *)logits_dev,
            cached_ids_dev[device],
            cached_probs_dev[device],
            cached_top2_ids_dev[device],
            n_vocab);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(host_ids_out,   cached_ids_dev[device],   n_rows * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_probs_out, cached_probs_dev[device], n_rows * sizeof(float),   cudaMemcpyDeviceToHost));
    if (host_top2_ids_out != nullptr) {
        CUDA_CHECK(cudaMemcpy(host_top2_ids_out, cached_top2_ids_dev[device], n_rows * sizeof(int32_t), cudaMemcpyDeviceToHost));
    }
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
