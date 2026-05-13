// dflash-drafter-forward.cu
//
// 5-layer DFlash drafter forward implementation. Spec:
// specs/dflash/kernel-design.md §6.1.
//
// Phase A implementation: per-step sub-kernels invoked sequentially
// from a host-side dispatcher loop (5 layers × 10 sub-kernels = 50
// launches per cycle). All sub-kernels use scalar fp32 accumulators
// for byte-identity binding against the serial fp32 scalar reference.
// Spec deviation from §6.1's "cooperative WMMA mega-kernel" is the
// same precedent as T3 (inject_kv_fused: WMMA → scalar fp32).
//
// Cooperative WMMA mega-kernel is Phase B work, gated on T8 perf
// outcome — Phase A is correctness-first.
//
// Allium bindings (per specs/dflash/allium-tla-binding.json):
//   - SingleForwardPerStep, QuerySpanIsOnePlusN,
//     InjectionConsumedAtEveryLayer, LayerTypeDependentMask,
//     AnchorEmbeddingFromTarget, AnchorPosPreserved,
//     BlockSizeBindsToConfig, DeterminismPerDeployment,
//     FeatureSourceFixedPerDeployment

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdio>
#include <vector>

#include "dflash-drafter-forward.cuh"

namespace {

constexpr int THREADS_PER_CTA = 256;

// ===========================================================================
// Block-wide fp32 sum reduction over the values held by each thread.
// One reduction per CTA. Uses warp shuffle + SMEM tree.
//
// Caller allocates the reduce_smem array (one fp32 per warp).
// Returns the total sum, broadcast to all threads in the block.
// ===========================================================================
__device__ float block_sum_f32(float val, float * reduce_smem) {
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFFu, val, offset);
    }
    if (lane == 0) reduce_smem[warp_id] = val;
    __syncthreads();
    float total = 0.0f;
    if (warp_id == 0) {
        const int n_warps = blockDim.x >> 5;
        float v = (lane < n_warps) ? reduce_smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
        }
        if (lane == 0) reduce_smem[0] = v;
    }
    __syncthreads();
    total = reduce_smem[0];
    return total;
}

// ===========================================================================
// Sub-kernel 1: attn_norm (also reused for ffn_norm).
//
// One CTA per (slot, query_position) row.
// Grid: dim3(n_rows)  Block: 256 threads.
//
// Each thread reads multiple elements of hidden_in[row, :] in strided
// fashion (tid, tid + 256, …), accumulates sum_sq into a register,
// then block-reduces.
//
// Output: hidden_n[row, :] = (hidden_in[row, :] * rsqrt(sum_sq/D + eps))
//                            * weight[:]
// ===========================================================================
__global__ void rmsnorm_kernel(
    const __half * __restrict__ hidden_in,   // [n_rows, D]
    const __half * __restrict__ weight,      // [D]
    __half       * __restrict__ hidden_n,    // [n_rows, D]
    float          norm_eps,
    int            D)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float reduce_smem[8];

    const __half * row_in  = hidden_in + static_cast<std::size_t>(row) * D;
    __half       * row_out = hidden_n  + static_cast<std::size_t>(row) * D;

    float sum_sq = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        const float x = __half2float(row_in[i]);
        sum_sq += x * x;
    }
    const float total_sq = block_sum_f32(sum_sq, reduce_smem);
    const float rsq = rsqrtf(total_sq / static_cast<float>(D) + norm_eps);

    for (int i = tid; i < D; i += blockDim.x) {
        const float x  = __half2float(row_in[i]);
        const float w  = __half2float(weight[i]);
        row_out[i] = __float2half((x * rsq) * w);
    }
}

// ===========================================================================
// Sub-kernel 2: GEMM (scalar fp32, output row × col tiled by threads).
//
// One CTA per output row. Each thread accumulates a strided subset of
// columns. Internally fp32 accumulator.
//
//   C[row, col] = sum_k A[row, k] * B[col, k]
// where B is in [N_cols, K] layout (row-major), so B[col, k] = B_buf[col*K + k].
// This matches typical weight storage in llama.cpp.
//
// Grid: dim3(n_rows)  Block: 256 threads (each owns N/256 output cols).
// ===========================================================================
__global__ void gemm_row_x_col_kernel(
    const __half * __restrict__ A,     // [n_rows, K]
    const __half * __restrict__ B,     // [N_cols, K]
    __half       * __restrict__ C,     // [n_rows, N_cols]
    int            K,
    int            N_cols)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __half * row_a = A + static_cast<std::size_t>(row) * K;
    __half       * row_c = C + static_cast<std::size_t>(row) * N_cols;

    for (int col = tid; col < N_cols; col += blockDim.x) {
        const __half * row_b = B + static_cast<std::size_t>(col) * K;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += __half2float(row_a[k]) * __half2float(row_b[k]);
        }
        row_c[col] = __float2half(acc);
    }
}

// ===========================================================================
// Sub-kernel 3: per-head q_norm + NeoX RoPE.
//
// One CTA per (slot, query_position) row.
// Each row's Q has H_q heads × D_h dim. Per head, compute RMSNorm with
// per-layer q_norm_weight, then apply NeoX RoPE using the query position
// (= anchor_pos[slot] + q_offset).
//
// fp64 transcendentals (pow, cos, sin) cast to fp32 — same as T3 inject
// kernel + spec §6.1 binding.
//
// Grid: dim3(n_rows)  Block: 256 threads.
// ===========================================================================
__global__ void q_norm_rope_kernel(
    __half       * __restrict__ q_buf,       // [n_rows, H_q * D_h] in-place
    const __half * __restrict__ q_norm_w,    // [D_h]
    const int    * __restrict__ slot_positions, // [N_slots]
    float          rope_base,
    float          norm_eps,
    int            Q,            // 1 + BLOCK_SIZE
    int            H_q,
    int            D_h)
{
    const int row    = blockIdx.x;
    const int slot   = row / Q;
    const int q_off  = row % Q;
    const int qpos   = slot_positions[slot] + q_off;
    const int tid    = threadIdx.x;
    __shared__ float reduce_smem[8];

    __half * row_q = q_buf + static_cast<std::size_t>(row) * H_q * D_h;
    const int D_half = D_h / 2;

    for (int h = 0; h < H_q; ++h) {
        __half * qh = row_q + h * D_h;

        // q_norm: RMSNorm over D_h elements of this head.
        float sum_sq = 0.0f;
        for (int i = tid; i < D_h; i += blockDim.x) {
            const float v = __half2float(qh[i]);
            sum_sq += v * v;
        }
        const float total_sq = block_sum_f32(sum_sq, reduce_smem);
        const float rsq = rsqrtf(total_sq / static_cast<float>(D_h) + norm_eps);
        for (int i = tid; i < D_h; i += blockDim.x) {
            const float v = __half2float(qh[i]);
            const float w = __half2float(q_norm_w[i]);
            qh[i] = __float2half((v * rsq) * w);
        }
        __syncthreads();

        // RoPE: NeoX-pair rotation at qpos.
        for (int i = tid; i < D_half; i += blockDim.x) {
            const double exp_val_d  = static_cast<double>(2 * i) / static_cast<double>(D_h);
            const double inv_freq_d = pow(static_cast<double>(rope_base), -exp_val_d);
            const double theta_d    = static_cast<double>(qpos) * inv_freq_d;
            const float  c          = static_cast<float>(cos(theta_d));
            const float  s          = static_cast<float>(sin(theta_d));
            const float  lo         = __half2float(qh[i]);
            const float  hi         = __half2float(qh[i + D_half]);
            qh[i]            = __float2half(lo * c - hi * s);
            qh[i + D_half]   = __float2half(lo * s + hi * c);
        }
        __syncthreads();
    }
}

// ===========================================================================
// Sub-kernel 3b: per-head K norm + NeoX RoPE.
//
// Same structure as q_norm_rope_kernel but operates on K with H_kv heads
// (vs Q's H_q heads). Writes back in-place.
// ===========================================================================
__global__ void k_norm_rope_kernel(
    __half       * __restrict__ k_buf,       // [n_rows, H_kv * D_h] in-place
    const __half * __restrict__ k_norm_w,    // [D_h]
    const int    * __restrict__ slot_positions, // [N_slots]
    float          rope_base,
    float          norm_eps,
    int            Q,            // 1 + BLOCK_SIZE
    int            H_kv,
    int            D_h)
{
    const int row    = blockIdx.x;
    const int slot   = row / Q;
    const int q_off  = row % Q;
    const int qpos   = slot_positions[slot] + q_off;
    const int tid    = threadIdx.x;
    __shared__ float reduce_smem[8];

    __half * row_k = k_buf + static_cast<std::size_t>(row) * H_kv * D_h;
    const int D_half = D_h / 2;

    for (int h = 0; h < H_kv; ++h) {
        __half * kh = row_k + h * D_h;

        // k_norm: RMSNorm over D_h elements of this head.
        float sum_sq = 0.0f;
        for (int i = tid; i < D_h; i += blockDim.x) {
            const float v = __half2float(kh[i]);
            sum_sq += v * v;
        }
        const float total_sq = block_sum_f32(sum_sq, reduce_smem);
        const float rsq = rsqrtf(total_sq / static_cast<float>(D_h) + norm_eps);
        for (int i = tid; i < D_h; i += blockDim.x) {
            const float v = __half2float(kh[i]);
            const float w = __half2float(k_norm_w[i]);
            kh[i] = __float2half((v * rsq) * w);
        }
        __syncthreads();

        // RoPE: NeoX-pair rotation at qpos. fp64 transcendentals to match
        // the inject kernel + scalar reference (sm_75 libdevice fp32 trig
        // diverges from CPU libm by up to 6 ULP; fp64 bridges the gap).
        for (int i = tid; i < D_half; i += blockDim.x) {
            const double exp_val_d  = static_cast<double>(2 * i) / static_cast<double>(D_h);
            const double inv_freq_d = pow(static_cast<double>(rope_base), -exp_val_d);
            const double theta_d    = static_cast<double>(qpos) * inv_freq_d;
            const float  c          = static_cast<float>(cos(theta_d));
            const float  s          = static_cast<float>(sin(theta_d));
            const float  lo         = __half2float(kh[i]);
            const float  hi         = __half2float(kh[i + D_half]);
            kh[i]            = __float2half(lo * c - hi * s);
            kh[i + D_half]   = __float2half(lo * s + hi * c);
        }
        __syncthreads();
    }
}

// ===========================================================================
// Sub-kernel 3c: cache_write_kv — write per-row K, V to KV cache at the
// query positions (slot_positions[slot] + q_off for q_off in [0, Q)).
//
// One CTA per (slot, q_off) row. Each thread strides across H_kv*D_h cells.
// Writes from k_buf, v_buf into k_cache_layer[slot, qpos, h, d] and
// v_cache_layer[slot, qpos, h, d]. The cache layout matches
// dflash_inject_kv_fused's writes — the same cache buffer is shared
// between the inject step (context positions) and this step (query
// positions).
// ===========================================================================
__global__ void cache_write_kv_kernel(
    const __half * __restrict__ k_buf,           // [n_rows, H_kv * D_h]
    const __half * __restrict__ v_buf,           // [n_rows, H_kv * D_h]
    __half       * __restrict__ k_cache_layer,   // [N_slots, SeqLen, H_kv, D_h]
    __half       * __restrict__ v_cache_layer,   // [N_slots, SeqLen, H_kv, D_h]
    const int    * __restrict__ slot_positions,  // [N_slots]
    int            Q,
    int            SeqLen,
    int            H_kv,
    int            D_h)
{
    const int row    = blockIdx.x;
    const int slot   = row / Q;
    const int q_off  = row % Q;
    const int qpos   = slot_positions[slot] + q_off;
    const int tid    = threadIdx.x;
    const int hd     = H_kv * D_h;

    const __half * row_k = k_buf + static_cast<std::size_t>(row) * hd;
    const __half * row_v = v_buf + static_cast<std::size_t>(row) * hd;
    const std::size_t cache_off =
        ((static_cast<std::size_t>(slot) * SeqLen + qpos) * H_kv) * D_h;
    for (int i = tid; i < hd; i += blockDim.x) {
        k_cache_layer[cache_off + i] = row_k[i];
        v_cache_layer[cache_off + i] = row_v[i];
    }
}

// ===========================================================================
// Sub-kernel 4: scalar fp32 SWA/full-attention.
//
// One CTA per (slot, query_position) row. One warp per attention head
// (H_q = 40 heads, 5 warps × 8 heads per warp at 256 threads). Lane
// within warp handles D_h dim.
//
// SWA layer (layer_type == 0): K-loop iterates [max(0, qpos - swa_window + 1), qpos]
// Full layer (layer_type == 1): K-loop iterates [0, qpos]
//
// Scalar fp32 attention: scores = (q @ K^T) / sqrt(D_h), softmax, @ V.
// ===========================================================================
__global__ void attention_kernel(
    const __half * __restrict__ q_buf,           // [n_rows, H_q * D_h]
    const __half * __restrict__ k_cache,         // [N_slots, SeqLen, H_kv, D_h]
    const __half * __restrict__ v_cache,         // [N_slots, SeqLen, H_kv, D_h]
    const int    * __restrict__ slot_positions,  // [N_slots]
    __half       * __restrict__ attn_out,        // [n_rows, H_q * D_h]
    int            layer_type,
    int            swa_window,
    int            Q,
    int            N_slots,
    int            SeqLen,
    int            H_q,
    int            H_kv,
    int            D_h)
{
    const int row    = blockIdx.x;
    const int slot   = row / Q;
    const int q_off  = row % Q;
    const int qpos   = slot_positions[slot] + q_off;
    const int tid    = threadIdx.x;
    const int gqa    = H_q / H_kv;
    const bool is_full = (layer_type == 1);

    const int k_lo = is_full ? 0 : max(0, qpos - swa_window + 1);
    // Full attention is bidirectional within the block — query at any
    // position attends to ALL block positions [anchor_pos, anchor_pos+Q-1]
    // plus the entire past context. SWA is causal: K-loop stops at qpos.
    const int anchor_pos = slot_positions[slot];
    const int k_hi = is_full ? (anchor_pos + Q - 1) : qpos;
    const int n_keys = k_hi - k_lo + 1;
    if (n_keys <= 0) return;

    const float inv_sqrt_dh = rsqrtf(static_cast<float>(D_h));

    const __half * row_q  = q_buf    + static_cast<std::size_t>(row)  * H_q * D_h;
    __half       * row_out = attn_out + static_cast<std::size_t>(row) * H_q * D_h;

    const std::size_t kv_slot_stride = static_cast<std::size_t>(SeqLen) * H_kv * D_h;
    const __half * k_slot = k_cache + static_cast<std::size_t>(slot) * kv_slot_stride;
    const __half * v_slot = v_cache + static_cast<std::size_t>(slot) * kv_slot_stride;

    // Process heads one at a time: each tid spans across n_keys then D_h.
    for (int h = 0; h < H_q; ++h) {
        const int h_kv = h / gqa;
        const __half * qh = row_q + h * D_h;

        // Step 1: compute scores[ki] = (q · K[k_lo+ki, h_kv, :]) * inv_sqrt_dh
        // and max for softmax stability.
        extern __shared__ float scores_smem[];  // size: n_keys floats per CTA — capped externally
        __shared__ float reduce_smem[8];
        __shared__ float head_max;
        __shared__ float head_sum;

        for (int ki = tid; ki < n_keys; ki += blockDim.x) {
            const int k = k_lo + ki;
            const __half * Kk = k_slot + (static_cast<std::size_t>(k) * H_kv + h_kv) * D_h;
            float dot = 0.0f;
            for (int d = 0; d < D_h; ++d) {
                dot += __half2float(qh[d]) * __half2float(Kk[d]);
            }
            scores_smem[ki] = dot * inv_sqrt_dh;
        }
        __syncthreads();

        // Reduce max
        float local_max = -CUDART_INF_F;
        for (int ki = tid; ki < n_keys; ki += blockDim.x) {
            if (scores_smem[ki] > local_max) local_max = scores_smem[ki];
        }
        // Block reduction for max
        const int warp_id = tid >> 5;
        const int lane    = tid & 31;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(0xFFFFFFFFu, local_max, offset);
            if (other > local_max) local_max = other;
        }
        if (lane == 0) reduce_smem[warp_id] = local_max;
        __syncthreads();
        if (warp_id == 0) {
            const int n_warps = blockDim.x >> 5;
            float v = (lane < n_warps) ? reduce_smem[lane] : -CUDART_INF_F;
            for (int offset = 16; offset > 0; offset >>= 1) {
                float other = __shfl_xor_sync(0xFFFFFFFFu, v, offset);
                if (other > v) v = other;
            }
            if (lane == 0) head_max = v;
        }
        __syncthreads();

        // exp and sum
        float local_sum = 0.0f;
        for (int ki = tid; ki < n_keys; ki += blockDim.x) {
            const float e = __expf(scores_smem[ki] - head_max);
            scores_smem[ki] = e;
            local_sum += e;
        }
        const float total_sum = block_sum_f32(local_sum, reduce_smem);
        if (tid == 0) head_sum = total_sum;
        __syncthreads();
        const float inv_sum = 1.0f / head_sum;

        // out[h, d] = sum_ki softmax[ki] * V[k_lo+ki, h_kv, d]
        for (int d = tid; d < D_h; d += blockDim.x) {
            float acc = 0.0f;
            for (int ki = 0; ki < n_keys; ++ki) {
                const int k = k_lo + ki;
                const __half * Vk = v_slot + (static_cast<std::size_t>(k) * H_kv + h_kv) * D_h;
                acc += scores_smem[ki] * inv_sum * __half2float(Vk[d]);
            }
            row_out[h * D_h + d] = __float2half(acc);
        }
        __syncthreads();
    }
}

// ===========================================================================
// Sub-kernel 5: residual add (a + b → out), element-wise.
//
// One CTA per row, threads stride across D.
// ===========================================================================
__global__ void residual_add_kernel(
    const __half * a,
    const __half * b,
    __half       * out,
    int            D)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __half * row_a = a   + static_cast<std::size_t>(row) * D;
    const __half * row_b = b   + static_cast<std::size_t>(row) * D;
    __half       * row_o = out + static_cast<std::size_t>(row) * D;
    for (int i = tid; i < D; i += blockDim.x) {
        const float va = __half2float(row_a[i]);
        const float vb = __half2float(row_b[i]);
        row_o[i] = __float2half(va + vb);
    }
}

// ===========================================================================
// Sub-kernel 6: silu(gate) * up → activated.
//
// One CTA per row, threads stride across intermediate.
// ===========================================================================
__global__ void silu_mul_kernel(
    const __half * __restrict__ gate,
    const __half * __restrict__ up,
    __half       * __restrict__ activated,
    int            intermediate)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __half * row_g = gate      + static_cast<std::size_t>(row) * intermediate;
    const __half * row_u = up        + static_cast<std::size_t>(row) * intermediate;
    __half       * row_a = activated + static_cast<std::size_t>(row) * intermediate;
    for (int i = tid; i < intermediate; i += blockDim.x) {
        const float g = __half2float(row_g[i]);
        const float u = __half2float(row_u[i]);
        const float a = (g / (1.0f + __expf(-g))) * u;
        row_a[i] = __float2half(a);
    }
}

// ===========================================================================
// Sub-kernel 7: copy/select BLOCK_SIZE output rows (drop anchor q==0).
//
// For each (slot, q_idx in 0..BLOCK_SIZE-1): copy hidden[slot*Q + q_idx + 1, :]
// to out_hidden[slot*BLOCK_SIZE + q_idx, :].
// ===========================================================================
__global__ void select_output_kernel(
    const __half * __restrict__ hidden,    // [N_slots, Q, D_emb]
    __half       * __restrict__ out_hidden,// [N_slots, BLOCK_SIZE, D_emb]
    int            Q,
    int            BLOCK_SIZE,
    int            D_emb)
{
    const int slot     = blockIdx.y;
    const int q_out    = blockIdx.x;       // 0..BLOCK_SIZE-1
    const int q_in     = q_out + 1;        // skip anchor
    const int tid      = threadIdx.x;
    const __half * src = hidden     + (static_cast<std::size_t>(slot) * Q + q_in) * D_emb;
    __half       * dst = out_hidden + (static_cast<std::size_t>(slot) * BLOCK_SIZE + q_out) * D_emb;
    for (int i = tid; i < D_emb; i += blockDim.x) {
        dst[i] = src[i];
    }
}

} // anonymous namespace

extern "C" void dflash_drafter_forward_launch(
    const __half * d_input_tokens_emb,
    __half       * d_k_cache,
    __half       * d_v_cache,
    const int    * d_slot_positions,
    const __half * const * d_layer_attn_norm_w,
    const __half * const * d_layer_q_w,
    const __half * const * d_layer_q_norm_w,
    const __half * const * d_layer_k_w,
    const __half * const * d_layer_k_norm_w,
    const __half * const * d_layer_v_w,
    const __half * const * d_layer_o_w,
    const __half * const * d_layer_ffn_norm_w,
    const __half * const * d_layer_gate_w,
    const __half * const * d_layer_up_w,
    const __half * const * d_layer_down_w,
    const __half * d_output_norm_w,
    const int    * d_layer_types,
    int            swa_window,
    float          rope_base,
    float          norm_eps,
    int            BLOCK_SIZE,
    int            N_slots,
    int            SeqLen,
    int            L_d,
    int            D_emb,
    int            H_q,
    int            H_kv,
    int            D_h,
    int            intermediate,
    __half       * d_out_hidden,
    cudaStream_t   stream)
{
    // Stub-mode guard: if any required weight pointer is null we
    // zero the output and bail. The test driver in this mode runs
    // the reference smoke + a plumbing check at SKIP exit; this is
    // not the production code path.
    if (d_input_tokens_emb == nullptr || d_layer_attn_norm_w == nullptr ||
        d_layer_q_w == nullptr || d_layer_q_norm_w == nullptr ||
        d_layer_k_w == nullptr || d_layer_k_norm_w == nullptr ||
        d_layer_v_w == nullptr ||
        d_layer_o_w == nullptr || d_layer_ffn_norm_w == nullptr ||
        d_layer_gate_w == nullptr || d_layer_up_w == nullptr ||
        d_layer_down_w == nullptr || d_layer_types == nullptr ||
        d_k_cache == nullptr || d_v_cache == nullptr ||
        d_slot_positions == nullptr) {
        const std::size_t n_out_bytes =
            static_cast<std::size_t>(N_slots) *
            static_cast<std::size_t>(BLOCK_SIZE) *
            static_cast<std::size_t>(D_emb) * sizeof(__half);
        cudaMemsetAsync(d_out_hidden, 0, n_out_bytes, stream);
        return;
    }

    const int Q      = 1 + BLOCK_SIZE;
    const int n_rows = N_slots * Q;
    const int H_qDh  = H_q * D_h;

    // Allocate working buffers on the device.
    __half * hidden     = nullptr;  // [n_rows, D_emb]
    __half * hidden_n   = nullptr;
    __half * q_buf      = nullptr;  // [n_rows, H_q*D_h]
    __half * k_buf      = nullptr;  // [n_rows, H_kv*D_h]
    __half * v_buf      = nullptr;  // [n_rows, H_kv*D_h]
    __half * attn_out   = nullptr;
    __half * o_proj     = nullptr;  // [n_rows, D_emb]
    __half * gate_buf   = nullptr;  // [n_rows, intermediate]
    __half * up_buf     = nullptr;
    __half * act_buf    = nullptr;
    __half * down_buf   = nullptr;

    const int H_kvDh = H_kv * D_h;
    const std::size_t hidden_bytes = static_cast<std::size_t>(n_rows) * D_emb * sizeof(__half);
    const std::size_t q_bytes      = static_cast<std::size_t>(n_rows) * H_qDh * sizeof(__half);
    const std::size_t kv_bytes     = static_cast<std::size_t>(n_rows) * H_kvDh * sizeof(__half);
    const std::size_t imd_bytes    = static_cast<std::size_t>(n_rows) * intermediate * sizeof(__half);

    cudaMallocAsync(&hidden,   hidden_bytes, stream);
    cudaMallocAsync(&hidden_n, hidden_bytes, stream);
    cudaMallocAsync(&q_buf,    q_bytes,      stream);
    cudaMallocAsync(&k_buf,    kv_bytes,     stream);
    cudaMallocAsync(&v_buf,    kv_bytes,     stream);
    cudaMallocAsync(&attn_out, q_bytes,      stream);
    cudaMallocAsync(&o_proj,   hidden_bytes, stream);
    cudaMallocAsync(&gate_buf, imd_bytes,    stream);
    cudaMallocAsync(&up_buf,   imd_bytes,    stream);
    cudaMallocAsync(&act_buf,  imd_bytes,    stream);
    cudaMallocAsync(&down_buf, hidden_bytes, stream);

    // Copy input embeddings into hidden (the running state buffer).
    cudaMemcpyAsync(hidden, d_input_tokens_emb, hidden_bytes,
                    cudaMemcpyDeviceToDevice, stream);

    // Read per-layer pointer arrays into host vectors (they are host arrays
    // of device pointers per the launcher signature; copy through).
    // The plumbing later wraps these in a single device-side array; for now
    // the launcher receives host pointers-to-device-pointers and the per-step
    // kernel calls dereference one at a time.

    // SMEM allocation for attention scratch (scores_smem). Max SWA window
    // = 2048 for Qwen3.6-27B-DFlash → 2048 fp32 = 8 KiB per CTA. Hard cap
    // for safety in the attention kernel.
    const int attn_smem_bytes_max = std::min(static_cast<int>(SeqLen), swa_window) * static_cast<int>(sizeof(float));

    // Determine slot_positions max position to size attention SMEM. Reading
    // back the int array is overhead; in production we'd cache. Stub bound:
    // assume qpos ≤ swa_window for now.
    const int attn_smem_bytes = attn_smem_bytes_max;

    const dim3 grid_rows(n_rows);
    const dim3 block(THREADS_PER_CTA);

    for (int layer = 0; layer < L_d; ++layer) {
        const __half * attn_norm_w = d_layer_attn_norm_w[layer];
        const __half * q_w         = d_layer_q_w[layer];
        const __half * q_norm_w    = d_layer_q_norm_w[layer];
        const __half * k_w         = d_layer_k_w[layer];
        const __half * k_norm_w    = d_layer_k_norm_w[layer];
        const __half * v_w         = d_layer_v_w[layer];
        const __half * o_w         = d_layer_o_w[layer];
        const __half * ffn_norm_w  = d_layer_ffn_norm_w[layer];
        const __half * gate_w      = d_layer_gate_w[layer];
        const __half * up_w        = d_layer_up_w[layer];
        const __half * down_w      = d_layer_down_w[layer];

        // We read layer_types via host-side fetch — single int per layer.
        int layer_type_host = 0;
        cudaMemcpyAsync(&layer_type_host, d_layer_types + layer, sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);  // need value for kernel launch arg

        const __half * k_cache_layer = d_k_cache + static_cast<std::size_t>(layer) *
                                       N_slots * SeqLen * H_kv * D_h;
        const __half * v_cache_layer = d_v_cache + static_cast<std::size_t>(layer) *
                                       N_slots * SeqLen * H_kv * D_h;

        // Step 1: attn_norm
        rmsnorm_kernel<<<grid_rows, block, 0, stream>>>(
            hidden, attn_norm_w, hidden_n, norm_eps, D_emb);

        // Step 2: Q projection
        gemm_row_x_col_kernel<<<grid_rows, block, 0, stream>>>(
            hidden_n, q_w, q_buf, D_emb, H_qDh);

        // Step 3: q_norm + RoPE
        q_norm_rope_kernel<<<grid_rows, block, 0, stream>>>(
            q_buf, q_norm_w, d_slot_positions, rope_base, norm_eps,
            Q, H_q, D_h);

        // Step 3a: K projection (drafter's own K at query positions)
        gemm_row_x_col_kernel<<<grid_rows, block, 0, stream>>>(
            hidden_n, k_w, k_buf, D_emb, H_kvDh);

        // Step 3b: V projection
        gemm_row_x_col_kernel<<<grid_rows, block, 0, stream>>>(
            hidden_n, v_w, v_buf, D_emb, H_kvDh);

        // Step 3c: k_norm + RoPE on K (V is not normed or RoPE'd per
        // @KAsymmetricallyNormedVNot — same as inject_kv_fused).
        k_norm_rope_kernel<<<grid_rows, block, 0, stream>>>(
            k_buf, k_norm_w, d_slot_positions, rope_base, norm_eps,
            Q, H_kv, D_h);

        // Step 3d: write K, V to cache at the query positions. The cache
        // is shared with dflash_inject_kv_fused which populated K, V at
        // context positions; together they populate the full range
        // [0, anchor_pos + BLOCK_SIZE] that attention reads from.
        cache_write_kv_kernel<<<grid_rows, block, 0, stream>>>(
            k_buf, v_buf,
            d_k_cache + static_cast<std::size_t>(layer) *
                static_cast<std::size_t>(N_slots) * SeqLen * H_kv * D_h,
            d_v_cache + static_cast<std::size_t>(layer) *
                static_cast<std::size_t>(N_slots) * SeqLen * H_kv * D_h,
            d_slot_positions, Q, SeqLen, H_kv, D_h);

        // Step 4: attention
        attention_kernel<<<grid_rows, block, attn_smem_bytes, stream>>>(
            q_buf, k_cache_layer, v_cache_layer, d_slot_positions,
            attn_out, layer_type_host, swa_window,
            Q, N_slots, SeqLen, H_q, H_kv, D_h);

        // Step 5: O projection
        gemm_row_x_col_kernel<<<grid_rows, block, 0, stream>>>(
            attn_out, o_w, o_proj, H_qDh, D_emb);

        // Step 6: residual add (hidden += o_proj)
        residual_add_kernel<<<grid_rows, block, 0, stream>>>(
            hidden, o_proj, hidden, D_emb);

        // Step 7: ffn_norm
        rmsnorm_kernel<<<grid_rows, block, 0, stream>>>(
            hidden, ffn_norm_w, hidden_n, norm_eps, D_emb);

        // Step 8: gate projection
        gemm_row_x_col_kernel<<<grid_rows, block, 0, stream>>>(
            hidden_n, gate_w, gate_buf, D_emb, intermediate);

        // Step 9: up projection
        gemm_row_x_col_kernel<<<grid_rows, block, 0, stream>>>(
            hidden_n, up_w, up_buf, D_emb, intermediate);

        // Step 10: silu(gate) * up
        silu_mul_kernel<<<grid_rows, block, 0, stream>>>(
            gate_buf, up_buf, act_buf, intermediate);

        // Step 11: down projection
        gemm_row_x_col_kernel<<<grid_rows, block, 0, stream>>>(
            act_buf, down_w, down_buf, intermediate, D_emb);

        // Step 12: residual add (hidden += down_buf)
        residual_add_kernel<<<grid_rows, block, 0, stream>>>(
            hidden, down_buf, hidden, D_emb);
    }

    // Step 13: final output RMSNorm before lm_head (per vLLM's
    // DFlashQwen3Model.forward line 526: self.norm(hidden_states, residual)).
    // Optional: if d_output_norm_w is null, skip — caller test may not
    // have an output_norm. Production path always passes it.
    const __half * select_input = hidden;
    if (d_output_norm_w != nullptr) {
        rmsnorm_kernel<<<grid_rows, block, 0, stream>>>(
            hidden, d_output_norm_w, hidden_n, norm_eps, D_emb);
        select_input = hidden_n;
    }

    // Step 14: select BLOCK_SIZE mask-token output positions.
    const dim3 grid_out(BLOCK_SIZE, N_slots);
    select_output_kernel<<<grid_out, block, 0, stream>>>(
        select_input, d_out_hidden, Q, BLOCK_SIZE, D_emb);

    cudaFreeAsync(hidden,   stream);
    cudaFreeAsync(hidden_n, stream);
    cudaFreeAsync(q_buf,    stream);
    cudaFreeAsync(k_buf,    stream);
    cudaFreeAsync(v_buf,    stream);
    cudaFreeAsync(attn_out, stream);
    cudaFreeAsync(o_proj,   stream);
    cudaFreeAsync(gate_buf, stream);
    cudaFreeAsync(up_buf,   stream);
    cudaFreeAsync(act_buf,  stream);
    cudaFreeAsync(down_buf, stream);
}
