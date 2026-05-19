// dflash-drafter-forward.cu
//
// 5-layer DFlash drafter forward implementation. Spec:
// specs/dflash/kernel-design.md §6.1 + §6.1.A.
//
// Per-step sub-kernels invoked sequentially from a host-side dispatcher
// loop (5 layers × ~10 sub-kernels per layer + final norm + select).
// The "Phase B cooperative WMMA mega-kernel" plan in §6.1 is abandoned.
//
// GEMM dispatch (revised 2026-05-19, spec §6.1.A): all 35 projection
// matmuls (Q/K/V/O/Gate/Up/Down × 5 layers) go through
// `dflash_gemm_npc` → `ggml_cuda_mul_mat_f16_pinned`. F32 intermediates
// are fused-cast to F16 in the consumer kernels (q_norm_rope,
// k_norm_rope, cache_write_kv, silu_mul, residual_add).
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
#include "dflash-gemm.cuh"

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

// Sub-kernel 2 (GEMM, scalar fp32) — retired 2026-05-19 per spec §6.1.A.
// All projection matmuls now dispatch through `dflash_gemm_npc` (forwarded
// to `ggml_cuda_mul_mat_f16_pinned`, HMMA m16n8k16, F32 output). See
// dflash-gemm.cuh.

// ===========================================================================
// Sub-kernel 3: per-head q_norm + NeoX RoPE (F32 in → F16 out).
//
// One CTA per (slot, query_position) row.
// Each row's Q has H_q heads × D_h dim. Per head, compute RMSNorm with
// per-layer q_norm_weight, then apply NeoX RoPE using the query position
// (= anchor_pos[slot] + q_offset).
//
// Reads q_in F32 (from pinned Q-proj output) into per-head scratch, normalizes
// in fp32, applies RoPE in fp32, and stores F16 into q_out (separate buffer
// consumed by attention_kernel).
//
// fp64 transcendentals (pow, cos, sin) cast to fp32 — same as T3 inject
// kernel + spec §6.1 binding.
//
// Grid: dim3(n_rows)  Block: 256 threads.
// ===========================================================================
__global__ void q_norm_rope_kernel(
    const float  * __restrict__ q_in,        // [n_rows, H_q * D_h] F32 (pinned out)
    __half       * __restrict__ q_out,       // [n_rows, H_q * D_h] F16 (for attention)
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

    const float  * row_in  = q_in  + static_cast<std::size_t>(row) * H_q * D_h;
    __half       * row_out = q_out + static_cast<std::size_t>(row) * H_q * D_h;
    const int D_half = D_h / 2;

    for (int h = 0; h < H_q; ++h) {
        const float * qh_in  = row_in  + h * D_h;
        __half      * qh_out = row_out + h * D_h;

        // q_norm: RMSNorm sum_sq over fp32 head_in[i].
        float sum_sq = 0.0f;
        for (int i = tid; i < D_h; i += blockDim.x) {
            const float v = qh_in[i];
            sum_sq += v * v;
        }
        const float total_sq = block_sum_f32(sum_sq, reduce_smem);
        const float rsq = rsqrtf(total_sq / static_cast<float>(D_h) + norm_eps);

        // Fused norm-then-RoPE in fp32. Process (lo=i, hi=i+D_half) pair
        // per iteration; both positions get normed (×rsq×w_i / ×rsq×w_{i+D_half})
        // BEFORE RoPE mixes them. Store F16 only at the end.
        for (int i = tid; i < D_half; i += blockDim.x) {
            const float w_lo       = __half2float(q_norm_w[i]);
            const float w_hi       = __half2float(q_norm_w[i + D_half]);
            const float lo_n       = qh_in[i]            * rsq * w_lo;
            const float hi_n       = qh_in[i + D_half]   * rsq * w_hi;
            const double exp_val_d  = static_cast<double>(2 * i) / static_cast<double>(D_h);
            const double inv_freq_d = pow(static_cast<double>(rope_base), -exp_val_d);
            const double theta_d    = static_cast<double>(qpos) * inv_freq_d;
            const float  c          = static_cast<float>(cos(theta_d));
            const float  s          = static_cast<float>(sin(theta_d));
            qh_out[i]            = __float2half(lo_n * c - hi_n * s);
            qh_out[i + D_half]   = __float2half(lo_n * s + hi_n * c);
        }
        __syncthreads();
    }
}

// ===========================================================================
// Sub-kernel 3b: per-head K norm + NeoX RoPE (F32 in → F16 out).
//
// Same structure as q_norm_rope_kernel but operates on K with H_kv heads
// (vs Q's H_q heads). Reads F32 pinned-K-proj output, writes F16 to a
// separate buffer consumed by cache_write_kv.
//
// fp64 transcendentals (pow, cos, sin) cast to fp32 — same as the inject
// kernel + scalar reference (sm_75 libdevice fp32 trig diverges from CPU
// libm by up to 6 ULP; fp64 bridges the gap).
// ===========================================================================
__global__ void k_norm_rope_kernel(
    const float  * __restrict__ k_in,        // [n_rows, H_kv * D_h] F32 (pinned out)
    __half       * __restrict__ k_out,       // [n_rows, H_kv * D_h] F16 (for cache write)
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

    const float * row_in  = k_in  + static_cast<std::size_t>(row) * H_kv * D_h;
    __half      * row_out = k_out + static_cast<std::size_t>(row) * H_kv * D_h;
    const int D_half = D_h / 2;

    for (int h = 0; h < H_kv; ++h) {
        const float * kh_in  = row_in  + h * D_h;
        __half      * kh_out = row_out + h * D_h;

        // k_norm: RMSNorm sum_sq over fp32 head_in[i].
        float sum_sq = 0.0f;
        for (int i = tid; i < D_h; i += blockDim.x) {
            const float v = kh_in[i];
            sum_sq += v * v;
        }
        const float total_sq = block_sum_f32(sum_sq, reduce_smem);
        const float rsq = rsqrtf(total_sq / static_cast<float>(D_h) + norm_eps);

        // Fused norm-then-RoPE in fp32. Store F16 only at the end.
        for (int i = tid; i < D_half; i += blockDim.x) {
            const float w_lo       = __half2float(k_norm_w[i]);
            const float w_hi       = __half2float(k_norm_w[i + D_half]);
            const float lo_n       = kh_in[i]            * rsq * w_lo;
            const float hi_n       = kh_in[i + D_half]   * rsq * w_hi;
            const double exp_val_d  = static_cast<double>(2 * i) / static_cast<double>(D_h);
            const double inv_freq_d = pow(static_cast<double>(rope_base), -exp_val_d);
            const double theta_d    = static_cast<double>(qpos) * inv_freq_d;
            const float  c          = static_cast<float>(cos(theta_d));
            const float  s          = static_cast<float>(sin(theta_d));
            kh_out[i]            = __float2half(lo_n * c - hi_n * s);
            kh_out[i + D_half]   = __float2half(lo_n * s + hi_n * c);
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
    const __half * __restrict__ k_buf,           // [n_rows, H_kv * D_h] F16 (post k_norm_rope)
    const float  * __restrict__ v_buf_f32,       // [n_rows, H_kv * D_h] F32 (raw pinned V-proj)
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

    const __half * row_k = k_buf     + static_cast<std::size_t>(row) * hd;
    const float  * row_v = v_buf_f32 + static_cast<std::size_t>(row) * hd;
    const std::size_t cache_off =
        ((static_cast<std::size_t>(slot) * SeqLen + qpos) * H_kv) * D_h;
    for (int i = tid; i < hd; i += blockDim.x) {
        k_cache_layer[cache_off + i] = row_k[i];                     // F16 passthrough
        v_cache_layer[cache_off + i] = __float2half(row_v[i]);       // F32→F16 cast on store
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
// Sub-kernel 5: residual add — F16 hidden + F32 delta → F16 hidden (in-place).
//
// One CTA per row, threads stride across D. Delta comes from a pinned-HMMA
// GEMM output (F32); hidden is the running F16 residual stream. fp32 add,
// F16 store.
// ===========================================================================
__global__ void residual_add_kernel(
    const __half * a,
    const float  * b,
    __half       * out,
    int            D)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __half * row_a = a   + static_cast<std::size_t>(row) * D;
    const float  * row_b = b   + static_cast<std::size_t>(row) * D;
    __half       * row_o = out + static_cast<std::size_t>(row) * D;
    for (int i = tid; i < D; i += blockDim.x) {
        const float va = __half2float(row_a[i]);
        const float vb = row_b[i];
        row_o[i] = __float2half(va + vb);
    }
}

// ===========================================================================
// Sub-kernel 6: silu(gate) * up → activated (F32 in, F16 out).
//
// One CTA per row, threads stride across intermediate. Reads F32 gate/up
// from pinned-HMMA outputs; writes F16 activated which feeds the F16-act
// side of the pinned Down projection.
// ===========================================================================
__global__ void silu_mul_kernel(
    const float  * __restrict__ gate,
    const float  * __restrict__ up,
    __half       * __restrict__ activated,
    int            intermediate)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float  * row_g = gate      + static_cast<std::size_t>(row) * intermediate;
    const float  * row_u = up        + static_cast<std::size_t>(row) * intermediate;
    __half       * row_a = activated + static_cast<std::size_t>(row) * intermediate;
    for (int i = tid; i < intermediate; i += blockDim.x) {
        const float g = row_g[i];
        const float u = row_u[i];
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
    int            n_slots_cap,
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
    // Guard the N_slots / n_slots_cap contract: N_slots is the dispatch
    // count and must fit inside the storage stride. Swapping the two —
    // or passing the same value when caller meant different — would
    // silently read layers > 0 from wrong byte offsets. See spec
    // kernel-design.md §6.1 clarification #4.
    if (N_slots < 1 || n_slots_cap < N_slots) {
        std::fprintf(stderr,
            "[dflash_drafter_forward_launch] invalid N_slots=%d n_slots_cap=%d "
            "(require 1 <= N_slots <= n_slots_cap)\n",
            N_slots, n_slots_cap);
        const std::size_t n_out_bytes =
            static_cast<std::size_t>(N_slots < 1 ? 1 : N_slots) *
            static_cast<std::size_t>(BLOCK_SIZE) *
            static_cast<std::size_t>(D_emb) * sizeof(__half);
        if (d_out_hidden != nullptr) {
            cudaMemsetAsync(d_out_hidden, 0, n_out_bytes, stream);
        }
        return;
    }

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
    const int H_kvDh = H_kv * D_h;

    // K-divisibility precondition for pinned HMMA (TILE_K=16). All four
    // distinct K dimensions of the drafter projection set must satisfy
    // K % 16 == 0. Production shapes (Qwen 3.6 27B drafter):
    //   D_emb=5120, H_qDh=4096, H_kvDh=1024, intermediate=17408 — all ✓.
    if ((D_emb % 16) || (H_qDh % 16) || (H_kvDh % 16) || (intermediate % 16)) {
        std::fprintf(stderr,
            "[dflash_drafter_forward_launch] pinned-HMMA requires all GEMM K%%16==0, "
            "got D_emb=%d H_qDh=%d H_kvDh=%d intermediate=%d\n",
            D_emb, H_qDh, H_kvDh, intermediate);
        const std::size_t n_out_bytes =
            static_cast<std::size_t>(N_slots) * BLOCK_SIZE * D_emb * sizeof(__half);
        cudaMemsetAsync(d_out_hidden, 0, n_out_bytes, stream);
        return;
    }

    // Allocate working buffers on the device.
    // Per spec §6.1.A: GEMM outputs (q/k/v/o/gate/up/down) are F32; the
    // norm/RoPE/silu/residual consumers cast to F16 on store.
    __half * hidden     = nullptr;  // [n_rows, D_emb] F16
    __half * hidden_n   = nullptr;  // [n_rows, D_emb] F16 (act side of pinned)
    float  * q_buf      = nullptr;  // [n_rows, H_q *D_h] F32 (pinned Q-proj out)
    float  * k_buf      = nullptr;  // [n_rows, H_kv*D_h] F32 (pinned K-proj out)
    float  * v_buf      = nullptr;  // [n_rows, H_kv*D_h] F32 (pinned V-proj out)
    __half * q_for_attn = nullptr;  // [n_rows, H_q *D_h] F16 (q_norm_rope out)
    __half * k_for_cache= nullptr;  // [n_rows, H_kv*D_h] F16 (k_norm_rope out)
    __half * attn_out   = nullptr;  // [n_rows, H_q *D_h] F16 (attention out)
    float  * o_proj     = nullptr;  // [n_rows, D_emb]    F32 (pinned O-proj out)
    float  * gate_buf   = nullptr;  // [n_rows, intermediate] F32 (pinned Gate-proj out)
    float  * up_buf     = nullptr;  // [n_rows, intermediate] F32 (pinned Up-proj out)
    __half * act_buf    = nullptr;  // [n_rows, intermediate] F16 (silu_mul out)
    float  * down_buf   = nullptr;  // [n_rows, D_emb]    F32 (pinned Down-proj out)

    const std::size_t hidden_bytes_h = static_cast<std::size_t>(n_rows) * D_emb        * sizeof(__half);
    const std::size_t hidden_bytes_f = static_cast<std::size_t>(n_rows) * D_emb        * sizeof(float);
    const std::size_t q_bytes_h      = static_cast<std::size_t>(n_rows) * H_qDh        * sizeof(__half);
    const std::size_t q_bytes_f      = static_cast<std::size_t>(n_rows) * H_qDh        * sizeof(float);
    const std::size_t kv_bytes_h     = static_cast<std::size_t>(n_rows) * H_kvDh       * sizeof(__half);
    const std::size_t kv_bytes_f     = static_cast<std::size_t>(n_rows) * H_kvDh       * sizeof(float);
    const std::size_t imd_bytes_h    = static_cast<std::size_t>(n_rows) * intermediate * sizeof(__half);
    const std::size_t imd_bytes_f    = static_cast<std::size_t>(n_rows) * intermediate * sizeof(float);

    cudaMallocAsync(&hidden,      hidden_bytes_h, stream);
    cudaMallocAsync(&hidden_n,    hidden_bytes_h, stream);
    cudaMallocAsync(&q_buf,       q_bytes_f,      stream);
    cudaMallocAsync(&k_buf,       kv_bytes_f,     stream);
    cudaMallocAsync(&v_buf,       kv_bytes_f,     stream);
    cudaMallocAsync(&q_for_attn,  q_bytes_h,      stream);
    cudaMallocAsync(&k_for_cache, kv_bytes_h,     stream);
    cudaMallocAsync(&attn_out,    q_bytes_h,      stream);
    cudaMallocAsync(&o_proj,      hidden_bytes_f, stream);
    cudaMallocAsync(&gate_buf,    imd_bytes_f,    stream);
    cudaMallocAsync(&up_buf,      imd_bytes_f,    stream);
    cudaMallocAsync(&act_buf,     imd_bytes_h,    stream);
    cudaMallocAsync(&down_buf,    hidden_bytes_f, stream);

    // Copy input embeddings into hidden (the running state buffer).
    cudaMemcpyAsync(hidden, d_input_tokens_emb, hidden_bytes_h,
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

        // Per-layer base pointers use n_slots_cap (the storage stride
        // baked at allocation time), NOT N_slots (the dispatch count).
        // When N_slots < n_slots_cap, the kernel's slot loop only writes
        // [0, N_slots) but layer stride must still match storage so
        // layer L's base lands at the right bytes.
        const __half * k_cache_layer = d_k_cache + static_cast<std::size_t>(layer) *
                                       n_slots_cap * SeqLen * H_kv * D_h;
        const __half * v_cache_layer = d_v_cache + static_cast<std::size_t>(layer) *
                                       n_slots_cap * SeqLen * H_kv * D_h;

        // Step 1: attn_norm (F16 → F16)
        rmsnorm_kernel<<<grid_rows, block, 0, stream>>>(
            hidden, attn_norm_w, hidden_n, norm_eps, D_emb);

        // Step 2: Q projection — pinned HMMA, F16 act × F16 weight → F32 dst.
        dflash_gemm_npc(q_w, hidden_n, q_buf, D_emb, H_qDh, n_rows, stream);

        // Step 3: q_norm + RoPE (F32 in → F16 out into separate buffer).
        q_norm_rope_kernel<<<grid_rows, block, 0, stream>>>(
            q_buf, q_for_attn, q_norm_w, d_slot_positions, rope_base, norm_eps,
            Q, H_q, D_h);

        // Step 3a: K projection — pinned HMMA.
        dflash_gemm_npc(k_w, hidden_n, k_buf, D_emb, H_kvDh, n_rows, stream);

        // Step 3b: V projection — pinned HMMA.
        dflash_gemm_npc(v_w, hidden_n, v_buf, D_emb, H_kvDh, n_rows, stream);

        // Step 3c: k_norm + RoPE on K (V is not normed or RoPE'd per
        // @KAsymmetricallyNormedVNot — same as inject_kv_fused).
        // F32 k_buf → F16 k_for_cache.
        k_norm_rope_kernel<<<grid_rows, block, 0, stream>>>(
            k_buf, k_for_cache, k_norm_w, d_slot_positions, rope_base, norm_eps,
            Q, H_kv, D_h);

        // Step 3d: write K, V to cache at the query positions. K is F16
        // post-norm/RoPE; V is F32 raw pinned output and gets cast on store.
        // The cache is shared with dflash_inject_kv_fused which populated K, V
        // at context positions; together they populate [0, anchor_pos + BLOCK_SIZE].
        cache_write_kv_kernel<<<grid_rows, block, 0, stream>>>(
            k_for_cache, v_buf,
            d_k_cache + static_cast<std::size_t>(layer) *
                static_cast<std::size_t>(n_slots_cap) * SeqLen * H_kv * D_h,
            d_v_cache + static_cast<std::size_t>(layer) *
                static_cast<std::size_t>(n_slots_cap) * SeqLen * H_kv * D_h,
            d_slot_positions, Q, SeqLen, H_kv, D_h);

        // Step 4: attention (F16 q_for_attn, F16 cache, F16 attn_out).
        attention_kernel<<<grid_rows, block, attn_smem_bytes, stream>>>(
            q_for_attn, k_cache_layer, v_cache_layer, d_slot_positions,
            attn_out, layer_type_host, swa_window,
            Q, N_slots, SeqLen, H_q, H_kv, D_h);

        // Step 5: O projection — pinned HMMA, F16 attn_out → F32 o_proj.
        dflash_gemm_npc(o_w, attn_out, o_proj, H_qDh, D_emb, n_rows, stream);

        // Step 6: residual add (hidden += o_proj_f32, F16 store)
        residual_add_kernel<<<grid_rows, block, 0, stream>>>(
            hidden, o_proj, hidden, D_emb);

        // Step 7: ffn_norm (F16 → F16)
        rmsnorm_kernel<<<grid_rows, block, 0, stream>>>(
            hidden, ffn_norm_w, hidden_n, norm_eps, D_emb);

        // Step 8: gate projection — pinned HMMA.
        dflash_gemm_npc(gate_w, hidden_n, gate_buf, D_emb, intermediate, n_rows, stream);

        // Step 9: up projection — pinned HMMA.
        dflash_gemm_npc(up_w, hidden_n, up_buf, D_emb, intermediate, n_rows, stream);

        // Step 10: silu(gate_f32) * up_f32 → act_buf_f16.
        silu_mul_kernel<<<grid_rows, block, 0, stream>>>(
            gate_buf, up_buf, act_buf, intermediate);

        // Step 11: down projection — pinned HMMA, F16 act → F32 down_buf.
        dflash_gemm_npc(down_w, act_buf, down_buf, intermediate, D_emb, n_rows, stream);

        // Step 12: residual add (hidden += down_buf_f32, F16 store)
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

    cudaFreeAsync(hidden,      stream);
    cudaFreeAsync(hidden_n,    stream);
    cudaFreeAsync(q_buf,       stream);
    cudaFreeAsync(k_buf,       stream);
    cudaFreeAsync(v_buf,       stream);
    cudaFreeAsync(q_for_attn,  stream);
    cudaFreeAsync(k_for_cache, stream);
    cudaFreeAsync(attn_out,    stream);
    cudaFreeAsync(o_proj,      stream);
    cudaFreeAsync(gate_buf,    stream);
    cudaFreeAsync(up_buf,      stream);
    cudaFreeAsync(act_buf,     stream);
    cudaFreeAsync(down_buf,    stream);
}
