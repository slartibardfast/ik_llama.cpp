// fattn-per-slot-kv-singlewarp-sm75.cu — FIX-C v5 (2026-05-16)
//
// Single-warp per-row CTA flash-attention kernel for the production
// per-slot-kv path (Qwen 3.6 27B: Dq=Dv=256, Q4_0 KV cache).
//
// Architecture:
//   - One CTA per (token, head, seq). gridDim = (n_tokens, n_heads_q, n_seqs).
//   - Block = single warp (32 threads). __launch_bounds__(WARP_SIZE, 2).
//   - Each thread handles Dk/WARP_SIZE = 8 K-dim elements (for Dk=256).
//   - K-loop iterates ne11 positions in canonical [0..ne11) order.
//   - Per K position: dot(K[k], Q) via per-thread partial + warp_reduce_sum;
//     add mask; online Welford softmax updates kqmax/kqsum/VKQ all in fp32.
//   - No cross-warp anything. fp32 throughout. No SMEM tree.
//
// Why it's batch-invariant for same-prompt slots:
//   Masked-out k contributes via (VKQ += 0; VKQ *= 1; kqsum *= 1) — all
//   fp32 no-ops. Valid-k contributions accumulate in canonical k-order,
//   which is the SAME RELATIVE ORDER for any slot whose valid set is
//   {(p_0, ..., p_{n_valid-1})} with byte-identical K/V/Q content. Slot
//   0's valid k at {0..11, 48} and slot 2's valid k at {24..35, 50}
//   both see the same 13 nonzero contributions in the same effective
//   order — same fp32 partial sum chain → byte-identical output.
//
// References:
//   - data/trace-{1..6}-2026-05-16/findings.md
//   - PHASE_MMQ_Q4_0_AR16.md §6b CX.D
//   - RESEARCH_2026-05-16.md
//   - Thinking Machines Lab "Defeating Nondeterminism in LLM Inference"

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/fattn-common.cuh"

// Single-warp per-row CTA kernel.
// Template params:
//   Dk, Dv   — head dims (must be Dk = Dv = 256 in our config; static_assert).
//   type_K   — K cache dtype (GGML_TYPE_Q4_0 in production).
//   type_V   — V cache dtype (GGML_TYPE_Q4_0 in production).
template<int Dk, int Dv, ggml_type type_K, ggml_type type_V>
__launch_bounds__(WARP_SIZE, 2)
static __global__ void flash_attn_per_slot_kv_singlewarp_kernel(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ /*sinks*/,
        float      * __restrict__ dst,
        float2     * __restrict__ /*dst_meta*/,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const float /*softcap*/,
        const uint32_t n_head_log2,
        const int ne00, const int ne01, const int ne02, const int ne03,
        const int ne10, const int ne11, const int ne12, const int ne13,
        const int ne31, const int nb31,
        const int nb01, const int nb02, const int nb03,
        const int nb11, const int nb12, const int nb13,
        const int nb21, const int nb22, const int nb23,
        const int ne0, const int ne1, const int ne2, const int ne3) {
#ifdef FP16_AVAILABLE
    static_assert(Dk == 256 && Dv == 256, "FIX-C v5 single-warp kernel hard-coded for Dk=Dv=256");
    static_assert(Dk / WARP_SIZE == 8, "8 Q/V slots per thread for Dk=256");

    constexpr int Q_PER_THREAD = Dk / WARP_SIZE;  // = 8
    constexpr int V_PER_THREAD = Dv / WARP_SIZE;  // = 8

    const int lane = threadIdx.x;
    const int tok  = blockIdx.x;
    const int head = blockIdx.y;
    const int seq  = blockIdx.z;
    const int gqa_ratio = ne02 / ne12;
    const int head_kv   = head / gqa_ratio;

    // Q row pointer (F32 input).
    const float * Q_f = (const float *) (Q + nb03*seq + nb02*head + nb01*tok);
    const half  * maskh = (const half  *) (mask + nb31*tok);  // [ne11] per token
    const float  slope  = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // K and V base pointers for this (seq, head_kv).
    const char * K_base = K + nb13*seq + nb12*head_kv;
    const char * V_base = V + nb23*seq + nb22*head_kv;

    // Q load into registers. Each thread holds 8 fp32 elements.
    // Q stride along head_dim is 1 element; index = lane + i*WARP_SIZE.
    float Q_reg[Q_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < Q_PER_THREAD; ++i) {
        const int d = lane + i*WARP_SIZE;
        Q_reg[i] = Q_f[d] * scale;
    }

    // VKQ accumulator: each thread holds V_PER_THREAD outputs.
    float VKQ[V_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < V_PER_THREAD; ++i) VKQ[i] = 0.0f;

    float kqmax = -FLT_MAX/2.0f;
    float kqsum = 0.0f;

    // Type-specific helpers for Q4_0 (production cache).
    constexpr int K_qk = QK4_0;  // = 32 (Q4_0 block size)
    constexpr int K_qr = QR4_0;  // = 2
    // bytes per K row = (Dk / qk) * sizeof(block_q4_0)
    // We use the same `nb11` ggml stride which is bytes-per-K-row.

    // K-loop in canonical [0..ne11) order — the determinism contract:
    for (int k = 0; k < ne11; ++k) {
        // -------- dot(K[k], Q_reg) --------
        // Per-thread partial: 8 multiply-add over the thread's 8 head_dim
        // elements. K is Q4_0; dequant inline per element using the block
        // structure: 32 quants share one fp16 scale.
        float partial = 0.0f;

        if constexpr (type_K == GGML_TYPE_Q4_0) {
            const block_q4_0 * K_row = (const block_q4_0 *)(K_base + k*nb11);
            #pragma unroll
            for (int i = 0; i < Q_PER_THREAD; ++i) {
                const int d        = lane + i*WARP_SIZE;
                const int blk_idx  = d / K_qk;
                const int blk_off  = d % K_qk;
                const block_q4_0 * blk = K_row + blk_idx;
                const float d_scale = __half2float(blk->d);
                // Q4_0 layout: qs[0..15] hold (d=blk_off, d=blk_off+16) as
                // (low_nibble, high_nibble) of qs[blk_off & 15].
                const int byte_idx = blk_off & (K_qk/2 - 1);  // = blk_off % 16
                const int shift    = blk_off / (K_qk/2);      // 0 or 1
                const int q_nib    = (blk->qs[byte_idx] >> (shift*4)) & 0xF;
                const float k_val  = d_scale * (float)(q_nib - 8);
                partial += k_val * Q_reg[i];
            }
        } else {
            // F16 fallback path (used for testing).
            const half * K_row = (const half *)(K_base + k*nb11);
            #pragma unroll
            for (int i = 0; i < Q_PER_THREAD; ++i) {
                const int d = lane + i*WARP_SIZE;
                partial += __half2float(K_row[d]) * Q_reg[i];
            }
        }

        // Warp reduce sum: TRACE-4 commutativity-stable for our same-value-
        // set partitioning. Same-prompt slots produce identical lane partials
        // → identical warp-reduced kq.
        const float kq = warp_reduce_sum(partial) + slope * __half2float(maskh[k]);

        // -------- online Welford softmax --------
        const float new_max  = fmaxf(kqmax, kq);
        const float diff_old = kqmax - new_max;
        const float diff_cur = kq    - new_max;
        const float scale_corr = diff_old >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_old) : 0.0f;
        const float phi        = diff_cur >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_cur) : 0.0f;
        kqmax = new_max;
        kqsum = kqsum * scale_corr + phi;
        #pragma unroll
        for (int i = 0; i < V_PER_THREAD; ++i) VKQ[i] *= scale_corr;

        // -------- V[k] · phi accumulation --------
        if constexpr (type_V == GGML_TYPE_Q4_0) {
            const block_q4_0 * V_row = (const block_q4_0 *)(V_base + k*nb21);
            #pragma unroll
            for (int i = 0; i < V_PER_THREAD; ++i) {
                const int d        = lane + i*WARP_SIZE;
                const int blk_idx  = d / K_qk;
                const int blk_off  = d % K_qk;
                const block_q4_0 * blk = V_row + blk_idx;
                const float d_scale = __half2float(blk->d);
                const int byte_idx = blk_off & (K_qk/2 - 1);
                const int shift    = blk_off / (K_qk/2);
                const int q_nib    = (blk->qs[byte_idx] >> (shift*4)) & 0xF;
                const float v_val  = d_scale * (float)(q_nib - 8);
                VKQ[i] += phi * v_val;
            }
        } else {
            const half * V_row = (const half *)(V_base + k*nb21);
            #pragma unroll
            for (int i = 0; i < V_PER_THREAD; ++i) {
                const int d = lane + i*WARP_SIZE;
                VKQ[i] += phi * __half2float(V_row[d]);
            }
        }
    }

    // Normalize and write output.
    const float inv_kqsum = 1.0f / kqsum;
    // Output ne = {Dv, Q->ne[2]=n_heads_q, Q->ne[1]=n_tokens, Q->ne[3]=n_seqs}
    // per ggml_flash_attn_ext_per_slot_kv (ggml.c:10284).
    // Memory: idx = d + head*Dv + tok*Dv*n_heads_q + seq*Dv*n_heads_q*n_tokens
    //         = d + head*Dv + tok*Dv*ne02 + seq*Dv*ne02*ne01
    // where ne01 = Q->ne[1] = n_tokens and ne02 = Q->ne[2] = n_heads_q.
    #pragma unroll
    for (int i = 0; i < V_PER_THREAD; ++i) {
        const int d = lane + i*WARP_SIZE;
        const size_t idx = (size_t)d
                         + (size_t)head * Dv
                         + (size_t)tok  * Dv * ne02
                         + (size_t)seq  * Dv * ne02 * ne01;
        dst[idx] = VKQ[i] * inv_kqsum;
    }
#else
    NO_DEVICE_CODE;
    (void)Q; (void)K; (void)V; (void)mask; (void)dst;
    (void)scale; (void)max_bias; (void)m0; (void)m1; (void)n_head_log2;
    (void)ne00; (void)ne01; (void)ne02; (void)ne03;
    (void)ne10; (void)ne11; (void)ne12; (void)ne13;
    (void)ne31; (void)nb31;
    (void)nb01; (void)nb02; (void)nb03;
    (void)nb11; (void)nb12; (void)nb13;
    (void)nb21; (void)nb22; (void)nb23;
    (void)ne0; (void)ne1; (void)ne2; (void)ne3;
#endif
}

// Dispatcher wrapper for the per-slot-kv path.
extern "C" void ggml_cuda_flash_attn_ext_per_slot_kv_singlewarp_sm75(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    GGML_ASSERT(Q && K && V && mask);
    GGML_ASSERT(Q->ne[0] == 256 && V->ne[0] == 256);
    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->type == GGML_TYPE_F16);
    GGML_ASSERT(K->type == V->type && "K and V must have same dtype");
    GGML_ASSERT((K->type == GGML_TYPE_Q4_0 || K->type == GGML_TYPE_F16) &&
                "FIX-C v5 singlewarp supports Q4_0 (production) or F16 (test) KV cache");

    float scale, max_bias, softcap;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    GGML_ASSERT(softcap == 0.0f && "softcap not supported by single-warp variant");

    const int ne00 = (int)Q->ne[0], ne01 = (int)Q->ne[1];
    const int ne02 = (int)Q->ne[2], ne03 = (int)Q->ne[3];
    const int ne10 = (int)K->ne[0], ne11 = (int)K->ne[1];
    const int ne12 = (int)K->ne[2], ne13 = (int)K->ne[3];

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // ggml_flash_attn_ext_per_slot_kv output ne = {Dv, Q->ne[2], Q->ne[1], Q->ne[3]}.
    // Grid: (n_tokens, n_heads_q, n_seqs).
    const dim3 grid((unsigned)Q->ne[1], (unsigned)Q->ne[2], (unsigned)Q->ne[3]);
    const dim3 block(WARP_SIZE, 1, 1);

    auto launch_kernel = [&](auto kernel) {
        kernel<<<grid, block, 0, ctx.stream()>>>(
            (const char *) Q->data,
            (const char *) K->data,
            (const char *) V->data,
            (const char *) mask->data,
            nullptr,
            (float *) dst->data,
            nullptr,
            scale, max_bias, m0, m1, softcap, n_head_log2,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            (int)mask->ne[1], (int)mask->nb[1],
            (int)Q->nb[1], (int)Q->nb[2], (int)Q->nb[3],
            (int)K->nb[1], (int)K->nb[2], (int)K->nb[3],
            (int)V->nb[1], (int)V->nb[2], (int)V->nb[3],
            (int)dst->ne[0], (int)dst->ne[1], (int)dst->ne[2], (int)dst->ne[3]);
    };
    if (K->type == GGML_TYPE_Q4_0) {
        launch_kernel(flash_attn_per_slot_kv_singlewarp_kernel<256, 256, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0>);
    } else { // GGML_TYPE_F16
        launch_kernel(flash_attn_per_slot_kv_singlewarp_kernel<256, 256, GGML_TYPE_F16, GGML_TYPE_F16>);
    }
    CUDA_CHECK(cudaGetLastError());
}
