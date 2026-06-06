// fattn-per-slot-kv-split-sm75.cu — PHASE_FATTN_KV_SPLIT_SM75 bring-up v1 (2026-06-06)
//
// KV-split (flash-decoding style) replacement for the singlewarp per-slot FA
// kernel at DECODE (ne01 == 1). Root cause it fixes: the singlewarp kernel
// streams the full KV with one warp per (token, q-head) — 12 warps/GPU at
// decode, O(depth) serial latency-bound cost (ncu: SM 2%, DRAM 0.25%,
// occupancy 3.1% at 2k KV; ~20 ms/launch at 44k KV x 16 attention layers
// = 320 ms of a 362 ms token). See docs/active/PHASE_FATTN_KV_SPLIT_SM75.md.
//
// Design (v1):
//   - Kernel 1 (partial): grid (n_chunks, n_kv_heads, n_seqs), block
//     (WARP_SIZE, gqa_ratio, 1). Warp w handles q-head = kv_head*gqa + w.
//     Each CTA processes LOGICAL k in [chunk*CHUNK, min((chunk+1)*CHUNK, ne11))
//     in canonical ascending order with the singlewarp kernel's exact fp32
//     arithmetic (4-way ILP dot, FTZ-threshold online softmax). The gqa warps
//     read the same K/V lines concurrently — L1 serves the 6x GQA dedup in v1
//     (explicit smem double-buffering is the v2 optimization, ncu-guided).
//     Emits per-(chunk, q-head) partials: O[Dv], (m, l).
//   - Kernel 2 (merge): one warp per (q-head, seq); folds partials in FIXED
//     ascending chunk order with the FA2 combine and the same FTZ guard;
//     writes dst with the singlewarp epilogue's exact layout.
//
// Determinism: chunk boundaries are functions of LOGICAL per-seq k only
// (CHUNK is a compile-time constant, a multiple of the paged BLOCK=64), the
// in-chunk chain is the canonical scalar chain, and the merge order is fixed
// ascending chunk id => identical (Q, K/V content, valid set) yields a
// byte-identical fp32 chain independent of slot/batch placement under the
// per-slot paged layout (T5.x). The TRACE-6 translation hazard applied to the
// OLD global-packed unified KV; the paged layout removed it — hence this path
// requires block_table != nullptr and falls back to singlewarp for legacy.
// Outputs are deterministic but NOT bit-identical to the singlewarp chain
// (different fixed grouping) — NPC hashes are re-baselined once (phase doc
// step 3).
//
// Inherited determinism micro-lessons (mandatory):
//   - FTZ-threshold guard on exp underflow (SOFTMAX_FTZ_THRESHOLD), exactly
//     as the singlewarp kernel computes scale_corr/phi.
//   - fp32 throughout; no mma in any partial chain.
//   - masked positions contribute structurally (phi = 0), never numerically.

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/fattn-common.cuh"

static constexpr int PSKV_SPLIT_CHUNK_TOKENS      = 1024; // multiple of paged BLOCK_SIZE 64
static constexpr int PSKV_SPLIT_PAGED_BLOCK_TOKENS = 64;

template<int Dk, int Dv, ggml_type type_K, ggml_type type_V>
__launch_bounds__(WARP_SIZE*8, 2)
static __global__ void flash_attn_per_slot_kv_split_partial(
        const char * __restrict__ Q,
        const char * __restrict__ K_direct,
        const char * __restrict__ V_direct,
        const char * __restrict__ mask,
        const int  * __restrict__ block_table,     // REQUIRED non-null (paged only)
        const int                 n_blocks_per_seq,
        float      * __restrict__ part_O,          // [seq][chunk][q_head][Dv]
        float2     * __restrict__ part_ml,         // [seq][chunk][q_head] {m, l}
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const int ne02,                             // n_heads_q
        const int ne11,                             // logical KV length (bucketed)
        const int ne12,                             // n_kv_heads
        const int nb31, const int nb33,             // mask strides (tok, seq)
        const int nb01, const int nb02, const int nb03,
        const int nb11,
        const int nb21,
        const int n_chunks) {
#ifdef FP16_AVAILABLE
    static_assert(Dk == 256 && Dv == 256, "split kernel hard-coded for Dk=Dv=256");
    constexpr int Q_PER_THREAD = Dk / WARP_SIZE;  // 8
    constexpr int V_PER_THREAD = Dv / WARP_SIZE;  // 8

    const int lane    = threadIdx.x;
    const int warp    = threadIdx.y;               // q-head within the GQA group
    const int chunk   = blockIdx.x;
    const int head_kv = blockIdx.y;
    const int seq     = blockIdx.z;
    const int gqa     = blockDim.y;                // gqa_ratio (= ne02/ne12)
    const int head    = head_kv * gqa + warp;      // q-head index

    const int k_begin = chunk * PSKV_SPLIT_CHUNK_TOKENS;
    if (k_begin >= ne11) return;                   // inactive chunk (merge clamps)
    const int k_end   = min(k_begin + PSKV_SPLIT_CHUNK_TOKENS, ne11);

    // Q row (F32), decode: tok = 0.
    const float * Q_f   = (const float *) (Q + (size_t)nb03*seq + (size_t)nb02*head);
    const half  * maskh = (const half  *) (mask + (size_t)nb33*seq + (size_t)nb31*0);
    const float   slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    const int * bt_seq          = block_table + (size_t)seq * n_blocks_per_seq;
    const size_t paged_nb12     = (size_t)nb11 * PSKV_SPLIT_PAGED_BLOCK_TOKENS;
    const size_t paged_nb13     = paged_nb12 * ne12;
    const size_t paged_nb22     = (size_t)nb21 * PSKV_SPLIT_PAGED_BLOCK_TOKENS;
    const size_t paged_nb23     = paged_nb22 * ne12;

    float Q_reg[Q_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < Q_PER_THREAD; ++i) {
        Q_reg[i] = Q_f[lane + i*WARP_SIZE] * scale;
    }

    float VKQ[V_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < V_PER_THREAD; ++i) VKQ[i] = 0.0f;
    float kqmax = -FLT_MAX/2.0f;
    float kqsum = 0.0f;

    constexpr int K_qk  = QK4_0;
    constexpr int ILP_W = 4;
    // CHUNK is a multiple of 4; only the LAST active chunk can have a tail.
    const int n_in_chunk   = k_end - k_begin;
    const int n_aligned    = (n_in_chunk / ILP_W) * ILP_W;
    const int k_end_align  = k_begin + n_aligned;

    auto kv_bases = [&](int k, const char *& K_base, const char *& V_base, int & off) {
        const int bid = bt_seq[k / PSKV_SPLIT_PAGED_BLOCK_TOKENS];
        K_base = K_direct + (size_t)bid * paged_nb13 + (size_t)head_kv * paged_nb12;
        V_base = V_direct + (size_t)bid * paged_nb23 + (size_t)head_kv * paged_nb22;
        off    = k & (PSKV_SPLIT_PAGED_BLOCK_TOKENS - 1);
    };

    auto dot_k_row = [&](const char * K_base, int off) -> float {
        float partial = 0.0f;
        if constexpr (type_K == GGML_TYPE_Q4_0) {
            const block_q4_0 * K_row = (const block_q4_0 *)(K_base + (size_t)off*nb11);
            #pragma unroll
            for (int i = 0; i < Q_PER_THREAD; ++i) {
                const int d        = lane + i*WARP_SIZE;
                const block_q4_0 * blk = K_row + d / K_qk;
                const int blk_off  = d % K_qk;
                const int byte_idx = blk_off & (K_qk/2 - 1);
                const int shift    = blk_off / (K_qk/2);
                const int q_nib    = (blk->qs[byte_idx] >> (shift*4)) & 0xF;
                partial += __half2float(blk->d) * (float)(q_nib - 8) * Q_reg[i];
            }
        } else {
            const half * K_row = (const half *)(K_base + (size_t)off*nb11);
            #pragma unroll
            for (int i = 0; i < Q_PER_THREAD; ++i) {
                partial += __half2float(K_row[lane + i*WARP_SIZE]) * Q_reg[i];
            }
        }
        return partial;
    };

    auto accum_v_row = [&](const char * V_base, int off, float phi) {
        if constexpr (type_V == GGML_TYPE_Q4_0) {
            const block_q4_0 * V_row = (const block_q4_0 *)(V_base + (size_t)off*nb21);
            #pragma unroll
            for (int i = 0; i < V_PER_THREAD; ++i) {
                const int d        = lane + i*WARP_SIZE;
                const block_q4_0 * blk = V_row + d / K_qk;
                const int blk_off  = d % K_qk;
                const int byte_idx = blk_off & (K_qk/2 - 1);
                const int shift    = blk_off / (K_qk/2);
                const int q_nib    = (blk->qs[byte_idx] >> (shift*4)) & 0xF;
                VKQ[i] += phi * __half2float(blk->d) * (float)(q_nib - 8);
            }
        } else {
            const half * V_row = (const half *)(V_base + (size_t)off*nb21);
            #pragma unroll
            for (int i = 0; i < V_PER_THREAD; ++i) {
                VKQ[i] += phi * __half2float(V_row[lane + i*WARP_SIZE]);
            }
        }
    };

    auto softmax_step = [&](float kq) {
        const float new_max    = fmaxf(kqmax, kq);
        const float diff_old   = kqmax - new_max;
        const float diff_cur   = kq    - new_max;
        const float scale_corr = diff_old >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_old) : 0.0f;
        const float phi        = diff_cur >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_cur) : 0.0f;
        kqmax = new_max;
        kqsum = kqsum * scale_corr + phi;
        #pragma unroll
        for (int i = 0; i < V_PER_THREAD; ++i) VKQ[i] *= scale_corr;
        return phi;
    };

    // ILP-4 main loop over the chunk, canonical ascending k.
    for (int k = k_begin; k < k_end_align; k += ILP_W) {
        const char * K_base; const char * V_base; int off;
        kv_bases(k, K_base, V_base, off);   // BLOCK=64 is a multiple of 4: one lookup per quad

        float kq_arr[ILP_W];
        #pragma unroll
        for (int s = 0; s < ILP_W; ++s) {
            kq_arr[s] = dot_k_row(K_base, off + s);
        }
        #pragma unroll
        for (int s = 0; s < ILP_W; ++s) {
            kq_arr[s] = warp_reduce_sum(kq_arr[s]) + slope * __half2float(maskh[k + s]);
        }
        #pragma unroll
        for (int s = 0; s < ILP_W; ++s) {
            const float phi = softmax_step(kq_arr[s]);
            accum_v_row(V_base, off + s, phi);
        }
    }
    // Scalar tail (last active chunk only).
    for (int k = k_end_align; k < k_end; ++k) {
        const char * K_base; const char * V_base; int off;
        kv_bases(k, K_base, V_base, off);
        float kq = warp_reduce_sum(dot_k_row(K_base, off)) + slope * __half2float(maskh[k]);
        const float phi = softmax_step(kq);
        accum_v_row(V_base, off, phi);
    }

    // Emit the partial: O (unnormalized), m, l.
    const size_t pidx = ((size_t)seq * n_chunks + chunk) * ne02 + head;
    float * O_out = part_O + pidx * Dv;
    #pragma unroll
    for (int i = 0; i < V_PER_THREAD; ++i) {
        O_out[lane + i*WARP_SIZE] = VKQ[i];
    }
    if (lane == 0) {
        part_ml[pidx] = make_float2(kqmax, kqsum);
    }
#else
    NO_DEVICE_CODE;
    (void)Q; (void)K_direct; (void)V_direct; (void)mask; (void)block_table;
    (void)n_blocks_per_seq; (void)part_O; (void)part_ml; (void)scale; (void)max_bias;
    (void)m0; (void)m1; (void)n_head_log2; (void)ne02; (void)ne11; (void)ne12;
    (void)nb31; (void)nb33; (void)nb01; (void)nb02; (void)nb03; (void)nb11; (void)nb21;
    (void)n_chunks;
#endif
}

template<int Dv>
static __global__ void flash_attn_per_slot_kv_split_merge(
        const float  * __restrict__ part_O,
        const float2 * __restrict__ part_ml,
        float        * __restrict__ dst,
        const int ne01,        // n_tokens (= 1 at decode)
        const int ne02,        // n_heads_q
        const int n_chunks,    // allocated chunk slots (== gridDim.x of kernel 1)
        const int n_active) {  // ceil(ne11 / CHUNK) — chunks that actually ran
    constexpr int V_PER_THREAD = Dv / WARP_SIZE;
    const int lane = threadIdx.x;
    const int head = blockIdx.x;
    const int seq  = blockIdx.z;

    float O[V_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < V_PER_THREAD; ++i) O[i] = 0.0f;
    float m = -FLT_MAX/2.0f;
    float l = 0.0f;

    // FIXED ascending chunk order — the determinism contract for the merge.
    for (int c = 0; c < n_active; ++c) {
        const size_t pidx = ((size_t)seq * n_chunks + c) * ne02 + head;
        const float2 ml   = part_ml[pidx];
        const float new_m     = fmaxf(m, ml.x);
        const float diff_old  = m    - new_m;
        const float diff_cur  = ml.x - new_m;
        const float corr_old  = diff_old >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_old) : 0.0f;
        const float corr_cur  = diff_cur >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_cur) : 0.0f;
        const float * Oc = part_O + pidx * Dv;
        #pragma unroll
        for (int i = 0; i < V_PER_THREAD; ++i) {
            const int d = lane + i*WARP_SIZE;
            O[i] = O[i] * corr_old + Oc[d] * corr_cur;
        }
        m = new_m;
        l = l * corr_old + ml.y * corr_cur;
    }

    const float inv_l = 1.0f / l;
    // dst layout identical to the singlewarp epilogue:
    // idx = d + head*Dv + tok*Dv*ne02 + seq*Dv*ne02*ne01 (decode: tok = 0)
    #pragma unroll
    for (int i = 0; i < V_PER_THREAD; ++i) {
        const int d = lane + i*WARP_SIZE;
        const size_t idx = (size_t)d + (size_t)head * Dv + (size_t)seq * Dv * ne02 * ne01;
        dst[idx] = O[i] * inv_l;
    }
}

// Host wrapper. Caller (dispatcher in fattn-per-slot-kv-sm75.cu) guarantees:
// decode (Q->ne[1] == 1), paged KV (block_table non-null), Dk = Dv = 256,
// K/V type in {Q4_0, F16}.
void ggml_cuda_flash_attn_ext_per_slot_kv_split_sm75(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst);  // self-declaration (-Wmissing-declarations)
void ggml_cuda_flash_attn_ext_per_slot_kv_split_sm75(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    // src[5] (per_row_k_bound) is decorative — the mask is the validity
    // primitive (see singlewarp CY.F.17); intentionally not read here.
    const ggml_tensor * block_table     = dst->src[6];

    GGML_ASSERT(Q->ne[1] == 1 && "split path is decode-only (ne01 == 1)");
    GGML_ASSERT(block_table && block_table->type == GGML_TYPE_I32);
    GGML_ASSERT(Q->type == GGML_TYPE_F32 && mask->type == GGML_TYPE_F16);
    GGML_ASSERT(K->type == V->type);
    GGML_ASSERT(K->type == GGML_TYPE_Q4_0 || K->type == GGML_TYPE_F16);

    float scale, max_bias, softcap;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    GGML_ASSERT(softcap == 0.0f);

    const int ne02 = (int)Q->ne[2];          // n_heads_q
    const int ne11 = (int)K->ne[1];          // logical KV length (bucketed)
    const int ne12 = (int)K->ne[2];          // n_kv_heads
    const int ne13 = (int)K->ne[3];          // n_seqs
    const int gqa  = ne02 / ne12;
    GGML_ASSERT(gqa >= 1 && gqa <= 8 && ne02 % ne12 == 0);

    const uint32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) ne02));
    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const int n_chunks = (ne11 + PSKV_SPLIT_CHUNK_TOKENS - 1) / PSKV_SPLIT_CHUNK_TOKENS;

    // Scratch from the pool: partial O + (m,l) per (seq, chunk, q-head).
    const size_t n_part = (size_t)ne13 * n_chunks * ne02;
    ggml_cuda_pool_alloc<float>  part_O (ctx.pool(), n_part * 256);
    ggml_cuda_pool_alloc<float2> part_ml(ctx.pool(), n_part);

    const dim3 grid1((unsigned)n_chunks, (unsigned)ne12, (unsigned)ne13);
    const dim3 block1(WARP_SIZE, (unsigned)gqa, 1);
    const dim3 grid2((unsigned)ne02, 1, (unsigned)ne13);
    const dim3 block2(WARP_SIZE, 1, 1);

    const int n_bps = (int) block_table->ne[0];

    auto launch = [&](auto kpartial) {
        kpartial<<<grid1, block1, 0, ctx.stream()>>>(
            (const char *) Q->data, (const char *) K->data, (const char *) V->data,
            (const char *) mask->data,
            (const int *) block_table->data, n_bps,
            part_O.get(), part_ml.get(),
            scale, max_bias, m0, m1, n_head_log2,
            ne02, ne11, ne12,
            (int)mask->nb[1], (int)mask->nb[3],
            (int)Q->nb[1], (int)Q->nb[2], (int)Q->nb[3],
            (int)K->nb[1],
            (int)V->nb[1],
            n_chunks);
    };
    if (K->type == GGML_TYPE_Q4_0) {
        launch(flash_attn_per_slot_kv_split_partial<256, 256, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0>);
    } else {
        launch(flash_attn_per_slot_kv_split_partial<256, 256, GGML_TYPE_F16, GGML_TYPE_F16>);
    }

    flash_attn_per_slot_kv_split_merge<256><<<grid2, block2, 0, ctx.stream()>>>(
        part_O.get(), part_ml.get(), (float *) dst->data,
        (int)Q->ne[1], ne02, n_chunks, n_chunks);

    CUDA_CHECK(cudaGetLastError());
}
