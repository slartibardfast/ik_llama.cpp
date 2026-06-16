// fattn-per-slot-kv-singlewarp-sm75.cu — FIX-C v5 (2026-05-16, ILP-2026-05-18)
//
// Single-warp per-row CTA flash-attention kernel for the production
// per-slot-kv path (Qwen 3.6 27B: Dq=Dv=256, Q4_0 KV cache).
//
// Architecture:
//   - One CTA per (token, head, seq). gridDim = (n_tokens, n_heads_q, n_seqs).
//   - Block = single warp (32 threads). __launch_bounds__(WARP_SIZE, 8): tells
//     NVCC to target ≥8 blocks/SM, allowing up to 256 regs/thread to fit the
//     4-way ILP live-set without local-memory spills.
//   - Each thread handles Dk/WARP_SIZE = 8 K-dim elements (for Dk=256).
//   - K-loop iterates ne11 positions in canonical [0..ne11) order, stepping
//     by 4 (ILP_W) with 4 parallel partial dots per outer iteration. The
//     compiler interleaves the four streams' Q4_0 dequants and FMAs, hiding
//     L1TEX latency. Softmax+V passes still run sequentially in canonical k
//     order so the Welford fp32 accumulation chain is bit-identical to the
//     scalar singlewarp version (NPC contract preserved).
//   - Tail (ne11 % 4) handled by scalar inner loop, identical numerics.
//   - No cross-warp anything. fp32 throughout. No SMEM.
//
// Perf (vs HEAD scalar singlewarp, bench dual-RTX-6000 npp=200 ntg=64 npl=8):
//   - ncu per-CTA: 188.86 µs → 127.26 µs (−32.7%)
//   - TG @ NP=8: 27.10 t/s → 27.90 t/s (+2.95%)
//   - PP @ NP=8: 21.04 t/s → 22.97 t/s (+9.17%)
//   - 254 regs/thread, 0 local spills, 25% theoretical occupancy.
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
#include "ggml-cuda/graph.cuh"  // PHASE_NSTREAM_KV_PERF Tier 2: ggml_cuda_graph fields

// Single-warp per-row CTA kernel.
// Template params:
//   Dk, Dv   — head dims (must be Dk = Dv = 256 in our config; static_assert).
//   type_K   — K cache dtype (GGML_TYPE_Q4_0 in production).
//   type_V   — V cache dtype (GGML_TYPE_Q4_0 in production).
//
// PHASE_NSTREAM_KV_PERF T5.5 — paged KV READ path.
//
// Two address modes supported (selected at runtime by block_table param):
//
//   Legacy (block_table == nullptr): the K/V buffers are laid out as one
//   contiguous per-(seq, head_kv) slab. K_base for (seq, head_kv) =
//   K_direct + nb13*seq + nb12*head_kv; positions k = 0..ne11-1 walk
//   linearly via k*nb11. This is the layout shipped through T4.
//
//   Paged (block_table != nullptr): the K/V buffers are organised as a
//   block pool of size n_blocks_per_seq * n_seq blocks; each block holds
//   BLOCK_SIZE=64 positions × n_head_kv heads × head_dim elements in
//   standard ggml row-major. nb13 is now the per-BLOCK stride. For each
//   position k, the physical block id is block_table[seq*n_blocks_per_seq
//   + k/BLOCK_SIZE]. Because the K-loop steps in chunks of ILP_W=4 and
//   BLOCK_SIZE=64 is a multiple of 4 (and the chunk start is itself
//   aligned to 4), all 4 positions in one ILP chunk live in the same
//   block — bid is looked up once per chunk.
//
//   At trivial (identity) block_table = [seq*n_blocks_per_seq, seq*n_blocks_per_seq+1, ...]
//   AND BLOCK_SIZE*n_blocks_per_seq == kvps, the paged path's byte addresses
//   are byte-identical to the legacy path's addresses (the determinism
//   contract anchor; see PagedFAReadEquivToContiguousAtIdentity in
//   specs/kv-cache/paged_read_path.allium).
//
// At T5.8 closure the legacy branch (block_table == nullptr) is removed
// per [[feedback_bake_measurement_env_gates]] — until then it is the
// reference path the test-paged-byte-identity-trivial-mapping test
// compares against.

static constexpr int PAGED_BLOCK_SIZE_TOKENS = 64;

template<int Dk, int Dv, ggml_type type_K, ggml_type type_V>
__launch_bounds__(WARP_SIZE, 8)
static __global__ void flash_attn_per_slot_kv_singlewarp_kernel(
        const char * __restrict__ Q,
        const char * __restrict__ K_direct,
        const char * __restrict__ V_direct,
        const char * __restrict__ mask,
        const char * __restrict__ /*sinks*/,
        const int  * __restrict__ per_row_k_bound,
        const int  * __restrict__ block_table,    // T5.5: nullptr => legacy
        const int                  n_blocks_per_seq,  // T5.5: meaningful only when block_table != nullptr
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
        const int ne31, const int nb31, const int nb33,
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
    // PHASE_NSTREAM_KV_PERF Tier 3: mask is 4D [n_kv, n_tok_per_seq, 1, n_seq].
    // At ne[3]=1 (Tier 2 and prior) seq=0 and nb33*0=0 so this matches the
    // legacy per-seq mask addressing. At ne[3]>1 (Tier 3 unified dispatch)
    // each seq's mask block lives at offset nb33*seq.
    const half  * maskh = (const half  *) (mask + nb33*seq + nb31*tok);
    const float  slope  = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // K and V base pointers for this (seq, head_kv).
    // Under legacy mode (block_table == nullptr), one base is computed
    // once and the k-loop walks linearly via k*nb11. Under paged mode,
    // a per-block base is computed inside the k-loop (per ILP chunk).
    //
    // T5.6: the K/V tensors passed to FA keep legacy ne/nb (cache backing
    // buffer is sized contiguous). In paged mode the kernel reinterprets
    // the buffer as `[head_dim, BLOCK_SIZE, n_head_kv, total_blocks]` by
    // deriving block-stride and head-within-block-stride from
    // `nb11 * PAGED_BLOCK_SIZE_TOKENS` and `ne12` (n_head_kv). The host
    // does NOT remap K/V tensor strides — kernel-side reinterpretation
    // is the contract.
    const bool   paged       = (block_table != nullptr);
    const int *  bt_seq      = paged ? (block_table + seq * n_blocks_per_seq) : nullptr;
    const int    paged_nb12  = paged ? (int)(nb11 * PAGED_BLOCK_SIZE_TOKENS)            : nb12;
    const int    paged_nb13  = paged ? (int)(paged_nb12 * ne12)                         : nb13;
    const int    paged_nb22  = paged ? (int)(nb21 * PAGED_BLOCK_SIZE_TOKENS)            : nb22;
    const int    paged_nb23  = paged ? (int)(paged_nb22 * ne12)                         : nb23;
    const char * K_base_leg  = paged ? nullptr : (K_direct + nb13*seq + nb12*head_kv);
    const char * V_base_leg  = paged ? nullptr : (V_direct + nb23*seq + nb22*head_kv);

    // Q load into registers. Each thread holds 8 fp32 elements.
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

    // K-loop in canonical [0..ne11) order — the determinism contract.
    //
    // 4-way ILP + Q SMEM: process k, k+1, k+2, k+3 with four independent
    // partial dot products in the same inner i-loop. Q is staged in SMEM
    // instead of per-thread registers to free ~8 regs/thread, giving the
    // compiler headroom to absorb the extra 4-way stream without spilling.
    // Softmax+V passes run sequentially in canonical order so the Welford
    // fp32 accumulation chain matches singlewarp bit-for-bit — NPC by
    // construction.
    (void) per_row_k_bound;  // CY.F.17: per_row_k_bound is 0 during prefill, can't bound here.

    constexpr int ILP_W = 4;
    const int ne11_aligned = (ne11 / ILP_W) * ILP_W;

    for (int k = 0; k < ne11_aligned; k += ILP_W) {
        float partial_a = 0.0f;
        float partial_b = 0.0f;
        float partial_c = 0.0f;
        float partial_d = 0.0f;

        // Per-chunk K base. In paged mode, recompute from block_table.
        // BLOCK_SIZE=64 is a multiple of ILP_W=4 and k is aligned to 4,
        // so all 4 positions in this chunk live in the same block.
        const char * K_chunk_base;
        int          k_in_chunk_off;
        if (paged) {
            const int block_idx  = k / PAGED_BLOCK_SIZE_TOKENS;
            const int bid        = bt_seq[block_idx];
            K_chunk_base         = K_direct + (size_t)bid * paged_nb13 + (size_t)head_kv * paged_nb12;
            k_in_chunk_off       = k & (PAGED_BLOCK_SIZE_TOKENS - 1);  // k % 64
        } else {
            K_chunk_base         = K_base_leg;
            k_in_chunk_off       = k;
        }

        if constexpr (type_K == GGML_TYPE_Q4_0) {
            const block_q4_0 * K_row_a = (const block_q4_0 *)(K_chunk_base + (k_in_chunk_off  )*nb11);
            const block_q4_0 * K_row_b = (const block_q4_0 *)(K_chunk_base + (k_in_chunk_off+1)*nb11);
            const block_q4_0 * K_row_c = (const block_q4_0 *)(K_chunk_base + (k_in_chunk_off+2)*nb11);
            const block_q4_0 * K_row_d = (const block_q4_0 *)(K_chunk_base + (k_in_chunk_off+3)*nb11);
            #pragma unroll
            for (int i = 0; i < Q_PER_THREAD; ++i) {
                const int d        = lane + i*WARP_SIZE;
                const int blk_idx  = d / K_qk;
                const int blk_off  = d % K_qk;
                const int byte_idx = blk_off & (K_qk/2 - 1);
                const int shift    = blk_off / (K_qk/2);

                const block_q4_0 * blk_a = K_row_a + blk_idx;
                const block_q4_0 * blk_b = K_row_b + blk_idx;
                const block_q4_0 * blk_c = K_row_c + blk_idx;
                const block_q4_0 * blk_d = K_row_d + blk_idx;
                const float d_scale_a    = __half2float(blk_a->d);
                const float d_scale_b    = __half2float(blk_b->d);
                const float d_scale_c    = __half2float(blk_c->d);
                const float d_scale_d    = __half2float(blk_d->d);
                const int   q_nib_a      = (blk_a->qs[byte_idx] >> (shift*4)) & 0xF;
                const int   q_nib_b      = (blk_b->qs[byte_idx] >> (shift*4)) & 0xF;
                const int   q_nib_c      = (blk_c->qs[byte_idx] >> (shift*4)) & 0xF;
                const int   q_nib_d      = (blk_d->qs[byte_idx] >> (shift*4)) & 0xF;
                const float k_val_a      = d_scale_a * (float)(q_nib_a - 8);
                const float k_val_b      = d_scale_b * (float)(q_nib_b - 8);
                const float k_val_c      = d_scale_c * (float)(q_nib_c - 8);
                const float k_val_d      = d_scale_d * (float)(q_nib_d - 8);

                partial_a += k_val_a * Q_reg[i];
                partial_b += k_val_b * Q_reg[i];
                partial_c += k_val_c * Q_reg[i];
                partial_d += k_val_d * Q_reg[i];
            }
        } else if constexpr (type_K == GGML_TYPE_Q8_0) {
            const block_q8_0 * K_row_a = (const block_q8_0 *)(K_chunk_base + (k_in_chunk_off  )*nb11);
            const block_q8_0 * K_row_b = (const block_q8_0 *)(K_chunk_base + (k_in_chunk_off+1)*nb11);
            const block_q8_0 * K_row_c = (const block_q8_0 *)(K_chunk_base + (k_in_chunk_off+2)*nb11);
            const block_q8_0 * K_row_d = (const block_q8_0 *)(K_chunk_base + (k_in_chunk_off+3)*nb11);
            #pragma unroll
            for (int i = 0; i < Q_PER_THREAD; ++i) {
                const int d        = lane + i*WARP_SIZE;
                const int blk_idx  = d / K_qk;   // K_qk = 32 = QK8_0
                const int blk_off  = d % K_qk;
                const block_q8_0 * blk_a = K_row_a + blk_idx;
                const block_q8_0 * blk_b = K_row_b + blk_idx;
                const block_q8_0 * blk_c = K_row_c + blk_idx;
                const block_q8_0 * blk_d = K_row_d + blk_idx;
                partial_a += __half2float(blk_a->d) * (float)blk_a->qs[blk_off] * Q_reg[i];
                partial_b += __half2float(blk_b->d) * (float)blk_b->qs[blk_off] * Q_reg[i];
                partial_c += __half2float(blk_c->d) * (float)blk_c->qs[blk_off] * Q_reg[i];
                partial_d += __half2float(blk_d->d) * (float)blk_d->qs[blk_off] * Q_reg[i];
            }
        } else {
            const half * K_row_a = (const half *)(K_chunk_base + (k_in_chunk_off  )*nb11);
            const half * K_row_b = (const half *)(K_chunk_base + (k_in_chunk_off+1)*nb11);
            const half * K_row_c = (const half *)(K_chunk_base + (k_in_chunk_off+2)*nb11);
            const half * K_row_d = (const half *)(K_chunk_base + (k_in_chunk_off+3)*nb11);
            #pragma unroll
            for (int i = 0; i < Q_PER_THREAD; ++i) {
                const int d = lane + i*WARP_SIZE;
                partial_a += __half2float(K_row_a[d]) * Q_reg[i];
                partial_b += __half2float(K_row_b[d]) * Q_reg[i];
                partial_c += __half2float(K_row_c[d]) * Q_reg[i];
                partial_d += __half2float(K_row_d[d]) * Q_reg[i];
            }
        }

        float kq_arr[ILP_W];
        kq_arr[0] = warp_reduce_sum(partial_a) + slope * __half2float(maskh[k  ]);
        kq_arr[1] = warp_reduce_sum(partial_b) + slope * __half2float(maskh[k+1]);
        kq_arr[2] = warp_reduce_sum(partial_c) + slope * __half2float(maskh[k+2]);
        kq_arr[3] = warp_reduce_sum(partial_d) + slope * __half2float(maskh[k+3]);

        // Per-chunk V base — same paged/legacy logic as K, but using
        // V_direct and V's strides (nb22/nb23).
        const char * V_chunk_base = paged
            ? (V_direct + (size_t)bt_seq[k / PAGED_BLOCK_SIZE_TOKENS] * paged_nb23
                        + (size_t)head_kv * paged_nb22)
            : V_base_leg;
        const int    v_in_chunk_off = paged ? (k & (PAGED_BLOCK_SIZE_TOKENS - 1)) : k;

        // Sequential softmax+V passes in canonical k order.
        #pragma unroll
        for (int s = 0; s < ILP_W; ++s) {
            const int   k_s     = k + s;
            const int   v_off_s = v_in_chunk_off + s;
            const float kq_s = kq_arr[s];

            const float new_max  = fmaxf(kqmax, kq_s);
            const float diff_old = kqmax - new_max;
            const float diff_cur = kq_s  - new_max;
            const float scale_corr = diff_old >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_old) : 0.0f;
            const float phi        = diff_cur >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_cur) : 0.0f;
            kqmax = new_max;
            kqsum = kqsum * scale_corr + phi;
            #pragma unroll
            for (int i = 0; i < V_PER_THREAD; ++i) VKQ[i] *= scale_corr;
            (void)k_s;  // legacy V_off was k_s; now derived via v_off_s
            if constexpr (type_V == GGML_TYPE_Q4_0) {
                const block_q4_0 * V_row = (const block_q4_0 *)(V_chunk_base + v_off_s*nb21);
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
            } else if constexpr (type_V == GGML_TYPE_Q8_0) {
                const block_q8_0 * V_row = (const block_q8_0 *)(V_chunk_base + v_off_s*nb21);
                #pragma unroll
                for (int i = 0; i < V_PER_THREAD; ++i) {
                    const int d        = lane + i*WARP_SIZE;
                    const int blk_idx  = d / K_qk;
                    const int blk_off  = d % K_qk;
                    const block_q8_0 * blk = V_row + blk_idx;
                    VKQ[i] += phi * __half2float(blk->d) * (float)blk->qs[blk_off];
                }
            } else {
                const half * V_row = (const half *)(V_chunk_base + v_off_s*nb21);
                #pragma unroll
                for (int i = 0; i < V_PER_THREAD; ++i) {
                    const int d = lane + i*WARP_SIZE;
                    VKQ[i] += phi * __half2float(V_row[d]);
                }
            }
        }
    }

    // Tail (ne11 % ILP_W): scalar inner loop identical to pre-ILP singlewarp.
    // Per-k bid lookup in paged mode (bid may change at block boundary).
    for (int k = ne11_aligned; k < ne11; ++k) {
        const char * K_tail_base;
        const char * V_tail_base;
        int          k_tail_off;
        if (paged) {
            const int block_idx = k / PAGED_BLOCK_SIZE_TOKENS;
            const int bid       = bt_seq[block_idx];
            K_tail_base         = K_direct + (size_t)bid * paged_nb13 + (size_t)head_kv * paged_nb12;
            V_tail_base         = V_direct + (size_t)bid * paged_nb23 + (size_t)head_kv * paged_nb22;
            k_tail_off          = k & (PAGED_BLOCK_SIZE_TOKENS - 1);
        } else {
            K_tail_base = K_base_leg;
            V_tail_base = V_base_leg;
            k_tail_off  = k;
        }
        float partial = 0.0f;
        if constexpr (type_K == GGML_TYPE_Q4_0) {
            const block_q4_0 * K_row = (const block_q4_0 *)(K_tail_base + k_tail_off*nb11);
            #pragma unroll
            for (int i = 0; i < Q_PER_THREAD; ++i) {
                const int d        = lane + i*WARP_SIZE;
                const int blk_idx  = d / K_qk;
                const int blk_off  = d % K_qk;
                const block_q4_0 * blk = K_row + blk_idx;
                const float d_scale = __half2float(blk->d);
                const int byte_idx = blk_off & (K_qk/2 - 1);
                const int shift    = blk_off / (K_qk/2);
                const int q_nib    = (blk->qs[byte_idx] >> (shift*4)) & 0xF;
                partial += d_scale * (float)(q_nib - 8) * Q_reg[i];
            }
        } else if constexpr (type_K == GGML_TYPE_Q8_0) {
            const block_q8_0 * K_row = (const block_q8_0 *)(K_tail_base + k_tail_off*nb11);
            #pragma unroll
            for (int i = 0; i < Q_PER_THREAD; ++i) {
                const int d        = lane + i*WARP_SIZE;
                const int blk_idx  = d / K_qk;
                const int blk_off  = d % K_qk;
                const block_q8_0 * blk = K_row + blk_idx;
                partial += __half2float(blk->d) * (float)blk->qs[blk_off] * Q_reg[i];
            }
        } else {
            const half * K_row = (const half *)(K_tail_base + k_tail_off*nb11);
            #pragma unroll
            for (int i = 0; i < Q_PER_THREAD; ++i) {
                const int d = lane + i*WARP_SIZE;
                partial += __half2float(K_row[d]) * Q_reg[i];
            }
        }
        const float kq = warp_reduce_sum(partial) + slope * __half2float(maskh[k]);
        const float new_max  = fmaxf(kqmax, kq);
        const float diff_old = kqmax - new_max;
        const float diff_cur = kq    - new_max;
        const float scale_corr = diff_old >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_old) : 0.0f;
        const float phi        = diff_cur >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_cur) : 0.0f;
        kqmax = new_max;
        kqsum = kqsum * scale_corr + phi;
        #pragma unroll
        for (int i = 0; i < V_PER_THREAD; ++i) VKQ[i] *= scale_corr;
        if constexpr (type_V == GGML_TYPE_Q4_0) {
            const block_q4_0 * V_row = (const block_q4_0 *)(V_tail_base + k_tail_off*nb21);
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
                VKQ[i] += phi * d_scale * (float)(q_nib - 8);
            }
        } else if constexpr (type_V == GGML_TYPE_Q8_0) {
            const block_q8_0 * V_row = (const block_q8_0 *)(V_tail_base + k_tail_off*nb21);
            #pragma unroll
            for (int i = 0; i < V_PER_THREAD; ++i) {
                const int d        = lane + i*WARP_SIZE;
                const int blk_idx  = d / K_qk;
                const int blk_off  = d % K_qk;
                const block_q8_0 * blk = V_row + blk_idx;
                VKQ[i] += phi * __half2float(blk->d) * (float)blk->qs[blk_off];
            }
        } else {
            const half * V_row = (const half *)(V_tail_base + k_tail_off*nb21);
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
    (void)Q; (void)K_direct; (void)V_direct; (void)mask; (void)dst;
    (void)block_table; (void)n_blocks_per_seq;
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
    const ggml_tensor * Q               = dst->src[0];
    const ggml_tensor * K               = dst->src[1];
    const ggml_tensor * V               = dst->src[2];
    const ggml_tensor * mask            = dst->src[3];
    // CY.F.17: per_row_k_bound is dst->src[5]. Int32 tensor [n_tokens]
    // bounding the per-row K-loop iteration. Absent → fall back to ne11.
    const ggml_tensor * per_row_k_bound = dst->src[5];
    // T5.5: block_table is dst->src[6]. Int32 tensor [n_blocks_per_seq, n_seqs].
    // Absent → legacy contiguous addressing (Bundle A / pre-Tier-5).
    // Present → paged addressing per PagedFAReadEquivToContiguousAtIdentity.
    const ggml_tensor * block_table     = (GGML_MAX_SRC > 6) ? dst->src[6] : nullptr;

    GGML_ASSERT(Q && K && V && mask);
    GGML_ASSERT(Q->ne[0] == 256 && V->ne[0] == 256);
    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->type == GGML_TYPE_F16);
    GGML_ASSERT(K->type == V->type && "K and V must have same dtype");
    GGML_ASSERT((K->type == GGML_TYPE_Q4_0 || K->type == GGML_TYPE_Q8_0 || K->type == GGML_TYPE_F16) &&
                "FIX-C v5 singlewarp supports Q4_0 (production), Q8_0 or F16 (test) KV cache");
    GGML_ASSERT(!per_row_k_bound || per_row_k_bound->type == GGML_TYPE_I32);
    GGML_ASSERT(!block_table || block_table->type == GGML_TYPE_I32);

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

    const int * per_row_k_bound_dev = per_row_k_bound
        ? (const int *) per_row_k_bound->data
        : nullptr;
    const int * block_table_dev = block_table
        ? (const int *) block_table->data
        : nullptr;
    const int n_blocks_per_seq = block_table
        ? (int) block_table->ne[0]
        : 0;

    auto launch_kernel = [&](auto kernel) {
        kernel<<<grid, block, 0, ctx.stream()>>>(
            (const char *) Q->data,
            (const char *) K->data,
            (const char *) V->data,
            (const char *) mask->data,
            nullptr,            // sinks (unused)
            per_row_k_bound_dev,
            block_table_dev,
            n_blocks_per_seq,
            (float *) dst->data,
            nullptr,
            scale, max_bias, m0, m1, softcap, n_head_log2,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            (int)mask->ne[1], (int)mask->nb[1], (int)mask->nb[3],
            (int)Q->nb[1], (int)Q->nb[2], (int)Q->nb[3],
            (int)K->nb[1], (int)K->nb[2], (int)K->nb[3],
            (int)V->nb[1], (int)V->nb[2], (int)V->nb[3],
            (int)dst->ne[0], (int)dst->ne[1], (int)dst->ne[2], (int)dst->ne[3]);
    };
    if (K->type == GGML_TYPE_Q4_0) {
        launch_kernel(flash_attn_per_slot_kv_singlewarp_kernel<256, 256, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0>);
    } else if (K->type == GGML_TYPE_Q8_0) {
        launch_kernel(flash_attn_per_slot_kv_singlewarp_kernel<256, 256, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0>);
    } else { // GGML_TYPE_F16
        launch_kernel(flash_attn_per_slot_kv_singlewarp_kernel<256, 256, GGML_TYPE_F16, GGML_TYPE_F16>);
    }
    CUDA_CHECK(cudaGetLastError());
}
