// fattn-per-slot-kv-sm75.cu
//
// SoTA sm_75 batch-invariant flash-attention replacement for
// ggml_cuda_flash_attn_ext_wmma_f16. Targets the Qwen 3.5/3.6 production
// tuple (HEAD_DIM_Q = HEAD_DIM_V = 256). See
// specs/deltanet/fattn-per-slot-kv-sm75.md for the full design.
//
// ============================================================================
// PHASE 1 (this file): naïve scalar device kernel.
// ============================================================================
// One CTA per (slot, q_row, head); 32 threads (1 warp) per CTA;
// warp-shuffle reduction for the Q•K dot product; per-thread accumulation
// of HEAD_DIM_V / 32 = 8 output dimensions (strided). KV_BLOCK_SIZE-chunked
// K loop mirrors the host-side oracle in
// tests/dflash-speculative/fattn-per-slot-kv-sm75-reference.h step-by-step.
// fp32 throughout; cast to fp16 at output via the test's float buffer (no
// cast happens here — output is fp32).
//
// Closure binding (per test scenarios A/B/C):
//   A oracle_vs_kernel:  cosine ≥ 0.9999 AND nmse ≤ 1e-4
//   B rep_determinism:   byte-identical across reps (kernel must be
//                         deterministic on identical inputs)
//   C batch_invariance:  byte-identical slot-0 output across NP
//
// Phase 1 does NOT exercise split-K via parallel_blocks. Per CTA processes
// the full per-slot K range. dst_partial / dst_meta are not produced; the
// kernel writes the final combined output directly into out_final. This is
// a mathematically equivalent collapse of the oracle's
// reference_one_cta + combine step when pb=1.
//
// ============================================================================
// PHASE 2 (TODO): SoTA path per spec §2.
// ============================================================================
//   - Block 128 threads = 4 warps
//   - mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 PTX
//   - ldmatrix.sync.aligned.x4.m8n8.shared.b16 for fragment loads
//   - SMEM swizzle for K, V tiles
//   - parallel_blocks split per-slot; combine via existing
//     flash_attn_combine_results
//   - Target ≤ 64 regs/thread, 2 blocks/SM @ 25 KiB SMEM
//
// ============================================================================
// ABI note:
// ============================================================================
// The test driver declares the launcher with a host-side AttnConfig struct
// in namespace `fattn_per_slot_kv_sm75`. We define an ABI-matching
// `LauncherConfig` struct (same field order, same types) in this TU. The
// extern "C" symbol strips namespaces so the binding is by symbol name +
// pointer-level ABI. **Any change to the test's AttnConfig layout must be
// mirrored here.**
//
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "mma_new.cuh"  // ggml_cuda_mma::tile + Turing mma.sync.m16n8k8 wrappers

// ABI-mirror of `tests/dflash-speculative/fattn-per-slot-kv-sm75-reference.h`
// struct AttnConfig (in namespace fattn_per_slot_kv_sm75). Order and types
// must match exactly.
struct LauncherConfig {
    int   head_dim_q;
    int   head_dim_v;
    int   kv_block_size;
    int   n_tokens;
    int   n_heads_q;
    int   n_kv_heads;
    int   n_seqs;
    int   n_kv_max;
    int   parallel_blocks;
    float scale;
    float softcap;
    bool  use_softcap;
};

// Host Half = struct { uint16_t bits; }. ABI-compatible with uint16_t* for
// trivial wrappers.
struct HostHalf { uint16_t bits; };

// ============================================================================
// Device kernel — naïve scalar path.
// ============================================================================
//
// Layout (matches oracle):
//   Q:             [HEAD_DIM_Q, n_tokens, n_heads_q, n_seqs]   column-major
//   K_cache:       [HEAD_DIM_Q, n_kv_max, n_kv_heads, n_seqs]
//   V_cache:       [HEAD_DIM_V, n_kv_max, n_kv_heads, n_seqs]
//   mask:          [n_kv_max, n_tokens]
//   slot_seq_lens: [n_seqs]
//   out_final:     [HEAD_DIM_V, n_tokens, n_heads_q, n_seqs]   fp32
//
// Grid:  (n_seqs, n_tokens, n_heads_q)
// Block: (32, 1, 1)  — 1 warp per CTA
//
// Per-CTA work: ONE (slot, q_row, head) tuple. Full K range [0, slot_seq_len).

__device__ __forceinline__ float half_bits_to_float(uint16_t h) {
    __half hv = __ushort_as_half(h);
    return __half2float(hv);
}

__global__ void fattn_per_slot_kv_sm75_naive_kernel(
    const uint16_t * __restrict__ Q,
    const uint16_t * __restrict__ K,
    const uint16_t * __restrict__ V,
    const uint16_t * __restrict__ mask,
    const int32_t  * __restrict__ slot_seq_lens,
    float          * __restrict__ out_final,
    int Dq, int Dv, int KVB,
    int n_tokens, int n_heads_q, int n_kv_heads, int n_seqs, int n_kv_max,
    float scale, float softcap, bool use_softcap)
{
    const int slot  = blockIdx.x;
    const int q_row = blockIdx.y;
    const int head  = blockIdx.z;
    const int tid   = threadIdx.x;  // 0..31

    if (slot >= n_seqs || q_row >= n_tokens || head >= n_heads_q) return;

    const int gqa     = n_heads_q / n_kv_heads;
    const int kv_head = head / gqa;
    const int n_kv    = slot_seq_lens[slot];

    // Offset functions (mirror reference.h exactly).
    auto Q_off = [=] __device__ (int d) -> size_t {
        return (((size_t)slot * n_heads_q + head) * n_tokens + q_row) * Dq + d;
    };
    auto K_off = [=] __device__ (int d, int k) -> size_t {
        return (((size_t)slot * n_kv_heads + kv_head) * n_kv_max + k) * Dq + d;
    };
    auto V_off = [=] __device__ (int d, int k) -> size_t {
        return (((size_t)slot * n_kv_heads + kv_head) * n_kv_max + k) * Dv + d;
    };
    auto M_off = [=] __device__ (int k) -> size_t {
        return (size_t)q_row * n_kv_max + k;
    };
    auto O_off = [=] __device__ (int d) -> size_t {
        return (((size_t)slot * n_heads_q + head) * n_tokens + q_row) * Dv + d;
    };

    // Per-thread VKQ accumulator: handles 8 output dims (Dv=256 / 32 threads = 8).
    // Index pattern: thread tid handles dims {tid, tid+32, tid+64, ..., tid+224}.
    // (Coalesces V loads at fp16 stride within a warp.)
    // We allow up to 16 dims/thread for Dv up to 512 (future-proofing).
    constexpr int MAX_DV_PER_THREAD = 16;
    float VKQ[MAX_DV_PER_THREAD] = {0};
    const int dv_per_thread = Dv / 32;

    float kqmax    = -INFINITY;
    float kqrowsum = 0.0f;

    if (n_kv <= 0) {
        // Empty slot — write zeros.
        for (int i = 0; i < dv_per_thread; i++) {
            int d = tid + i * 32;
            if (d < Dv) out_final[O_off(d)] = 0.0f;
        }
        return;
    }

    // K-loop in KV_BLOCK_SIZE-sized chunks. Each chunk produces KQ[k - kb] for
    // k in [kb, kb_end), then updates online softmax + accumulates V.
    constexpr int MAX_KQ = 64;  // upper bound for KV_BLOCK_SIZE ∈ {16, 32, 64}
    float KQ[MAX_KQ];

    for (int kb = 0; kb < n_kv; kb += KVB) {
        const int kb_end = min(kb + KVB, n_kv);
        const int blk    = kb_end - kb;

        // Compute KQ[i] = scale * Q • K[i+kb]  via per-thread partial sum +
        // warp-shuffle reduction. All 32 threads in the warp end with the same
        // KQ[i] value.
        for (int i = 0; i < blk; i++) {
            const int k = kb + i;
            float acc = 0.0f;
            for (int j = 0; j < dv_per_thread; j++) {  // Dq == Dv at our shape
                const int d = tid + j * 32;
                if (d < Dq) {
                    acc += half_bits_to_float(Q[Q_off(d)]) *
                           half_bits_to_float(K[K_off(d, k)]);
                }
            }
            // Warp-shuffle reduction (pairwise tree, deterministic).
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc += __shfl_xor_sync(0xFFFFFFFF, acc, offset);
            }
            acc *= scale;
            if (use_softcap) {
                acc = softcap * tanhf(acc / softcap);
            }
            acc += half_bits_to_float(mask[M_off(k)]);
            KQ[i] = acc;
        }

        // Find new max over this chunk + running max.
        float new_max = kqmax;
        for (int i = 0; i < blk; i++) {
            if (KQ[i] > new_max) new_max = KQ[i];
        }
        if (new_max == -INFINITY) {
            // All-masked chunk — nothing to add.
            continue;
        }

        // Scale running state by exp(old_max - new_max).
        const float scale_factor =
            (kqmax == -INFINITY) ? 0.0f : __expf(kqmax - new_max);
        kqrowsum *= scale_factor;
        for (int i = 0; i < dv_per_thread; i++) {
            VKQ[i] *= scale_factor;
        }
        kqmax = new_max;

        // Softmax weights + V accumulation.
        for (int i = 0; i < blk; i++) {
            const int k  = kb + i;
            const float sm = __expf(KQ[i] - new_max);
            kqrowsum += sm;
            if (sm == 0.0f) continue;
            for (int j = 0; j < dv_per_thread; j++) {
                const int d = tid + j * 32;
                if (d < Dv) {
                    VKQ[j] += sm * half_bits_to_float(V[V_off(d, k)]);
                }
            }
        }
    }

    // Normalize and write output. Match the oracle's denominator: `num / den`
    // where den is the combined rowsum. For pb=1, den == kqrowsum directly.
    const float den = kqrowsum;
    for (int j = 0; j < dv_per_thread; j++) {
        const int d = tid + j * 32;
        if (d < Dv) {
            const float val = (den > 0.0f) ? (VKQ[j] / den) : 0.0f;
            out_final[O_off(d)] = val;
        }
    }
}

// ============================================================================
// Stage 2.1 kernel — replace inner Q·K dot product with mma.sync.m16n8k8 PTX.
// ============================================================================
//
// Same grid (n_seqs, n_tokens, n_heads_q) and block (32 threads = 1 warp)
// as Phase 1. Same algorithm: per-K-block softmax with fp32 accumulators,
// per-thread VKQ accumulation, single-pass (no parallel_blocks split).
//
// ONLY DIFFERENCE FROM PHASE 1: the per-K-block Q·K dot product is computed
// via the `ggml_cuda_mma::mma()` wrapper around mma.sync.m16n8k8 (k=8 fp16
// inputs, fp32 accumulator on Turing — see mma_new.cuh:334-358).
//
// At sm_75 the wrapper internally decomposes one m16n8k16 logical call into
// two m16n8k8 PTX instructions; this is the spec §2 SoTA tensor-core path.
//
// Q TILE PADDING. The mma's m=16 forces 16 query rows per call. We have one
// real query row at this CTA's (slot, q_row, head) tuple. Rows 1-15 of the
// A tile are filled with zero; rows 1-15 of the mma output are discarded.
// 15/16 of mma compute is throwaway at this stage. Stage 2.2 fixes via
// Approach C multi-head packing.
//
// COSINE BIND. mma's internal reduction order (16 k-products summed inside
// each instruction; accumulated across k-iterations into D) differs from
// Phase 1's per-thread partial + warp-shuffle reduction. So Stage 2.1 is
// NOT byte-identical to Phase 1; but cosine ≥ 0.9999 and NMSE ≤ 1e-4
// because all arithmetic is fp32 at the same logical precision.

__global__ void fattn_per_slot_kv_sm75_stage21_kernel(
    const uint16_t * __restrict__ Q,
    const uint16_t * __restrict__ K,
    const uint16_t * __restrict__ V,
    const uint16_t * __restrict__ mask,
    const int32_t  * __restrict__ slot_seq_lens,
    float          * __restrict__ out_final,
    int Dq, int Dv, int KVB,
    int n_tokens, int n_heads_q, int n_kv_heads, int n_seqs, int n_kv_max,
    float scale, float softcap, bool use_softcap)
{
    using namespace ggml_cuda_mma;

    const int slot  = blockIdx.x;
    const int q_row = blockIdx.y;
    const int head  = blockIdx.z;
    const int tid   = threadIdx.x;

    if (slot >= n_seqs || q_row >= n_tokens || head >= n_heads_q) return;

    const int gqa     = n_heads_q / n_kv_heads;
    const int kv_head = head / gqa;
    const int n_kv    = slot_seq_lens[slot];

    auto Q_off = [=] __device__ (int d) -> size_t {
        return (((size_t)slot * n_heads_q + head) * n_tokens + q_row) * Dq + d;
    };
    auto K_off = [=] __device__ (int d, int k) -> size_t {
        return (((size_t)slot * n_kv_heads + kv_head) * n_kv_max + k) * Dq + d;
    };
    auto V_off = [=] __device__ (int d, int k) -> size_t {
        return (((size_t)slot * n_kv_heads + kv_head) * n_kv_max + k) * Dv + d;
    };
    auto M_off = [=] __device__ (int k) -> size_t {
        return (size_t)q_row * n_kv_max + k;
    };
    auto O_off = [=] __device__ (int d) -> size_t {
        return (((size_t)slot * n_heads_q + head) * n_tokens + q_row) * Dv + d;
    };

    constexpr int MAX_DV_PER_THREAD = 16;
    float VKQ[MAX_DV_PER_THREAD] = {0};
    const int dv_per_thread = Dv / 32;

    float kqmax    = -INFINITY;
    float kqrowsum = 0.0f;

    if (n_kv <= 0) {
        for (int i = 0; i < dv_per_thread; i++) {
            int d = tid + i * 32;
            if (d < Dv) out_final[O_off(d)] = 0.0f;
        }
        return;
    }

    // Shared memory staging for KQ extraction (top row of D per n-tile).
    // KV_BLOCK_SIZE ≤ 32 in our spec, so 32 floats is the upper bound.
    __shared__ float KQ_smem[32];

    for (int kb = 0; kb < n_kv; kb += KVB) {
        const int kb_end = min(kb + KVB, n_kv);
        const int blk    = kb_end - kb;

        // Compute KQ[0..blk-1] via mma. Each n-tile covers 8 K positions.
        for (int n_tile = 0; n_tile * 8 < blk; n_tile++) {
            const int n_start = n_tile * 8;
            const int n_count = min(8, blk - n_start);  // valid K positions in this n-tile

            tile<16, 8, float> D;
            #pragma unroll
            for (int l = 0; l < D.ne; l++) D.x[l] = 0.0f;

            // Iterate over k-chunks of 16 fp16 features. Dq=256 → 16 chunks.
            for (int k_chunk = 0; k_chunk < Dq / 16; k_chunk++) {
                const int k_start = k_chunk * 16;

                // Build A tile (Q, only row 0 real).
                tile<16, 8, half2> A;
                #pragma unroll
                for (int l = 0; l < A.ne; l++) {
                    const int row       = A.get_i(l);
                    const int half2_col = A.get_j(l);
                    half a, b;
                    if (row == 0) {
                        const int d0 = k_start + 2 * half2_col;
                        const int d1 = d0 + 1;
                        a = __ushort_as_half(Q[Q_off(d0)]);
                        b = __ushort_as_half(Q[Q_off(d1)]);
                    } else {
                        a = __float2half(0.0f);
                        b = __float2half(0.0f);
                    }
                    A.x[l] = __halves2half2(a, b);
                }

                // Build B tile (K rows of n_tile's 8 K positions × 16 features).
                tile<8, 8, half2> B;
                #pragma unroll
                for (int l = 0; l < B.ne; l++) {
                    const int n_idx     = B.get_i(l);    // N-axis: K position 0..7 within n-tile
                    const int half2_col = B.get_j(l);    // K-axis: 16-feature half2 pair 0..7
                    const int k_global  = kb + n_start + n_idx;
                    half a, b;
                    if (n_idx < n_count) {
                        const int d0 = k_start + 2 * half2_col;
                        const int d1 = d0 + 1;
                        a = __ushort_as_half(K[K_off(d0, k_global)]);
                        b = __ushort_as_half(K[K_off(d1, k_global)]);
                    } else {
                        // K position beyond valid range — zero so this column
                        // contributes nothing to D. Mask handling below also
                        // skips invalid positions via -INFINITY in KQ.
                        a = __float2half(0.0f);
                        b = __float2half(0.0f);
                    }
                    B.x[l] = __halves2half2(a, b);
                }

                mma(D, A, B);
            }

            // Extract top row of D (= real query × 8 K positions for this n-tile).
            // D layout for tile<16, 8, float>: thread tx holds D[tx/4][2*(tx%4)]
            // in D.x[0] and D[tx/4][2*(tx%4)+1] in D.x[1]. Top row is held by
            // threads tx ∈ {0, 1, 2, 3}.
            __syncwarp();
            if (tid < 4) {
                KQ_smem[n_start + 2 * tid    ] = D.x[0];
                KQ_smem[n_start + 2 * tid + 1] = D.x[1];
            }
            __syncwarp();
        }

        // Apply scale, softcap, mask → KQ[i]. Same as Phase 1 from here.
        float KQ[64];
        for (int i = 0; i < blk; i++) {
            float acc = KQ_smem[i] * scale;
            if (use_softcap) {
                acc = softcap * tanhf(acc / softcap);
            }
            acc += half_bits_to_float(mask[M_off(kb + i)]);
            KQ[i] = acc;
        }

        // Find new max.
        float new_max = kqmax;
        for (int i = 0; i < blk; i++) {
            if (KQ[i] > new_max) new_max = KQ[i];
        }
        if (new_max == -INFINITY) continue;

        // Scale running state by exp(old_max - new_max).
        const float scale_factor =
            (kqmax == -INFINITY) ? 0.0f : __expf(kqmax - new_max);
        kqrowsum *= scale_factor;
        for (int i = 0; i < dv_per_thread; i++) {
            VKQ[i] *= scale_factor;
        }
        kqmax = new_max;

        // Softmax + V accumulation (same as Phase 1).
        for (int i = 0; i < blk; i++) {
            const int k    = kb + i;
            const float sm = __expf(KQ[i] - new_max);
            kqrowsum += sm;
            if (sm == 0.0f) continue;
            for (int j = 0; j < dv_per_thread; j++) {
                const int d = tid + j * 32;
                if (d < Dv) {
                    VKQ[j] += sm * half_bits_to_float(V[V_off(d, k)]);
                }
            }
        }
    }

    // Normalize and write output.
    const float den = kqrowsum;
    for (int j = 0; j < dv_per_thread; j++) {
        const int d = tid + j * 32;
        if (d < Dv) {
            const float val = (den > 0.0f) ? (VKQ[j] / den) : 0.0f;
            out_final[O_off(d)] = val;
        }
    }
}

// ============================================================================
// Stage 2.2a kernel — 4-warp CTA (128 threads), same Approach A geometry.
// ============================================================================
//
// Sub-stage of Stage 2.2 per Q&A decision (7b incremental). ONLY change from
// Stage 2.1: scale from 1 warp to 4 warps per CTA. Approach C multi-head
// packing remains deferred to 2.2b; Q/K/V SMEM-staging deferred to 2.2c/2.2d.
//
// WORK PARTITION (4 warps × 32 threads = 128 threads):
//   Each warp handles 4 of the 16 k-chunks (Dq/16 = 16) per n-tile.
//   Warp w computes mma over k-chunks {w*4, w*4+1, w*4+2, w*4+3}.
//   Partial D per warp = 16×8 fp32. After all 4 warps finish, sum
//   partial D's via SMEM to get full D.
//
//   KQ extraction: warp 0 reads summed D's top row from SMEM into the
//   shared KQ array. All warps see the same KQ via SMEM.
//
//   Online softmax: all warps independently update kqmax / kqrowsum
//   (identical state because identical inputs).
//
//   V accumulation: each warp owns 1/4 of V dims (Dv/4 = 64 dims).
//   Within a warp, 32 threads × 2 fp32/thread = 64 dims.
//   Each warp writes its slice of out_final directly (no cross-warp combine).
//
// COSINE BIND. Same numerical class as Stage 2.1 — all fp32 accumulators,
// mma.sync internal reduction. Cross-warp SMEM sum introduces another
// associative-add ordering, but cosine ≥ 0.9999 holds at fp32 precision.

__global__ void fattn_per_slot_kv_sm75_stage22a_kernel(
    const uint16_t * __restrict__ Q,
    const uint16_t * __restrict__ K,
    const uint16_t * __restrict__ V,
    const uint16_t * __restrict__ mask,
    const int32_t  * __restrict__ slot_seq_lens,
    float          * __restrict__ out_final,
    int Dq, int Dv, int KVB,
    int n_tokens, int n_heads_q, int n_kv_heads, int n_seqs, int n_kv_max,
    float scale, float softcap, bool use_softcap)
{
    using namespace ggml_cuda_mma;

    constexpr int NWARPS = 4;

    const int slot    = blockIdx.x;
    const int q_row   = blockIdx.y;
    const int head    = blockIdx.z;
    const int lane    = threadIdx.x;       // 0..31 (within warp)
    const int warp_id = threadIdx.y;       // 0..3
    const int tid     = warp_id * 32 + lane;  // 0..127 (CTA-wide)

    if (slot >= n_seqs || q_row >= n_tokens || head >= n_heads_q) return;

    const int gqa     = n_heads_q / n_kv_heads;
    const int kv_head = head / gqa;
    const int n_kv    = slot_seq_lens[slot];

    auto Q_off = [=] __device__ (int d) -> size_t {
        return (((size_t)slot * n_heads_q + head) * n_tokens + q_row) * Dq + d;
    };
    auto K_off = [=] __device__ (int d, int k) -> size_t {
        return (((size_t)slot * n_kv_heads + kv_head) * n_kv_max + k) * Dq + d;
    };
    auto V_off = [=] __device__ (int d, int k) -> size_t {
        return (((size_t)slot * n_kv_heads + kv_head) * n_kv_max + k) * Dv + d;
    };
    auto M_off = [=] __device__ (int k) -> size_t {
        return (size_t)q_row * n_kv_max + k;
    };
    auto O_off = [=] __device__ (int d) -> size_t {
        return (((size_t)slot * n_heads_q + head) * n_tokens + q_row) * Dv + d;
    };

    // Each warp owns Dv/4 = 64 V dims. Within a warp, 32 threads × 2 fp32 = 64 dims.
    // Dim index for thread tid: warp_id * (Dv/NWARPS) + lane * (Dv/(NWARPS*32))
    constexpr int VKQ_PER_THREAD = 2;  // Dv=256 / 128 = 2
    const int dv_base = warp_id * (Dv / NWARPS);  // 0, 64, 128, 192
    float VKQ[VKQ_PER_THREAD] = {0};

    float kqmax    = -INFINITY;
    float kqrowsum = 0.0f;

    if (n_kv <= 0) {
        for (int i = 0; i < VKQ_PER_THREAD; i++) {
            int d = dv_base + lane + i * 32;
            if (d < Dv) out_final[O_off(d)] = 0.0f;
        }
        return;
    }

    // Shared memory:
    //   D_partials[NWARPS][16][8] fp32 — partial D per warp. Sized at compile.
    //   D_summed[16][8]                — summed D after cross-warp reduction.
    //                                    Reuse D_partials[0] to save SMEM.
    //   KQ_smem[32]                    — extracted top-row KQ values.
    // Total: NWARPS * 16 * 8 * 4 + 32 * 4 = 2048 + 128 = 2176 bytes. Tiny.
    __shared__ float D_partials[NWARPS][16][8];
    __shared__ float KQ_smem[32];

    for (int kb = 0; kb < n_kv; kb += KVB) {
        const int kb_end = min(kb + KVB, n_kv);
        const int blk    = kb_end - kb;

        // ----- KQ computation via mma -----
        // Each warp handles its k-chunks (4 of 16) per n-tile.
        // n_tile iterates 0..(blk-1)/8 — at most 2 for KVB ≤ 16.
        for (int n_tile = 0; n_tile * 8 < blk; n_tile++) {
            const int n_start = n_tile * 8;
            const int n_count = min(8, blk - n_start);

            tile<16, 8, float> D;
            #pragma unroll
            for (int l = 0; l < D.ne; l++) D.x[l] = 0.0f;

            // This warp handles k-chunks {warp_id*4, warp_id*4+1, ..., warp_id*4+3}.
            // 4 k-chunks × 16 features = 64 features per warp. Total across warps:
            // 4 × 64 = 256 = Dq. ✓
            const int k_chunks_per_warp = (Dq / 16) / NWARPS;  // 16/4 = 4
            const int k_chunk_start = warp_id * k_chunks_per_warp;
            const int k_chunk_end   = k_chunk_start + k_chunks_per_warp;

            for (int k_chunk = k_chunk_start; k_chunk < k_chunk_end; k_chunk++) {
                const int k_start = k_chunk * 16;

                tile<16, 8, half2> A;
                #pragma unroll
                for (int l = 0; l < A.ne; l++) {
                    const int row       = A.get_i(l);
                    const int half2_col = A.get_j(l);
                    half a, b;
                    if (row == 0) {
                        const int d0 = k_start + 2 * half2_col;
                        const int d1 = d0 + 1;
                        a = __ushort_as_half(Q[Q_off(d0)]);
                        b = __ushort_as_half(Q[Q_off(d1)]);
                    } else {
                        a = __float2half(0.0f);
                        b = __float2half(0.0f);
                    }
                    A.x[l] = __halves2half2(a, b);
                }

                tile<8, 8, half2> B;
                #pragma unroll
                for (int l = 0; l < B.ne; l++) {
                    const int n_idx     = B.get_i(l);
                    const int half2_col = B.get_j(l);
                    const int k_global  = kb + n_start + n_idx;
                    half a, b;
                    if (n_idx < n_count) {
                        const int d0 = k_start + 2 * half2_col;
                        const int d1 = d0 + 1;
                        a = __ushort_as_half(K[K_off(d0, k_global)]);
                        b = __ushort_as_half(K[K_off(d1, k_global)]);
                    } else {
                        a = __float2half(0.0f);
                        b = __float2half(0.0f);
                    }
                    B.x[l] = __halves2half2(a, b);
                }

                mma(D, A, B);
            }

            // Write this warp's partial D to SMEM at known thread→(row,col) layout.
            // For tile<16, 8, float>: thread tx holds
            //   D.x[0] at D[tx/4][2*(tx%4)]
            //   D.x[1] at D[tx/4][2*(tx%4)+1]
            //   D.x[2] at D[8+tx/4][2*(tx%4)]
            //   D.x[3] at D[8+tx/4][2*(tx%4)+1]
            {
                const int row_lo = lane / 4;
                const int row_hi = 8 + row_lo;
                const int col_lo = 2 * (lane % 4);
                D_partials[warp_id][row_lo][col_lo    ] = D.x[0];
                D_partials[warp_id][row_lo][col_lo + 1] = D.x[1];
                D_partials[warp_id][row_hi][col_lo    ] = D.x[2];
                D_partials[warp_id][row_hi][col_lo + 1] = D.x[3];
            }
            __syncthreads();

            // Sum partials across warps + extract top row to KQ_smem.
            // Top row of D = D[0][col] for col in 0..7. Map 8 cols to 8 threads.
            if (tid < 8) {
                float sum = 0.0f;
                #pragma unroll
                for (int w = 0; w < NWARPS; w++) {
                    sum += D_partials[w][0][tid];
                }
                KQ_smem[n_start + tid] = sum;
            }
            __syncthreads();
        }

        // ----- Apply scale + softcap + mask -----
        float KQ[64];
        for (int i = 0; i < blk; i++) {
            float acc = KQ_smem[i] * scale;
            if (use_softcap) {
                acc = softcap * tanhf(acc / softcap);
            }
            acc += half_bits_to_float(mask[M_off(kb + i)]);
            KQ[i] = acc;
        }

        // ----- Update online softmax state (identical across warps) -----
        float new_max = kqmax;
        for (int i = 0; i < blk; i++) {
            if (KQ[i] > new_max) new_max = KQ[i];
        }
        if (new_max == -INFINITY) continue;

        const float scale_factor =
            (kqmax == -INFINITY) ? 0.0f : __expf(kqmax - new_max);
        kqrowsum *= scale_factor;
        for (int i = 0; i < VKQ_PER_THREAD; i++) {
            VKQ[i] *= scale_factor;
        }
        kqmax = new_max;

        // ----- V accumulation (this warp's V-dim slice) -----
        for (int i = 0; i < blk; i++) {
            const int k    = kb + i;
            const float sm = __expf(KQ[i] - new_max);
            kqrowsum += sm;
            if (sm == 0.0f) continue;
            for (int j = 0; j < VKQ_PER_THREAD; j++) {
                const int d = dv_base + lane + j * 32;
                if (d < Dv) {
                    VKQ[j] += sm * half_bits_to_float(V[V_off(d, k)]);
                }
            }
        }
    }

    // ----- Normalize + write output (this warp's V-dim slice) -----
    const float den = kqrowsum;
    for (int j = 0; j < VKQ_PER_THREAD; j++) {
        const int d = dv_base + lane + j * 32;
        if (d < Dv) {
            const float val = (den > 0.0f) ? (VKQ[j] / den) : 0.0f;
            out_final[O_off(d)] = val;
        }
    }
}

// ============================================================================
// Stage 2.2b kernel — Approach C multi-head packing (decode case).
// ============================================================================
//
// Per spec §1 / §4 (corrected). At n_tokens=1 (decode), pack H=gqa query
// heads (one q_row each) into a single CTA's 16-row mma tile. Rows
// 0..H-1 are real heads; rows H..15 are zero-padded.
//
// At gqa=6 (production target Qwen 3.6 27B): 6 real rows + 10 padded.
// Useful mma work: 6/16 = 37.5% (vs Stage 2.2a's 1/16 = 6.25% at decode).
// 6x effective mma-throughput improvement at decode shape.
//
// SCOPE: this kernel handles ONLY n_tokens=1 with H_packed <= 16 packed
// heads (one CTA per (slot, kv_head)). For n_tokens > 1 or H_packed > 16,
// the launcher falls through to Stage 2.2a (Approach A).
//
// GRID & BLOCK:
//   grid (n_seqs, n_kv_heads, 1) — was (n_seqs, n_tokens, n_heads_q)
//   block (32, 4, 1) — 4 warps = 128 threads
//
// PER-CTA WORK:
//   1 (slot, kv_head) tuple. All H heads of the gqa group processed
//   together. Single q_row = 0 (decode).
//   K-cache and V-cache are SHARED across all H heads in this gqa group
//   (one K head feeds all H Q heads) — no extra K/V loading vs Stage 2.2a.
//
// VKQ DISTRIBUTION:
//   16 rows × Dv=256 dims = 4096 fp32 per CTA = 32 fp32/thread (128 threads).
//   Layout: warp w owns rows [w*4, w*4+4); within a warp, 32 threads cover
//   256 V dims via strided access (lane + d_offset*32 → coalesced loads).
//   Per-thread VKQ: 4 rows × 8 dims = 32 fp32.
//
// SOFTMAX STATE: per-row (16 rows). Stored in SMEM (kqmax_smem,
// kqrowsum_smem arrays of 16 fp32 each). All threads read; warp 0
// (or any chosen warp) writes via a per-row reduce.

__global__ void fattn_per_slot_kv_sm75_stage22b_kernel(
    const uint16_t * __restrict__ Q,
    const uint16_t * __restrict__ K,
    const uint16_t * __restrict__ V,
    const uint16_t * __restrict__ mask,
    const int32_t  * __restrict__ slot_seq_lens,
    float          * __restrict__ out_final,
    int Dq, int Dv, int KVB,
    int n_heads_q, int n_kv_heads, int n_seqs, int n_kv_max,
    int H_packed,                  // number of real query heads packed (<= 16)
    float scale, float softcap, bool use_softcap)
{
    using namespace ggml_cuda_mma;

    constexpr int NWARPS = 4;
    constexpr int N_ROWS = 16;     // mma m-axis
    constexpr int ROWS_PER_WARP = N_ROWS / NWARPS;  // 4
    constexpr int VKQ_PER_THREAD = 32;  // ROWS_PER_WARP * (Dv / 32) = 4 * 8 = 32

    const int slot    = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int tid     = warp_id * 32 + lane;

    if (slot >= n_seqs || kv_head >= n_kv_heads) return;

    const int gqa        = n_heads_q / n_kv_heads;
    const int head_start = kv_head * gqa;        // first Q head in this gqa group
    const int n_kv       = slot_seq_lens[slot];

    // Q[head, q_row=0] offset per row r in the CTA's mma tile.
    auto Q_off_row = [=] __device__ (int r, int d) -> size_t {
        const int head_global = head_start + r;
        // q_row = 0 always (decode case)
        return (((size_t)slot * n_heads_q + head_global) * /*n_tokens=*/1 + 0) * Dq + d;
    };
    auto K_off = [=] __device__ (int d, int k) -> size_t {
        return (((size_t)slot * n_kv_heads + kv_head) * n_kv_max + k) * Dq + d;
    };
    auto V_off = [=] __device__ (int d, int k) -> size_t {
        return (((size_t)slot * n_kv_heads + kv_head) * n_kv_max + k) * Dv + d;
    };
    auto M_off = [=] __device__ (int k) -> size_t {
        // mask[k, q_row=0]
        return (size_t)0 * n_kv_max + k;
    };
    auto O_off = [=] __device__ (int r, int d) -> size_t {
        const int head_global = head_start + r;
        return (((size_t)slot * n_heads_q + head_global) * /*n_tokens=*/1 + 0) * Dv + d;
    };

    // Per-thread VKQ: 4 rows × 8 dims (Dv/32) = 32 fp32.
    // Index: VKQ[r_in_warp * 8 + d_offset] where r_in_warp ∈ [0,4), d_offset ∈ [0,8).
    float VKQ[VKQ_PER_THREAD] = {0};

    // Per-row softmax state in SMEM (16 rows × 2 fp32 = 128 bytes).
    __shared__ float kqmax_smem[N_ROWS];
    __shared__ float kqrowsum_smem[N_ROWS];

    // Init per-row state.
    if (tid < N_ROWS) {
        kqmax_smem[tid]    = -INFINITY;
        kqrowsum_smem[tid] = 0.0f;
    }
    __syncthreads();

    if (n_kv <= 0) {
        for (int r_in_warp = 0; r_in_warp < ROWS_PER_WARP; r_in_warp++) {
            const int r = warp_id * ROWS_PER_WARP + r_in_warp;
            if (r >= H_packed) continue;
            for (int d_offset = 0; d_offset < 8; d_offset++) {
                const int d = lane + d_offset * 32;
                if (d < Dv) out_final[O_off(r, d)] = 0.0f;
            }
        }
        return;
    }

    // SMEM (per CTA):
    //   Q_smem[16][256]           = 8192 B for Q tile, loaded once per CTA
    //   K_smem[32][256]           = 16384 B for K block, loaded per K-iteration
    //   V_smem[32][256]           = 16384 B for V block, loaded per K-iteration
    //   D_partials[NWARPS][16][8] = 2048 B for mma cross-warp sum
    //   KQ_smem[16][32]           = 2048 B for per-row KQ values
    //   scale_factor_smem[16]     = 64 B
    //   Total: ~44.8 KiB (KVB=32) / ~28.5 KiB (KVB=16)
    //   At KVB=16, fits 2 blocks/SM at 32 KiB SMEM cap with margin.
    //   At KVB=32, 1 block/SM (over 32 KiB target but under 64 KiB cap).
    __shared__ half  Q_smem[N_ROWS * 256];      // 8 KiB
    __shared__ half  K_smem[32 * 256];           // 16 KiB
    __shared__ half  V_smem[32 * 256];           // 16 KiB
    __shared__ float D_partials[NWARPS][N_ROWS][8];
    __shared__ float KQ_smem[N_ROWS][32];
    __shared__ float scale_factor_smem[N_ROWS];

    // ----- Cooperative load Q into SMEM (rows 0..H_packed-1 real, rest zero) -----
    // 16 rows × Dq elements, 128 threads → ⌈16 * Dq / 128⌉ elements/thread.
    {
        const int total = N_ROWS * Dq;
        const int per_thread = (total + 127) / 128;
        for (int e = 0; e < per_thread; e++) {
            const int idx = tid + e * 128;
            if (idx >= total) break;
            const int r = idx / Dq;
            const int c = idx % Dq;
            half val;
            if (r < H_packed) {
                val = __ushort_as_half(Q[Q_off_row(r, c)]);
            } else {
                val = __float2half(0.0f);
            }
            Q_smem[r * Dq + c] = val;
        }
    }
    __syncthreads();

    for (int kb = 0; kb < n_kv; kb += KVB) {
        const int kb_end = min(kb + KVB, n_kv);
        const int blk    = kb_end - kb;

        // ----- Cooperative load K and V blocks into SMEM -----
        // K_smem[i][d] = K_cache[slot][kv_head][kb + i][d] for i in [0, blk).
        // V_smem[i][d] = V_cache[slot][kv_head][kb + i][d].
        // (Dq == Dv == 256 at our production tuple; same loop structure.)
        {
            const int total = blk * Dq;
            const int per_thread = (total + 127) / 128;
            for (int e = 0; e < per_thread; e++) {
                const int idx = tid + e * 128;
                if (idx >= total) break;
                const int n_idx = idx / Dq;
                const int d_idx = idx % Dq;
                const int k_global = kb + n_idx;
                K_smem[n_idx * Dq + d_idx] = __ushort_as_half(K[K_off(d_idx, k_global)]);
            }
            const int total_v = blk * Dv;
            const int per_thread_v = (total_v + 127) / 128;
            for (int e = 0; e < per_thread_v; e++) {
                const int idx = tid + e * 128;
                if (idx >= total_v) break;
                const int n_idx = idx / Dv;
                const int d_idx = idx % Dv;
                const int k_global = kb + n_idx;
                V_smem[n_idx * Dv + d_idx] = __ushort_as_half(V[V_off(d_idx, k_global)]);
            }
        }
        __syncthreads();

        // ----- KQ computation via mma (Approach C: rows 0..H-1 real) -----
        for (int n_tile = 0; n_tile * 8 < blk; n_tile++) {
            const int n_start = n_tile * 8;
            const int n_count = min(8, blk - n_start);

            tile<16, 8, float> D;
            #pragma unroll
            for (int l = 0; l < D.ne; l++) D.x[l] = 0.0f;

            const int k_chunks_per_warp = (Dq / 16) / NWARPS;
            const int k_chunk_start = warp_id * k_chunks_per_warp;
            const int k_chunk_end   = k_chunk_start + k_chunks_per_warp;

            for (int k_chunk = k_chunk_start; k_chunk < k_chunk_end; k_chunk++) {
                const int k_start = k_chunk * 16;

                // Build A tile from Q_smem (already populated with real Q for
                // rows 0..H_packed-1 and zeros for padded rows).
                tile<16, 8, half2> A;
                #pragma unroll
                for (int l = 0; l < A.ne; l++) {
                    const int row       = A.get_i(l);
                    const int half2_col = A.get_j(l);
                    const int d0 = k_start + 2 * half2_col;
                    const int d1 = d0 + 1;
                    half a = Q_smem[row * Dq + d0];
                    half b = Q_smem[row * Dq + d1];
                    A.x[l] = __halves2half2(a, b);
                }

                // B tile from K_smem via ldmatrix.sync.aligned.m8n8.x2.b16
                // (single warp-cooperative instruction → 4 half2/thread).
                // K_smem layout is row-major [n_idx][d]; pass base pointer to
                // (n_start, k_start) sub-block. Stride is Dq/2 in half2 units.
                // Boundary safety: ldmatrix reads garbage for invalid
                // n_in_blk ≥ blk; KQ_smem at those indices is then never
                // read in the softmax / V-accum loops (i iterates [0, blk)).
                tile<8, 8, half2> B;
                {
                    const half2 * K_h2 =
                        (const half2 *)(K_smem + n_start * Dq) + (k_start / 2);
                    load_ldmatrix(B, K_h2, Dq / 2);
                }

                mma(D, A, B);
            }

            // Write partial D to SMEM.
            {
                const int row_lo = lane / 4;
                const int row_hi = 8 + row_lo;
                const int col_lo = 2 * (lane % 4);
                D_partials[warp_id][row_lo][col_lo    ] = D.x[0];
                D_partials[warp_id][row_lo][col_lo + 1] = D.x[1];
                D_partials[warp_id][row_hi][col_lo    ] = D.x[2];
                D_partials[warp_id][row_hi][col_lo + 1] = D.x[3];
            }
            __syncthreads();

            // Cross-warp sum + extract ALL 16 rows × 8 cols of D into KQ_smem.
            {
                const int r_out = tid / 8;
                const int c_out = tid % 8;
                float sum = 0.0f;
                #pragma unroll
                for (int w = 0; w < NWARPS; w++) {
                    sum += D_partials[w][r_out][c_out];
                }
                KQ_smem[r_out][n_start + c_out] = sum;
            }
            __syncthreads();
        }

        // ----- Apply scale + softcap + mask to KQ_smem (all 16 rows × blk cols) -----
        // 128 threads × 1 element each = 128 elements covered if blk*16 ≤ 128.
        // For blk ≤ 8 (KVB=8): 8*16 = 128 ✓. For blk=16: 256 elements → 2 per thread.
        // For blk=32: 512 elements → 4 per thread.
        {
            const int total = N_ROWS * blk;
            const int per_thread = (total + 127) / 128;
            for (int i_thread = 0; i_thread < per_thread; i_thread++) {
                const int idx = tid + i_thread * 128;
                if (idx >= total) break;
                const int r = idx / blk;
                const int i = idx % blk;
                float acc = KQ_smem[r][i] * scale;
                if (use_softcap) {
                    acc = softcap * tanhf(acc / softcap);
                }
                acc += half_bits_to_float(mask[M_off(kb + i)]);
                // For padded rows (r >= H_packed), Q is zero so D was zero;
                // KQ_smem[r][i] holds (0 * scale + softcap_or_not + mask) = mask
                // value. Force these to -INF so they don't affect any other
                // row's softmax (they shouldn't, but safety) and so their own
                // sm = 0 → no spurious accumulation.
                if (r >= H_packed) acc = -INFINITY;
                KQ_smem[r][i] = acc;
            }
        }
        __syncthreads();

        // ----- Per-row online softmax state update -----
        // Lane 0..3 of each warp processes one row each (warp w handles rows
        // [w*4, w*4+4)). The scale_factor for VKQ rescale is published to
        // scale_factor_smem; the new kqmax / kqrowsum are written back.
        if (lane < ROWS_PER_WARP) {
            const int r = warp_id * ROWS_PER_WARP + lane;
            const float old_max     = kqmax_smem[r];
            const float old_rowsum  = kqrowsum_smem[r];

            float new_max = old_max;
            if (r < H_packed) {
                for (int i = 0; i < blk; i++) {
                    if (KQ_smem[r][i] > new_max) new_max = KQ_smem[r][i];
                }
            }

            if (new_max == -INFINITY) {
                // All-masked or padded row — no change to softmax state, no
                // VKQ contribution this block. scale_factor = 1 (no rescale).
                scale_factor_smem[r] = 1.0f;
            } else {
                const float scale_factor =
                    (old_max == -INFINITY) ? 0.0f : __expf(old_max - new_max);

                // Update kqrowsum: rescale old contribution + add new ones.
                float new_rowsum = old_rowsum * scale_factor;
                for (int i = 0; i < blk; i++) {
                    const float kq = KQ_smem[r][i];
                    new_rowsum += (kq == -INFINITY) ? 0.0f : __expf(kq - new_max);
                }

                kqmax_smem[r]        = new_max;
                kqrowsum_smem[r]     = new_rowsum;
                scale_factor_smem[r] = scale_factor;
            }
        }
        __syncthreads();

        // ----- VKQ rescale + V accumulation (this warp's 4 rows, V from SMEM) -----
        for (int r_in_warp = 0; r_in_warp < ROWS_PER_WARP; r_in_warp++) {
            const int r = warp_id * ROWS_PER_WARP + r_in_warp;
            if (r >= H_packed) continue;

            const float scale_f   = scale_factor_smem[r];
            const float row_kqmax = kqmax_smem[r];

            // Rescale OLD VKQ accumulator.
            if (scale_f != 1.0f) {
                #pragma unroll
                for (int d_offset = 0; d_offset < 8; d_offset++) {
                    VKQ[r_in_warp * 8 + d_offset] *= scale_f;
                }
            }

            if (row_kqmax == -INFINITY) continue;

            // Accumulate NEW V contributions for this K-block from V_smem.
            for (int i = 0; i < blk; i++) {
                const float kq = KQ_smem[r][i];
                const float sm = (kq == -INFINITY) ? 0.0f : __expf(kq - row_kqmax);
                if (sm == 0.0f) continue;
                #pragma unroll
                for (int d_offset = 0; d_offset < 8; d_offset++) {
                    const int d = lane + d_offset * 32;
                    if (d < Dv) {
                        VKQ[r_in_warp * 8 + d_offset] +=
                            sm * __half2float(V_smem[i * Dv + d]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // ----- Normalize + write output (this warp's 4 rows) -----
    for (int r_in_warp = 0; r_in_warp < ROWS_PER_WARP; r_in_warp++) {
        const int r = warp_id * ROWS_PER_WARP + r_in_warp;
        if (r >= H_packed) continue;

        const float den = kqrowsum_smem[r];
        for (int d_offset = 0; d_offset < 8; d_offset++) {
            const int d = lane + d_offset * 32;
            if (d < Dv) {
                const float val = (den > 0.0f) ? (VKQ[r_in_warp * 8 + d_offset] / den) : 0.0f;
                out_final[O_off(r, d)] = val;
            }
        }
    }
}

// ============================================================================
// Stage 2.3 kernel — Approach C decode pack + parallel_blocks split-K.
// ============================================================================
//
// Per spec §4 parallel_blocks. Each (slot, kv_head, ip) CTA processes a
// per-slot K-range slice [k_start, k_end) where:
//   k_chunk_size = ⌈n_kv_slot / pb_for_slot⌉
//   k_start = ip * k_chunk_size
//   k_end = min((ip+1) * k_chunk_size, n_kv_slot)
//
// Output: dst_partial[n_seqs][n_heads_q][max_pb][Dv] and
//         dst_meta[n_seqs][n_heads_q][max_pb] (float2 = max, rowsum).
//
// Companion `fattn_per_slot_kv_sm75_combine_kernel` reduces across ip per
// (slot, head_global) to produce the final output.
//
// Grid: (n_seqs, n_kv_heads, max_pb)
// Block: (32, 4, 1) — same 4-warp layout as stage22b
//
// Per-slot pb is read from `pb_per_slot[slot]` input. CTAs with ip >=
// pb_for_slot return early after writing (-INF, 0) to their dst_meta entry
// so combine treats them as no-contribution. dst_partial entries for those
// CTAs are left undefined (combine multiplies by 0 weight).
//
// All other aspects (Q SMEM, K SMEM, V SMEM, ldmatrix B-fragment, fp32
// accumulator, online softmax with per-row state, VKQ rescale) are
// identical to stage22b.

__global__ void fattn_per_slot_kv_sm75_stage23_kernel(
    const uint16_t * __restrict__ Q,
    const uint16_t * __restrict__ K,
    const uint16_t * __restrict__ V,
    const uint16_t * __restrict__ mask,
    const int32_t  * __restrict__ slot_seq_lens,
    const int32_t  * __restrict__ pb_per_slot,
    float          * __restrict__ dst_partial,    // [n_seqs][n_heads_q][max_pb][Dv]
    float2         * __restrict__ dst_meta,       // [n_seqs][n_heads_q][max_pb]
    int Dq, int Dv, int KVB,
    int n_heads_q, int n_kv_heads, int n_seqs, int n_kv_max,
    int H_packed, int max_pb,
    float scale, float softcap, bool use_softcap)
{
    using namespace ggml_cuda_mma;

    constexpr int NWARPS = 4;
    constexpr int N_ROWS = 16;
    constexpr int ROWS_PER_WARP = N_ROWS / NWARPS;
    constexpr int VKQ_PER_THREAD = 32;

    const int slot    = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int ip      = blockIdx.z;
    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int tid     = warp_id * 32 + lane;

    if (slot >= n_seqs || kv_head >= n_kv_heads || ip >= max_pb) return;

    const int gqa        = n_heads_q / n_kv_heads;
    const int head_start = kv_head * gqa;
    const int n_kv       = slot_seq_lens[slot];
    const int pb_for_slot = pb_per_slot[slot];

    // CTAs with ip >= pb_for_slot or empty slot: write sentinel meta and return.
    if (ip >= pb_for_slot || n_kv <= 0) {
        if (tid < H_packed) {
            const int r = tid;
            const int head_global = head_start + r;
            const size_t meta_idx = ((size_t)slot * n_heads_q + head_global) * max_pb + ip;
            dst_meta[meta_idx] = make_float2(-INFINITY, 0.0f);
        }
        return;
    }

    // Per-slot K range for this ip.
    const int k_chunk_size = (n_kv + pb_for_slot - 1) / pb_for_slot;
    const int k_start_global = ip * k_chunk_size;
    const int k_end_global   = min((ip + 1) * k_chunk_size, n_kv);

    if (k_start_global >= n_kv) {
        if (tid < H_packed) {
            const int r = tid;
            const int head_global = head_start + r;
            const size_t meta_idx = ((size_t)slot * n_heads_q + head_global) * max_pb + ip;
            dst_meta[meta_idx] = make_float2(-INFINITY, 0.0f);
        }
        return;
    }

    auto Q_off_row = [=] __device__ (int r, int d) -> size_t {
        const int head_global = head_start + r;
        return (((size_t)slot * n_heads_q + head_global) * /*n_tokens=*/1 + 0) * Dq + d;
    };
    auto K_off = [=] __device__ (int d, int k) -> size_t {
        return (((size_t)slot * n_kv_heads + kv_head) * n_kv_max + k) * Dq + d;
    };
    auto V_off = [=] __device__ (int d, int k) -> size_t {
        return (((size_t)slot * n_kv_heads + kv_head) * n_kv_max + k) * Dv + d;
    };
    auto M_off = [=] __device__ (int k) -> size_t {
        return (size_t)0 * n_kv_max + k;
    };

    float VKQ[VKQ_PER_THREAD] = {0};

    __shared__ half  Q_smem[N_ROWS * 256];
    __shared__ half  K_smem[32 * 256];
    __shared__ half  V_smem[32 * 256];
    __shared__ float D_partials[NWARPS][N_ROWS][8];
    __shared__ float KQ_smem[N_ROWS][32];
    __shared__ float scale_factor_smem[N_ROWS];
    __shared__ float kqmax_smem[N_ROWS];
    __shared__ float kqrowsum_smem[N_ROWS];

    // Load Q.
    {
        const int total = N_ROWS * Dq;
        const int per_thread = (total + 127) / 128;
        for (int e = 0; e < per_thread; e++) {
            const int idx = tid + e * 128;
            if (idx >= total) break;
            const int r = idx / Dq;
            const int c = idx % Dq;
            half val;
            if (r < H_packed) {
                val = __ushort_as_half(Q[Q_off_row(r, c)]);
            } else {
                val = __float2half(0.0f);
            }
            Q_smem[r * Dq + c] = val;
        }
    }

    if (tid < N_ROWS) {
        kqmax_smem[tid]    = -INFINITY;
        kqrowsum_smem[tid] = 0.0f;
    }
    __syncthreads();

    for (int kb = k_start_global; kb < k_end_global; kb += KVB) {
        const int kb_end = min(kb + KVB, k_end_global);
        const int blk    = kb_end - kb;

        // Cooperative load K and V blocks into SMEM.
        {
            const int total = blk * Dq;
            const int per_thread = (total + 127) / 128;
            for (int e = 0; e < per_thread; e++) {
                const int idx = tid + e * 128;
                if (idx >= total) break;
                const int n_idx = idx / Dq;
                const int d_idx = idx % Dq;
                const int k_global = kb + n_idx;
                K_smem[n_idx * Dq + d_idx] = __ushort_as_half(K[K_off(d_idx, k_global)]);
            }
            const int total_v = blk * Dv;
            const int per_thread_v = (total_v + 127) / 128;
            for (int e = 0; e < per_thread_v; e++) {
                const int idx = tid + e * 128;
                if (idx >= total_v) break;
                const int n_idx = idx / Dv;
                const int d_idx = idx % Dv;
                const int k_global = kb + n_idx;
                V_smem[n_idx * Dv + d_idx] = __ushort_as_half(V[V_off(d_idx, k_global)]);
            }
        }
        __syncthreads();

        for (int n_tile = 0; n_tile * 8 < blk; n_tile++) {
            const int n_start = n_tile * 8;

            tile<16, 8, float> D;
            #pragma unroll
            for (int l = 0; l < D.ne; l++) D.x[l] = 0.0f;

            const int k_chunks_per_warp = (Dq / 16) / NWARPS;
            const int k_chunk_start = warp_id * k_chunks_per_warp;
            const int k_chunk_end   = k_chunk_start + k_chunks_per_warp;

            for (int k_chunk = k_chunk_start; k_chunk < k_chunk_end; k_chunk++) {
                const int k_start = k_chunk * 16;

                tile<16, 8, half2> A;
                #pragma unroll
                for (int l = 0; l < A.ne; l++) {
                    const int row       = A.get_i(l);
                    const int half2_col = A.get_j(l);
                    const int d0 = k_start + 2 * half2_col;
                    const int d1 = d0 + 1;
                    half a = Q_smem[row * Dq + d0];
                    half b = Q_smem[row * Dq + d1];
                    A.x[l] = __halves2half2(a, b);
                }

                tile<8, 8, half2> B;
                {
                    const half2 * K_h2 =
                        (const half2 *)(K_smem + n_start * Dq) + (k_start / 2);
                    load_ldmatrix(B, K_h2, Dq / 2);
                }

                mma(D, A, B);
            }

            {
                const int row_lo = lane / 4;
                const int row_hi = 8 + row_lo;
                const int col_lo = 2 * (lane % 4);
                D_partials[warp_id][row_lo][col_lo    ] = D.x[0];
                D_partials[warp_id][row_lo][col_lo + 1] = D.x[1];
                D_partials[warp_id][row_hi][col_lo    ] = D.x[2];
                D_partials[warp_id][row_hi][col_lo + 1] = D.x[3];
            }
            __syncthreads();

            {
                const int r_out = tid / 8;
                const int c_out = tid % 8;
                float sum = 0.0f;
                #pragma unroll
                for (int w = 0; w < NWARPS; w++) {
                    sum += D_partials[w][r_out][c_out];
                }
                KQ_smem[r_out][n_start + c_out] = sum;
            }
            __syncthreads();
        }

        // Scale + softcap + mask.
        {
            const int total = N_ROWS * blk;
            const int per_thread = (total + 127) / 128;
            for (int i_thread = 0; i_thread < per_thread; i_thread++) {
                const int idx = tid + i_thread * 128;
                if (idx >= total) break;
                const int r = idx / blk;
                const int i = idx % blk;
                float acc = KQ_smem[r][i] * scale;
                if (use_softcap) {
                    acc = softcap * tanhf(acc / softcap);
                }
                acc += half_bits_to_float(mask[M_off(kb + i)]);
                if (r >= H_packed) acc = -INFINITY;
                KQ_smem[r][i] = acc;
            }
        }
        __syncthreads();

        // Per-row online softmax state update.
        if (lane < ROWS_PER_WARP) {
            const int r = warp_id * ROWS_PER_WARP + lane;
            const float old_max     = kqmax_smem[r];
            const float old_rowsum  = kqrowsum_smem[r];

            float new_max = old_max;
            if (r < H_packed) {
                for (int i = 0; i < blk; i++) {
                    if (KQ_smem[r][i] > new_max) new_max = KQ_smem[r][i];
                }
            }

            if (new_max == -INFINITY) {
                scale_factor_smem[r] = 1.0f;
            } else {
                const float scale_factor =
                    (old_max == -INFINITY) ? 0.0f : __expf(old_max - new_max);

                float new_rowsum = old_rowsum * scale_factor;
                for (int i = 0; i < blk; i++) {
                    const float kq = KQ_smem[r][i];
                    new_rowsum += (kq == -INFINITY) ? 0.0f : __expf(kq - new_max);
                }

                kqmax_smem[r]        = new_max;
                kqrowsum_smem[r]     = new_rowsum;
                scale_factor_smem[r] = scale_factor;
            }
        }
        __syncthreads();

        // VKQ rescale + V accumulation.
        for (int r_in_warp = 0; r_in_warp < ROWS_PER_WARP; r_in_warp++) {
            const int r = warp_id * ROWS_PER_WARP + r_in_warp;
            if (r >= H_packed) continue;

            const float scale_f   = scale_factor_smem[r];
            const float row_kqmax = kqmax_smem[r];

            if (scale_f != 1.0f) {
                #pragma unroll
                for (int d_offset = 0; d_offset < 8; d_offset++) {
                    VKQ[r_in_warp * 8 + d_offset] *= scale_f;
                }
            }

            if (row_kqmax == -INFINITY) continue;

            for (int i = 0; i < blk; i++) {
                const float kq = KQ_smem[r][i];
                const float sm = (kq == -INFINITY) ? 0.0f : __expf(kq - row_kqmax);
                if (sm == 0.0f) continue;
                #pragma unroll
                for (int d_offset = 0; d_offset < 8; d_offset++) {
                    const int d = lane + d_offset * 32;
                    if (d < Dv) {
                        VKQ[r_in_warp * 8 + d_offset] +=
                            sm * __half2float(V_smem[i * Dv + d]);
                    }
                }
            }
        }
        __syncthreads();
    }

    // ----- Write partial output to dst_partial and dst_meta -----
    // This CTA's (slot, kv_head, ip) writes H_packed (head_global, ip) entries.
    // Per-thread per-row: 8 VKQ dims (strided lane + d_offset*32).
    for (int r_in_warp = 0; r_in_warp < ROWS_PER_WARP; r_in_warp++) {
        const int r = warp_id * ROWS_PER_WARP + r_in_warp;
        if (r >= H_packed) continue;
        const int head_global = head_start + r;

        #pragma unroll
        for (int d_offset = 0; d_offset < 8; d_offset++) {
            const int d = lane + d_offset * 32;
            if (d < Dv) {
                const size_t part_idx =
                    (((size_t)slot * n_heads_q + head_global) * max_pb + ip) * Dv + d;
                dst_partial[part_idx] = VKQ[r_in_warp * 8 + d_offset];
            }
        }
        // Thread 0 of this row's owning warp writes the meta.
        if (lane == 0) {
            const size_t meta_idx =
                ((size_t)slot * n_heads_q + head_global) * max_pb + ip;
            dst_meta[meta_idx] = make_float2(kqmax_smem[r], kqrowsum_smem[r]);
        }
    }
}

// ============================================================================
// Stage 2.3 combine kernel — reduce across ip per (slot, head_global).
// ============================================================================
//
// Grid: (n_seqs, n_heads_q, 1)
// Block: (Dv, 1, 1)
//
// Each CTA reads up to max_pb partials for its (slot, head_global), finds
// the combined max, applies exp(ip_max - combined_max) weights, and writes
// the final output to out_final[n_seqs][n_heads_q][n_tokens=1][Dv].

__global__ void fattn_per_slot_kv_sm75_combine_kernel(
    const float  * __restrict__ dst_partial,   // [n_seqs][n_heads_q][max_pb][Dv]
    const float2 * __restrict__ dst_meta,      // [n_seqs][n_heads_q][max_pb]
    const int32_t * __restrict__ pb_per_slot,  // [n_seqs]
    float * __restrict__ out_final,            // [n_seqs][n_heads_q][Dv]
    int Dv, int n_heads_q, int n_seqs, int max_pb)
{
    const int slot = blockIdx.x;
    const int head = blockIdx.y;
    const int tid  = threadIdx.x;

    if (slot >= n_seqs || head >= n_heads_q || tid >= Dv) return;

    const int pb = pb_per_slot[slot];

    // Phase 1: combined max across all valid ip's max.
    __shared__ float kqmax_shared;
    if (tid == 0) {
        float m = -INFINITY;
        for (int ip = 0; ip < pb; ip++) {
            const size_t meta_idx = ((size_t)slot * n_heads_q + head) * max_pb + ip;
            const float ip_max = dst_meta[meta_idx].x;
            if (ip_max > m) m = ip_max;
        }
        kqmax_shared = m;
    }
    __syncthreads();

    const float kqmax = kqmax_shared;
    if (kqmax == -INFINITY) {
        out_final[((size_t)slot * n_heads_q + head) * Dv + tid] = 0.0f;
        return;
    }

    // Phase 2: weighted combine.
    float num = 0.0f, den = 0.0f;
    for (int ip = 0; ip < pb; ip++) {
        const size_t meta_idx = ((size_t)slot * n_heads_q + head) * max_pb + ip;
        const float ip_max = dst_meta[meta_idx].x;
        const float ip_sum = dst_meta[meta_idx].y;
        if (ip_max == -INFINITY) continue;
        const float w = __expf(ip_max - kqmax);
        const size_t part_idx =
            (((size_t)slot * n_heads_q + head) * max_pb + ip) * Dv + tid;
        num += w * dst_partial[part_idx];
        den += w * ip_sum;
    }

    const float val = (den > 0.0f) ? num / den : 0.0f;
    out_final[((size_t)slot * n_heads_q + head) * Dv + tid] = val;
}

// ============================================================================
// Host-side launcher (extern "C" symbol consumed by the unit test).
// ============================================================================

extern "C" int fattn_per_slot_kv_sm75_launch(
    const HostHalf * Q_h,
    const HostHalf * K_h,
    const HostHalf * V_h,
    const HostHalf * mask_h,
    const int32_t  * slot_seq_lens_h,
    float          * out_final_h,
    const LauncherConfig & cfg)
{
    const int Dq      = cfg.head_dim_q;
    const int Dv      = cfg.head_dim_v;
    const int KVB     = cfg.kv_block_size;
    const int n_tok   = cfg.n_tokens;
    const int n_hq    = cfg.n_heads_q;
    const int n_hkv   = cfg.n_kv_heads;
    const int n_seqs  = cfg.n_seqs;
    const int n_kvmax = cfg.n_kv_max;

    if (Dq != Dv || Dq != 256) {
        fprintf(stderr,
                "fattn_per_slot_kv_sm75_launch (skeleton): only HEAD_DIM_Q == "
                "HEAD_DIM_V == 256 supported; got (%d, %d)\n", Dq, Dv);
        return -10;
    }
    if (Dv % 32 != 0) return -11;
    if ((KVB != 16) && (KVB != 32)) {
        fprintf(stderr,
                "fattn_per_slot_kv_sm75_launch (skeleton): KV_BLOCK_SIZE must "
                "be 16 or 32; got %d\n", KVB);
        return -12;
    }
    if (n_hq % n_hkv != 0) return -13;

    // Element counts.
    const size_t n_Q = (size_t)Dq * n_tok * n_hq * n_seqs;
    const size_t n_K = (size_t)Dq * n_kvmax * n_hkv * n_seqs;
    const size_t n_V = (size_t)Dv * n_kvmax * n_hkv * n_seqs;
    const size_t n_M = (size_t)n_kvmax * n_tok;
    const size_t n_O = (size_t)Dv * n_tok * n_hq * n_seqs;
    const size_t n_S = (size_t)n_seqs;

    // Allocate device buffers.
    uint16_t * Q_d = nullptr;
    uint16_t * K_d = nullptr;
    uint16_t * V_d = nullptr;
    uint16_t * M_d = nullptr;
    int32_t  * S_d = nullptr;
    float    * O_d = nullptr;

    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&Q_d, n_Q * sizeof(uint16_t));            if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&K_d, n_K * sizeof(uint16_t));            if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&V_d, n_V * sizeof(uint16_t));            if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&M_d, n_M * sizeof(uint16_t));            if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&S_d, n_S * sizeof(int32_t));             if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&O_d, n_O * sizeof(float));               if (err != cudaSuccess) goto cleanup;

    // H2D.
    err = cudaMemcpy(Q_d, Q_h,             n_Q * sizeof(uint16_t), cudaMemcpyHostToDevice); if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(K_d, K_h,             n_K * sizeof(uint16_t), cudaMemcpyHostToDevice); if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(V_d, V_h,             n_V * sizeof(uint16_t), cudaMemcpyHostToDevice); if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(M_d, mask_h,          n_M * sizeof(uint16_t), cudaMemcpyHostToDevice); if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(S_d, slot_seq_lens_h, n_S * sizeof(int32_t),  cudaMemcpyHostToDevice); if (err != cudaSuccess) goto cleanup;
    err = cudaMemset(O_d, 0,               n_O * sizeof(float));                            if (err != cudaSuccess) goto cleanup;

    // Launch. Variant selectable via env (debug):
    //   FATTN_KERNEL_VARIANT=phase1   → 1-warp scalar (Phase 1)
    //   FATTN_KERNEL_VARIANT=stage21  → 1-warp mma.sync (Stage 2.1)
    //   FATTN_KERNEL_VARIANT=stage22a → 4-warp Approach A (forced — all shapes)
    //   default                        → Approach C decode pack (Stage 2.2b)
    //                                     for n_tokens=1, else Stage 2.2a.
    {
        const char * variant = std::getenv("FATTN_KERNEL_VARIANT");
        const bool use_phase1   = (variant && std::strcmp(variant, "phase1")   == 0);
        const bool use_stage21  = (variant && std::strcmp(variant, "stage21")  == 0);
        const bool use_stage22a = (variant && std::strcmp(variant, "stage22a") == 0);

        if (use_phase1) {
            const dim3 grid(n_seqs, n_tok, n_hq);
            const dim3 block(32, 1, 1);
            fattn_per_slot_kv_sm75_naive_kernel<<<grid, block>>>(
                Q_d, K_d, V_d, M_d, S_d, O_d,
                Dq, Dv, KVB,
                n_tok, n_hq, n_hkv, n_seqs, n_kvmax,
                cfg.scale, cfg.softcap, cfg.use_softcap
            );
        } else if (use_stage21) {
            const dim3 grid(n_seqs, n_tok, n_hq);
            const dim3 block(32, 1, 1);
            fattn_per_slot_kv_sm75_stage21_kernel<<<grid, block>>>(
                Q_d, K_d, V_d, M_d, S_d, O_d,
                Dq, Dv, KVB,
                n_tok, n_hq, n_hkv, n_seqs, n_kvmax,
                cfg.scale, cfg.softcap, cfg.use_softcap
            );
        } else if (use_stage22a || n_tok > 1) {
            // Stage 2.2a for prefill / multi-token paths (or forced via env).
            const dim3 grid(n_seqs, n_tok, n_hq);
            const dim3 block(32, 4, 1);
            fattn_per_slot_kv_sm75_stage22a_kernel<<<grid, block>>>(
                Q_d, K_d, V_d, M_d, S_d, O_d,
                Dq, Dv, KVB,
                n_tok, n_hq, n_hkv, n_seqs, n_kvmax,
                cfg.scale, cfg.softcap, cfg.use_softcap
            );
        } else {
            // Default decode path. Stage 2.3 (split-K) is opt-in via env
            // FATTN_KERNEL_VARIANT=stage23; default remains Stage 2.2b
            // (single-pass Approach C decode pack) until split-K is validated
            // at production shape.
            const bool use_stage23 =
                (variant && std::strcmp(variant, "stage23") == 0);
            const int gqa = n_hq / n_hkv;
            if (gqa > 16) {
                // Fall back to Stage 2.2a for unsupported gqa values.
                const dim3 grid(n_seqs, n_tok, n_hq);
                const dim3 block(32, 4, 1);
                fattn_per_slot_kv_sm75_stage22a_kernel<<<grid, block>>>(
                    Q_d, K_d, V_d, M_d, S_d, O_d,
                    Dq, Dv, KVB,
                    n_tok, n_hq, n_hkv, n_seqs, n_kvmax,
                    cfg.scale, cfg.softcap, cfg.use_softcap
                );
            } else if (!use_stage23) {
                const dim3 grid(n_seqs, n_hkv, 1);
                const dim3 block(32, 4, 1);
                fattn_per_slot_kv_sm75_stage22b_kernel<<<grid, block>>>(
                    Q_d, K_d, V_d, M_d, S_d, O_d,
                    Dq, Dv, KVB,
                    n_hq, n_hkv, n_seqs, n_kvmax,
                    /*H_packed=*/gqa,
                    cfg.scale, cfg.softcap, cfg.use_softcap
                );
            } else {
                // Stage 2.3: parallel_blocks split-K + combine kernel.
                // Compute pb_per_slot matching the oracle's formula
                // (max(1, (slot_seq_lens[s] + 255) / 256)).
                std::vector<int32_t> pb_per_slot_h(n_seqs);
                int max_pb = 1;
                for (int s = 0; s < n_seqs; s++) {
                    const int pb = std::max(1, (slot_seq_lens_h[s] + 255) / 256);
                    pb_per_slot_h[s] = pb;
                    if (pb > max_pb) max_pb = pb;
                }

                int32_t * pb_d = nullptr;
                float * dst_partial_d = nullptr;
                float2 * dst_meta_d = nullptr;
                err = cudaMalloc(&pb_d, n_seqs * sizeof(int32_t));
                if (err != cudaSuccess) goto cleanup_stage23;
                err = cudaMemcpy(pb_d, pb_per_slot_h.data(),
                                 n_seqs * sizeof(int32_t), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) goto cleanup_stage23;

                {
                    const size_t n_part = (size_t)n_seqs * n_hq * max_pb * Dv;
                    const size_t n_meta = (size_t)n_seqs * n_hq * max_pb;
                    err = cudaMalloc(&dst_partial_d, n_part * sizeof(float));
                    if (err != cudaSuccess) goto cleanup_stage23;
                    err = cudaMalloc(&dst_meta_d, n_meta * sizeof(float2));
                    if (err != cudaSuccess) goto cleanup_stage23;
                    // Init dst_meta to (-INF, 0). Easier from device side, but
                    // a cudaMemset to a bit pattern isn't trivial for fp32 -INF.
                    // Host-init + H2D for the small meta array.
                    std::vector<float2> meta_init(n_meta, make_float2(-INFINITY, 0.0f));
                    err = cudaMemcpy(dst_meta_d, meta_init.data(),
                                     n_meta * sizeof(float2), cudaMemcpyHostToDevice);
                    if (err != cudaSuccess) goto cleanup_stage23;
                }

                {
                    const dim3 grid(n_seqs, n_hkv, max_pb);
                    const dim3 block(32, 4, 1);
                    fattn_per_slot_kv_sm75_stage23_kernel<<<grid, block>>>(
                        Q_d, K_d, V_d, M_d, S_d, pb_d,
                        dst_partial_d, dst_meta_d,
                        Dq, Dv, KVB,
                        n_hq, n_hkv, n_seqs, n_kvmax,
                        /*H_packed=*/gqa, max_pb,
                        cfg.scale, cfg.softcap, cfg.use_softcap
                    );
                    err = cudaGetLastError();
                    if (err != cudaSuccess) goto cleanup_stage23;
                }

                {
                    const dim3 grid_combine(n_seqs, n_hq, 1);
                    const dim3 block_combine(Dv, 1, 1);
                    fattn_per_slot_kv_sm75_combine_kernel<<<grid_combine, block_combine>>>(
                        dst_partial_d, dst_meta_d, pb_d, O_d,
                        Dv, n_hq, n_seqs, max_pb
                    );
                    err = cudaGetLastError();
                    if (err != cudaSuccess) goto cleanup_stage23;
                }

                err = cudaDeviceSynchronize();

            cleanup_stage23:
                if (pb_d) cudaFree(pb_d);
                if (dst_partial_d) cudaFree(dst_partial_d);
                if (dst_meta_d) cudaFree(dst_meta_d);
                if (err != cudaSuccess) goto cleanup;
            }
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) goto cleanup;
    }

    // D2H.
    err = cudaMemcpy(out_final_h, O_d, n_O * sizeof(float), cudaMemcpyDeviceToHost);

cleanup:
    if (Q_d) cudaFree(Q_d);
    if (K_d) cudaFree(K_d);
    if (V_d) cudaFree(V_d);
    if (M_d) cudaFree(M_d);
    if (S_d) cudaFree(S_d);
    if (O_d) cudaFree(O_d);

    if (err != cudaSuccess) {
        fprintf(stderr, "fattn_per_slot_kv_sm75_launch CUDA error: %s\n",
                cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

// ============================================================================
// Production-side dispatch entry — called from fattn.cu when the new ggml op
// GGML_OP_FLASH_ATTN_EXT_PER_SLOT_KV is encountered.
// ============================================================================
//
// Takes device pointers (from ggml_tensor->data) + ggml_backend_cuda_context
// for pool allocation + stream. Uses Stage 2.3 split-K + combine path.
//
// Caller responsibility:
//   - dst->src[0]=Q, [1]=K, [2]=V, [3]=mask, [5]=slot_seq_lens (i32, n_seqs)
//   - All tensors fp16 (Q/K/V/mask); slot_seq_lens i32
//   - dst is fp32 output [Dv, n_heads_q, n_tokens, n_seqs]
//
// Phase 1 wiring scope: gqa <= 16 (Approach C decode pack supports up to 16
// rows). For n_tokens > 1, route to Stage 2.2a (prefill path). For n_tokens=1
// and gqa <= 16, route to Stage 2.3 (split-K + combine).

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/convert.cuh"

void ggml_cuda_flash_attn_ext_per_slot_kv_sm75(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst)
{
    const ggml_tensor * Q             = dst->src[0];
    const ggml_tensor * K             = dst->src[1];
    const ggml_tensor * V             = dst->src[2];
    const ggml_tensor * mask          = dst->src[3];
    const ggml_tensor * slot_seq_lens = dst->src[5];

    GGML_ASSERT(Q && K && V && mask && slot_seq_lens);
    GGML_ASSERT(Q->type == GGML_TYPE_F32);   // ggml FA convention: Q is fp32
    GGML_ASSERT(K->type == GGML_TYPE_F16);
    GGML_ASSERT(V->type == GGML_TYPE_F16);
    GGML_ASSERT(mask->type == GGML_TYPE_F16);
    GGML_ASSERT(slot_seq_lens->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int Dq      = (int) Q->ne[0];
    const int Dv      = (int) V->ne[0];
    const int n_tok   = (int) Q->ne[1];
    const int n_hq    = (int) Q->ne[2];
    const int n_seqs  = (int) Q->ne[3];
    const int n_kvmax = (int) K->ne[1];
    const int n_hkv   = (int) K->ne[2];
    const int KVB     = 16;  // primary at Dv=256 per spec §9

    GGML_ASSERT(Dq == 256 && Dv == 256);
    GGML_ASSERT(n_hq % n_hkv == 0);
    const int gqa = n_hq / n_hkv;
    GGML_ASSERT(gqa <= 16);

    // Op_params: scale, max_bias, softcap, prec, swa_window (same as FA_EXT).
    float scale = 1.0f, max_bias = 0.0f, softcap = 0.0f;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (softcap != 0.0f) {
        scale /= softcap;
    }
    const bool use_softcap = (softcap != 0.0f);

    // Convert Q from fp32 to fp16 (FA kernels consume fp16 Q).
    ggml_cuda_pool_alloc<uint16_t> Q_h_buf(ctx.pool(),
        (size_t)Dq * n_tok * n_hq * n_seqs);
    GGML_ASSERT(Q_h_buf.get() != nullptr);
    {
        auto to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
        to_fp16(Q->data, (half *) Q_h_buf.get(),
                1, (int64_t)Dq * n_tok * n_hq * n_seqs, ctx.stream());
    }

    cudaStream_t stream = ctx.stream();
    const uint16_t * Q_d = Q_h_buf.get();
    const uint16_t * K_d = (const uint16_t *) K->data;
    const uint16_t * V_d = (const uint16_t *) V->data;
    const uint16_t * M_d = (const uint16_t *) mask->data;
    const int32_t  * S_d = (const int32_t  *) slot_seq_lens->data;
    float          * O_d = (float          *) dst->data;

    if (n_tok > 1) {
        // Prefill path: Stage 2.2a (Approach A, no split-K).
        const dim3 grid(n_seqs, n_tok, n_hq);
        const dim3 block(32, 4, 1);
        fattn_per_slot_kv_sm75_stage22a_kernel<<<grid, block, 0, stream>>>(
            Q_d, K_d, V_d, M_d, S_d, O_d,
            Dq, Dv, KVB,
            n_tok, n_hq, n_hkv, n_seqs, n_kvmax,
            scale, softcap, use_softcap
        );
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Decode path: Stage 2.3 split-K + combine. Pool-allocate partials.
    // pb_per_slot must come from the device-side slot_seq_lens. We need it
    // on host to compute max_pb (grid.z). Tiny array (~n_seqs ints): copy
    // back to host transiently.
    std::vector<int32_t> slot_seq_lens_h(n_seqs);
    CUDA_CHECK(cudaMemcpyAsync(slot_seq_lens_h.data(), S_d,
        n_seqs * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int32_t> pb_per_slot_h(n_seqs);
    int max_pb = 1;
    for (int s = 0; s < n_seqs; s++) {
        const int pb = std::max(1, (slot_seq_lens_h[s] + 255) / 256);
        pb_per_slot_h[s] = pb;
        if (pb > max_pb) max_pb = pb;
    }

    ggml_cuda_pool_alloc<int32_t> pb_buf(ctx.pool(), n_seqs);
    ggml_cuda_pool_alloc<float>   dst_partial_buf(ctx.pool(),
        (size_t)n_seqs * n_hq * max_pb * Dv);
    ggml_cuda_pool_alloc<float2>  dst_meta_buf(ctx.pool(),
        (size_t)n_seqs * n_hq * max_pb);
    GGML_ASSERT(pb_buf.get() != nullptr);
    GGML_ASSERT(dst_partial_buf.get() != nullptr);
    GGML_ASSERT(dst_meta_buf.get() != nullptr);

    CUDA_CHECK(cudaMemcpyAsync(pb_buf.get(), pb_per_slot_h.data(),
        n_seqs * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    // Init dst_meta to (-INF, 0). Use a small kernel since cudaMemset can't
    // write -INF directly.
    {
        std::vector<float2> meta_init((size_t)n_seqs * n_hq * max_pb,
            make_float2(-INFINITY, 0.0f));
        CUDA_CHECK(cudaMemcpyAsync(dst_meta_buf.get(), meta_init.data(),
            meta_init.size() * sizeof(float2), cudaMemcpyHostToDevice, stream));
    }

    {
        const dim3 grid(n_seqs, n_hkv, max_pb);
        const dim3 block(32, 4, 1);
        fattn_per_slot_kv_sm75_stage23_kernel<<<grid, block, 0, stream>>>(
            Q_d, K_d, V_d, M_d, S_d, pb_buf.get(),
            dst_partial_buf.get(), dst_meta_buf.get(),
            Dq, Dv, KVB,
            n_hq, n_hkv, n_seqs, n_kvmax,
            /*H_packed=*/gqa, max_pb,
            scale, softcap, use_softcap
        );
        CUDA_CHECK(cudaGetLastError());
    }

    {
        const dim3 grid(n_seqs, n_hq, 1);
        const dim3 block(Dv, 1, 1);
        fattn_per_slot_kv_sm75_combine_kernel<<<grid, block, 0, stream>>>(
            dst_partial_buf.get(), dst_meta_buf.get(), pb_buf.get(), O_d,
            Dv, n_hq, n_seqs, max_pb
        );
        CUDA_CHECK(cudaGetLastError());
    }
}
