/*
 * ggml-turbo-kv.h — port of quantumaikr/quant.cpp v0.8.0 turbo_kv_4b
 *
 * Real TurboQuant (ICLR 2026 / HIGGS Nov 2024 algorithm family):
 *   Random Hadamard Transform + 16-entry Lloyd-Max-Gaussian codebook.
 *
 * Block layout (72 bytes per 128 elements, 4.5 bpe):
 *
 *   struct block_turbo_kv_4b {
 *       uint16_t norm;            // fp16, ||x||_2 before normalization
 *       uint16_t residual_norm;   // unused (reserved for future composite residual)
 *       uint16_t inv_std_fp16;    // fp16, = 2.7326 / max(|rotated|), per block
 *       uint16_t _pad;            // alignment
 *       uint8_t  mse_indices[64]; // 4-bit packed, 2 indices/byte, LSB-first
 *   };
 *
 * Pipeline (per 128-element block):
 *
 *   quantize:   norm <- ||x||
 *               r    <- x / norm
 *               r    <- RHT(r, seed=TURBO_KV_DEFAULT_SEED)
 *               inv_std <- 2.7326 / max(|r|)
 *               mse_indices[i] <- argmin_c |codebook[c] - r[i] * inv_std|
 *
 *   dequant:    r[i] <- codebook[mse_indices[i]] / inv_std
 *               r    <- RHT^{-1}(r)
 *               x[i] <- r[i] * norm
 *
 *   attention:  pre-rotate query once:   q_rot = RHT(query)
 *               per block: compute <q_rot, dequant_rotated_k_block> * norm
 *               -- inverse RHT not needed on K side because <a,b> = <RHT(a),RHT(b)>
 *                  for orthogonal RHT
 *
 * The 16-entry Lloyd-Max-Gaussian codebook is a static constant — no calibration
 * data needed. See tq_codebook.c in quant.cpp for provenance (Max 1960 tables).
 *
 * Source: github.com/quantumaikr/quant.cpp v0.8.0, Apache-2.0 licensed.
 * Vendored with attribution into the polaris-hybrid-cpu-opt branch to support
 * CPU-bound KV compression on Qwen3.5-35B-A3B and similar hybrid DeltaNet/MoE
 * models where KV bandwidth matters at long context.
 */

#ifndef GGML_TURBO_KV_H
#define GGML_TURBO_KV_H

#include "ggml.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TURBO_KV_BLOCK_SIZE 128    /* elements per block */
#define TURBO_KV_4B_BYTES   72     /* bytes per block */

/* Random Hadamard Transform seed (quant.cpp TKV_DEFAULT_SEED).
 * Shared between quantize, dequantize, and query pre-rotation so that
 * <query, key>  ==  <RHT(query), RHT(key)>.
 */
#define TURBO_KV_DEFAULT_SEED 0x12345678u

/* Largest Lloyd-Max centroid for N(0,1) at 4-bit precision.
 * Used to scale rotated values so max_abs maps to ±CENT_4BIT_MAX. */
#define TURBO_KV_4B_CENT_MAX 2.7326f

typedef struct {
    uint16_t norm;            /* fp16, ||x||_2 of the original vector */
    uint16_t residual_norm;   /* unused (kept for layout with future residual) */
    uint16_t inv_std_fp16;    /* fp16, = CENT_MAX / max(|rotated|), per block */
    uint16_t _pad;            /* alignment */
    uint8_t  mse_indices[TURBO_KV_BLOCK_SIZE / 2]; /* 4-bit packed, 2 per byte */
} block_turbo_kv_4b;

/* Compile-time size check: 2+2+2+2+64 = 72 */
typedef char turbo_kv_4b_check_block_size
    [(sizeof(block_turbo_kv_4b) == TURBO_KV_4B_BYTES) ? 1 : -1];

/* ============================================================
 * Lloyd-Max-Gaussian 16-entry codebook for N(0,1)
 * ============================================================ */
extern const float turbo_kv_4b_codebook[16];

/* ============================================================
 * Random Hadamard Transform (O(n log n) butterfly, in-place)
 *
 * n must be a power of 2. The transform is:
 *
 *   data[i] <- data[i] * random_sign(seed, i)
 *   data    <- walsh_hadamard(data)
 *   data[i] <- data[i] / sqrt(n)
 *
 * Self-inverse except for the sign step: to invert, apply the scale then
 * walsh_hadamard then the same sign flip.
 * ============================================================ */
void turbo_kv_rht_forward(float * data, int n, uint32_t seed);
void turbo_kv_rht_inverse(float * data, int n, uint32_t seed);

/* ============================================================
 * Quantize / dequantize API
 *
 * k must be a multiple of TURBO_KV_BLOCK_SIZE (128).
 * ============================================================ */
void quantize_row_turbo_kv_4b_ref(const float * x, block_turbo_kv_4b * y, int64_t k);
void dequantize_row_turbo_kv_4b  (const block_turbo_kv_4b * x, float * y, int64_t k);

/* ============================================================
 * Vector dot product: <f32 query, quantized key>
 *
 * Called by ggml mul_mat and flash-attention kq_vec_dot paths. This is
 * the hot inner loop during attention. Takes a pre-rotated query (see
 * turbo_kv_rotate_query_once below) and a pointer to one K block.
 *
 * n is the head dimension in elements (must be <= TURBO_KV_BLOCK_SIZE).
 * ============================================================ */
void ggml_vec_dot_turbo_kv_4b_f32(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by,
    int nrc);

/* ============================================================
 * Query pre-rotation helper.
 *
 * Called once per Q row per attention layer. The result is a rotated
 * f32 buffer that can be dotted directly against K blocks without any
 * per-block inverse rotation (RHT is orthogonal).
 *
 * dim must be a power of 2 <= TURBO_KV_BLOCK_SIZE; if dim < BK, the
 * upper entries of out are zero-padded.
 * ============================================================ */
void turbo_kv_rotate_query(const float * q, float * out, int dim);

/* ============================================================
 * Batched attention: compute scores for all K positions in one call.
 *
 * Rotates Q ONCE, then loops over valid_count K positions using the
 * SSSE3 inner kernel. Eliminates the per-call rotation overhead that
 * caused +52% wall-clock regression on tool-call workloads.
 *
 * Mirrors tq_kv_1b_attention_multi in ggml-turbo-quant.h. The FA path
 * in ops.cpp calls this once per thread with a slice of K. The non-FA
 * vec_dot wrapper calls it with valid_count=1 as a fallback.
 *
 * @param query         Raw f32 query vector (head_dim elements, NOT pre-rotated)
 * @param kv_cache      Pointer to the first K block for position 0 in this head
 * @param scores        Output: one f32 score per K position (valid_count elements)
 * @param valid_count   Number of K positions to process
 * @param head_dim      Head dimension (e.g. 256 for Qwen3.5), multiple of 128
 * @param k_stride_blocks  Number of turbo_kv_4b blocks between consecutive K
 *                         positions in the cache. For dense layout: head_dim/128.
 *                         For GQA interleave: n_embd_k_gqa / 128. Pass 0 for
 *                         the default (= head_dim / 128).
 * ============================================================ */
void turbo_kv_4b_attention_multi(
    const float              * query,
    const block_turbo_kv_4b  * kv_cache,
    float                    * scores,
    int                        valid_count,
    int                        head_dim,
    int                        k_stride_blocks);

#ifdef __cplusplus
}
#endif

#endif /* GGML_TURBO_KV_H */
