/*
 * ggml-turbo-kv.c — scalar C reference for turbo_kv_4b
 *
 * Ported from quantumaikr/quant.cpp v0.8.0 (Apache-2.0).
 *
 * This file contains:
 *   - The 16-entry Lloyd-Max-Gaussian codebook table
 *   - Random Hadamard Transform (forward + inverse) using an O(n log n)
 *     butterfly and a Knuth-hash-based random-sign diagonal.
 *   - quantize_row_turbo_kv_4b_ref and dequantize_row_turbo_kv_4b, operating
 *     block-at-a-time (128 elements in, 72 bytes out or vice versa).
 *   - turbo_kv_rotate_query: the Q-side preprocessing that happens once per
 *     attention row.
 *   - ggml_vec_dot_turbo_kv_4b_f32: the scalar fallback K dot used by the
 *     type-traits table. SIMD overrides live in arch-specific headers.
 *
 * The SSSE3 (_mm_shuffle_epi8) kernel for Westmere lives in
 * ggml/src/ggml-cpu/arch/x86/turbo_kv_4b_sse.h and overrides this scalar
 * path via the type_traits_cpu registration.
 */

#include "ggml-turbo-kv.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif
#include <float.h>

/* ============================================================
 * Lloyd-Max-Gaussian 16-entry codebook for N(0,1)
 * ============================================================ */

const float turbo_kv_4b_codebook[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f,
    -0.9423f, -0.6568f, -0.3881f, -0.1284f,
     0.1284f,  0.3881f,  0.6568f,  0.9423f,
     1.2562f,  1.6180f,  2.0690f,  2.7326f
};

/* Per-centroid decision boundaries are midpoints between consecutive
 * centroids. For nearest-centroid lookup we compute |x - c| and take min.
 * With only 16 entries, linear scan beats binary search (~15 cmps). */

/* ============================================================
 * Random sign generation from seed (Knuth multiplicative hash)
 * ============================================================ */

static inline int turbo_kv_random_sign(uint32_t seed, int idx) {
    uint32_t h = seed ^ (uint32_t) idx;
    h = h * 2654435761u;
    return (h & 1u) ? 1 : -1;
}

/* ============================================================
 * In-place Walsh-Hadamard butterfly (unnormalized).
 *
 * For len = 1, 2, 4, ..., n/2:
 *   for each block of 2*len elements:
 *     butterfly the pair (u[j], u[j+len]) -> (u+v, u-v)
 *
 * n must be power of 2. Cost: n * log2(n) adds/subs.
 * ============================================================ */
static void walsh_hadamard(float * data, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += (len << 1)) {
            for (int j = 0; j < len; j++) {
                const float u = data[i + j];
                const float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }
}

#ifdef __SSE4_1__
/* SSE4.1 Walsh-Hadamard butterfly. n must be power of 2, >= 4.
 *
 * Stage 1 (stride 1): [u0,v0,u1,v1] -> [u0+v0, u0-v0, u1+v1, u1-v1]
 *   hadd/hsub give horizontal add/sub, unpacklo interleaves them.
 *
 * Stage 2 (stride 2): [a,b,c,d] -> [a+c, b+d, a-c, b-d]
 *   movelh/movehl split low/high halves, add/sub, shuffle to recombine.
 *
 * Stage 3+ (stride >= 4): contiguous 4-wide load, add, sub, store. */
static void walsh_hadamard_sse(float * data, int n) {
    for (int i = 0; i < n; i += 4) {
        __m128 a = _mm_loadu_ps(&data[i]);
        _mm_storeu_ps(&data[i], _mm_unpacklo_ps(_mm_hadd_ps(a, a),
                                                 _mm_hsub_ps(a, a)));
    }
    for (int i = 0; i < n; i += 4) {
        __m128 a  = _mm_loadu_ps(&data[i]);
        __m128 lo = _mm_movelh_ps(a, a);
        __m128 hi = _mm_movehl_ps(a, a);
        _mm_storeu_ps(&data[i], _mm_shuffle_ps(_mm_add_ps(lo, hi),
                                                _mm_sub_ps(lo, hi),
                                                _MM_SHUFFLE(1, 0, 1, 0)));
    }
    for (int len = 4; len < n; len <<= 1) {
        for (int i = 0; i < n; i += (len << 1)) {
            for (int j = 0; j < len; j += 4) {
                __m128 u = _mm_loadu_ps(&data[i + j]);
                __m128 v = _mm_loadu_ps(&data[i + j + len]);
                _mm_storeu_ps(&data[i + j],       _mm_add_ps(u, v));
                _mm_storeu_ps(&data[i + j + len], _mm_sub_ps(u, v));
            }
        }
    }
}
#endif

/* Round n down to the nearest power of 2 (n2 <= n, n2 is a power of 2). */
static inline int turbo_kv_pow2_floor(int n) {
    int n2 = 1;
    while ((n2 << 1) <= n) n2 <<= 1;
    return n2;
}

void turbo_kv_rht_forward(float * data, int n, uint32_t seed) {
    if (!data || n <= 0) return;
    const int n2 = turbo_kv_pow2_floor(n);

#ifdef __SSE4_1__
    /* Vectorized sign flip + scale: XOR sign bits, mul by 1/sqrt(n2) */
    if (n2 >= 4) {
        for (int i = 0; i < n2; i += 4) {
            __m128 d = _mm_loadu_ps(&data[i]);
            uint32_t smask[4] __attribute__((aligned(16)));
            for (int k = 0; k < 4; k++) {
                smask[k] = turbo_kv_random_sign(seed, i+k) < 0 ? 0x80000000u : 0;
            }
            _mm_storeu_ps(&data[i], _mm_xor_ps(d, _mm_load_ps((const float*)smask)));
        }
        walsh_hadamard_sse(data, n2);
        const __m128 vscale = _mm_set1_ps(1.0f / sqrtf((float) n2));
        for (int i = 0; i < n2; i += 4) {
            _mm_storeu_ps(&data[i], _mm_mul_ps(_mm_loadu_ps(&data[i]), vscale));
        }
        return;
    }
#endif
    for (int i = 0; i < n2; i++) {
        data[i] *= (float) turbo_kv_random_sign(seed, i);
    }
    walsh_hadamard(data, n2);
    const float scale = 1.0f / sqrtf((float) n2);
    for (int i = 0; i < n2; i++) {
        data[i] *= scale;
    }
}

void turbo_kv_rht_inverse(float * data, int n, uint32_t seed) {
    if (!data || n <= 0) return;
    const int n2 = turbo_kv_pow2_floor(n);

#ifdef __SSE4_1__
    if (n2 >= 4) {
        const __m128 vscale = _mm_set1_ps(1.0f / sqrtf((float) n2));
        for (int i = 0; i < n2; i += 4) {
            _mm_storeu_ps(&data[i], _mm_mul_ps(_mm_loadu_ps(&data[i]), vscale));
        }
        walsh_hadamard_sse(data, n2);
        for (int i = 0; i < n2; i += 4) {
            __m128 d = _mm_loadu_ps(&data[i]);
            uint32_t smask[4] __attribute__((aligned(16)));
            for (int k = 0; k < 4; k++) {
                smask[k] = turbo_kv_random_sign(seed, i+k) < 0 ? 0x80000000u : 0;
            }
            _mm_storeu_ps(&data[i], _mm_xor_ps(d, _mm_load_ps((const float*)smask)));
        }
        return;
    }
#endif
    const float scale = 1.0f / sqrtf((float) n2);
    for (int i = 0; i < n2; i++) {
        data[i] *= scale;
    }
    walsh_hadamard(data, n2);
    for (int i = 0; i < n2; i++) {
        data[i] *= (float) turbo_kv_random_sign(seed, i);
    }
}

/* ============================================================
 * fp16 / fp32 conversion helpers.
 *
 * We use ggml's existing FP16_TO_FP32 / FP32_TO_FP16 macros via
 * ggml-impl.h so we don't introduce a second conversion path.
 * ============================================================ */
static inline float turbo_kv_fp16_to_fp32(uint16_t h) {
    ggml_fp16_t h16;
    memcpy(&h16, &h, sizeof(h16));
    return GGML_FP16_TO_FP32(h16);
}

static inline uint16_t turbo_kv_fp32_to_fp16(float f) {
    const ggml_fp16_t r = GGML_FP32_TO_FP16(f);
    uint16_t out;
    memcpy(&out, &r, sizeof(out));
    return out;
}

/* ============================================================
 * Quantize one block (128 elements -> 72 bytes)
 * ============================================================ */
static void quantize_block_turbo_kv_4b(const float * src, block_turbo_kv_4b * block, int dim) {
    if (dim > TURBO_KV_BLOCK_SIZE) dim = TURBO_KV_BLOCK_SIZE;

    /* Step 1: L2 norm */
    float norm_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm_sq += src[i] * src[i];
    }
    const float norm = sqrtf(norm_sq);
    block->norm = turbo_kv_fp32_to_fp16(norm);
    block->residual_norm = 0;
    block->_pad = 0;

    /* Step 2: normalize to unit vector */
    float rotated[TURBO_KV_BLOCK_SIZE];
    const float inv_norm = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;
    for (int i = 0; i < dim; i++) {
        rotated[i] = src[i] * inv_norm;
    }
    for (int i = dim; i < TURBO_KV_BLOCK_SIZE; i++) {
        rotated[i] = 0.0f;
    }

    /* Step 3: RHT (in-place) */
    turbo_kv_rht_forward(rotated, dim, TURBO_KV_DEFAULT_SEED);

    /* Step 4: compute max_abs and per-block inv_std */
    float max_abs = 0.0f;
    for (int i = 0; i < dim; i++) {
        const float a = fabsf(rotated[i]);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-10f) max_abs = 1.0f;
    const float inv_std = TURBO_KV_4B_CENT_MAX / max_abs;
    block->inv_std_fp16 = turbo_kv_fp32_to_fp16(inv_std);

    /* Step 5: per-element nearest-centroid lookup after scaling by inv_std */
    memset(block->mse_indices, 0, TURBO_KV_BLOCK_SIZE / 2);
    for (int i = 0; i < dim; i++) {
        const float x = rotated[i] * inv_std;
        int best = 0;
        float best_dist = fabsf(x - turbo_kv_4b_codebook[0]);
        for (int c = 1; c < 16; c++) {
            const float d = fabsf(x - turbo_kv_4b_codebook[c]);
            if (d < best_dist) {
                best_dist = d;
                best = c;
            }
        }
        const int byte_idx = i / 2;
        const int bit_pos  = (i & 1) * 4;
        block->mse_indices[byte_idx] |= (uint8_t) ((best & 0x0F) << bit_pos);
    }
}

void quantize_row_turbo_kv_4b_ref(const float * x, block_turbo_kv_4b * y, int64_t k) {
    GGML_ASSERT(k % TURBO_KV_BLOCK_SIZE == 0);
    const int64_t nb = k / TURBO_KV_BLOCK_SIZE;
    for (int64_t b = 0; b < nb; b++) {
        quantize_block_turbo_kv_4b(x + b * TURBO_KV_BLOCK_SIZE, y + b, TURBO_KV_BLOCK_SIZE);
    }
}

/* ============================================================
 * Dequantize one block (72 bytes -> 128 elements)
 * ============================================================ */
static void dequantize_block_turbo_kv_4b(const block_turbo_kv_4b * block, float * dst, int dim) {
    if (dim > TURBO_KV_BLOCK_SIZE) dim = TURBO_KV_BLOCK_SIZE;

    const float norm = turbo_kv_fp16_to_fp32(block->norm);
    float inv_std = turbo_kv_fp16_to_fp32(block->inv_std_fp16);
    if (inv_std < 1e-10f) inv_std = sqrtf((float) dim);
    const float scale = 1.0f / inv_std;

    /* Precomputed per-block dequant LUT (16 entries, scaled) */
    float lut[16];
    for (int c = 0; c < 16; c++) {
        lut[c] = turbo_kv_4b_codebook[c] * scale;
    }

    /* Unpack + codebook lookup into rotated[] */
    float rotated[TURBO_KV_BLOCK_SIZE];
    for (int i = 0; i < dim; i++) {
        const int byte_idx = i / 2;
        const int idx = (i & 1) ? (block->mse_indices[byte_idx] >> 4)
                                : (block->mse_indices[byte_idx] & 0x0F);
        rotated[i] = lut[idx];
    }
    for (int i = dim; i < TURBO_KV_BLOCK_SIZE; i++) {
        rotated[i] = 0.0f;
    }

    /* Inverse RHT */
    turbo_kv_rht_inverse(rotated, dim, TURBO_KV_DEFAULT_SEED);

    /* Scale by original norm */
    for (int i = 0; i < dim; i++) {
        dst[i] = rotated[i] * norm;
    }
}

void dequantize_row_turbo_kv_4b(const block_turbo_kv_4b * x, float * y, int64_t k) {
    GGML_ASSERT(k % TURBO_KV_BLOCK_SIZE == 0);
    const int64_t nb = k / TURBO_KV_BLOCK_SIZE;
    for (int64_t b = 0; b < nb; b++) {
        dequantize_block_turbo_kv_4b(x + b, y + b * TURBO_KV_BLOCK_SIZE, TURBO_KV_BLOCK_SIZE);
    }
}

/* Forward declare the per-block dot helper (defined below, after rotate_query). */
static inline float turbo_kv_4b_single_block_dot(
    const block_turbo_kv_4b * block, const float * q_rot_block, int dim_in_block);

/* ============================================================
 * Batched attention: rotate Q once, loop over all K positions.
 *
 * Mirrors tq_kv_1b_attention_multi (ggml-turbo-quant.c:377). The FA path
 * in ops.cpp calls this once per thread with a slice of K. The non-FA
 * vec_dot wrapper calls it with valid_count=1 as a single-position
 * fallback.
 * ============================================================ */
void turbo_kv_4b_attention_multi(
    const float              * query,
    const block_turbo_kv_4b  * kv_cache,
    float                    * scores,
    int                        valid_count,
    int                        head_dim,
    int                        k_stride_blocks)
{
    const int n_blocks = head_dim / TURBO_KV_BLOCK_SIZE;
    if (n_blocks * TURBO_KV_BLOCK_SIZE != head_dim || n_blocks <= 0 || n_blocks > 4) {
        /* Unsupported head_dim; fall back to per-call vec_dot. */
        for (int s = 0; s < valid_count; s++) {
            const block_turbo_kv_4b * blocks = &kv_cache[s * (k_stride_blocks > 0 ? k_stride_blocks : n_blocks)];
            ggml_vec_dot_turbo_kv_4b_f32(
                head_dim, &scores[s], 0, blocks, 0, query, 0, 1);
        }
        return;
    }

    if (k_stride_blocks <= 0) k_stride_blocks = n_blocks;

    /* Step 1: Pre-rotate query ONCE (per-block RHT). */
    float q_rot[4 * TURBO_KV_BLOCK_SIZE];
    turbo_kv_rotate_query(query, q_rot, head_dim);

    /* Step 2: Loop over K positions — sum per-block dot products. */
    for (int s = 0; s < valid_count; s++) {
        const block_turbo_kv_4b * k_row = &kv_cache[s * k_stride_blocks];
        float total = 0.0f;

        for (int b = 0; b < n_blocks; b++) {
            total += turbo_kv_4b_single_block_dot(
                &k_row[b], q_rot + b * TURBO_KV_BLOCK_SIZE, TURBO_KV_BLOCK_SIZE);
        }

        scores[s] = total;
    }
}

/* ============================================================
 * Query pre-rotation (public helper)
 *
 * Applies a per-block RHT(q) into out. dim may exceed TURBO_KV_BLOCK_SIZE
 * (e.g. Qwen3.5 uses head_dim=256 which is 2 × 128 blocks). Each
 * TURBO_KV_BLOCK_SIZE-aligned chunk is rotated independently with the
 * same seed, matching how the K cache stores multi-block rows. The full
 * dot product is then the sum of per-block dots (see vec_dot below).
 *
 * The output buffer must have room for nb × TURBO_KV_BLOCK_SIZE floats
 * where nb = ceil(dim / TURBO_KV_BLOCK_SIZE). Any trailing slack in the
 * last block is zero-padded before rotation.
 * ============================================================ */
void turbo_kv_rotate_query(const float * q, float * out, int dim) {
    if (dim <= 0) return;

    /* Whole blocks */
    int d = 0;
    while (d + TURBO_KV_BLOCK_SIZE <= dim) {
        memcpy(out + d, q + d, (size_t) TURBO_KV_BLOCK_SIZE * sizeof(float));
        turbo_kv_rht_forward(out + d, TURBO_KV_BLOCK_SIZE, TURBO_KV_DEFAULT_SEED);
        d += TURBO_KV_BLOCK_SIZE;
    }

    /* Tail block (zero-padded) */
    if (d < dim) {
        memcpy(out + d, q + d, (size_t) (dim - d) * sizeof(float));
        for (int i = dim; i < d + TURBO_KV_BLOCK_SIZE; i++) {
            out[i] = 0.0f;
        }
        turbo_kv_rht_forward(out + d, TURBO_KV_BLOCK_SIZE, TURBO_KV_DEFAULT_SEED);
    }
}

/* ============================================================
 * Scalar reference vec_dot: <f32 query, turbo_kv_4b row of blocks>
 *
 * The query MUST already be RHT-rotated per-block by turbo_kv_rotate_query.
 * The K row is n elements total; nb = ceil(n / 128) turbo_kv_4b blocks
 * stored contiguously. Each block has its own norm and inv_std. Because
 * the RHT is per-block AND orthogonal, the full dot product is the sum
 * of per-block dots:
 *
 *     <q, k>  ==  sum_b  norm_b * <q_rot_b, dequant_rot_b(k_b)>
 *
 * where q_rot_b is q_rot[b*128 .. b*128+127] and k_b is block b of the
 * K row. Mathematically correct because both RHT_b and the block-norm
 * factoring are linear operations, and RHT preserves dot products
 * within each block.
 *
 * n is the head dimension (e.g. 256 for Qwen3.5 attention heads, which
 * maps to 2 blocks per row).
 * ============================================================ */
static inline float turbo_kv_4b_single_block_dot(
    const block_turbo_kv_4b * block,
    const float * q_rot_block,
    int dim_in_block)
{
    const float norm = turbo_kv_fp16_to_fp32(block->norm);
    float inv_std = turbo_kv_fp16_to_fp32(block->inv_std_fp16);
    if (inv_std < 1e-10f) inv_std = sqrtf((float) dim_in_block);
    const float scale = 1.0f / inv_std;

    /* Pre-scaled 16-entry LUT (fused dequant + per-block scale) */
    float lut[16];
    for (int c = 0; c < 16; c++) {
        lut[c] = turbo_kv_4b_codebook[c] * scale;
    }

    const uint8_t * mi = block->mse_indices;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    int d = 0;
    for (; d + 7 < dim_in_block; d += 8) {
        const uint8_t b0 = mi[(d + 0) / 2];
        const uint8_t b1 = mi[(d + 2) / 2];
        const uint8_t b2 = mi[(d + 4) / 2];
        const uint8_t b3 = mi[(d + 6) / 2];
        a0 += q_rot_block[d + 0] * lut[b0 & 0x0F];
        a1 += q_rot_block[d + 1] * lut[b0 >> 4];
        a2 += q_rot_block[d + 2] * lut[b1 & 0x0F];
        a3 += q_rot_block[d + 3] * lut[b1 >> 4];
        a0 += q_rot_block[d + 4] * lut[b2 & 0x0F];
        a1 += q_rot_block[d + 5] * lut[b2 >> 4];
        a2 += q_rot_block[d + 6] * lut[b3 & 0x0F];
        a3 += q_rot_block[d + 7] * lut[b3 >> 4];
    }
    float mse_dot = (a0 + a1) + (a2 + a3);
    for (; d < dim_in_block; d++) {
        const uint8_t bv = mi[d / 2];
        const int idx = (d & 1) ? (bv >> 4) : (bv & 0x0F);
        mse_dot += q_rot_block[d] * lut[idx];
    }

    return norm * mse_dot;
}

void ggml_vec_dot_turbo_kv_4b_f32(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by,
    int nrc)
{
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    /* vy is raw f32 from ggml. We rotate per call since the current
     * TURBO_KV_4B vec_dot_type is F32 (hoisting via TURBO_KV_Q_ROT_F32
     * was tried but regressed — see the comment in ggml-cpu.c on the
     * CPU trait registration). */
    const float * q_raw = (const float *) vy;
    const block_turbo_kv_4b * blocks = (const block_turbo_kv_4b *) vx;

    float q_rot[4 * TURBO_KV_BLOCK_SIZE];
    int rot_n = n;
    if (rot_n > 4 * TURBO_KV_BLOCK_SIZE) {
        rot_n = 4 * TURBO_KV_BLOCK_SIZE;
    }
    turbo_kv_rotate_query(q_raw, q_rot, rot_n);

    float total = 0.0f;
    int d = 0;
    int b = 0;
    while (d + TURBO_KV_BLOCK_SIZE <= rot_n) {
        total += turbo_kv_4b_single_block_dot(
            &blocks[b], q_rot + d, TURBO_KV_BLOCK_SIZE);
        d += TURBO_KV_BLOCK_SIZE;
        b++;
    }
    /* Tail block (partial) */
    if (d < rot_n) {
        total += turbo_kv_4b_single_block_dot(
            &blocks[b], q_rot + d, rot_n - d);
    }

    *s = total;
}

/* turbo_kv_precompute_rope_cache and turbo_kv_4b_attention_fused_rope
 * removed — superseded by split K cache (--cache-type-k q8_0:q4_0). */
