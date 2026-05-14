// fattn-per-slot-kv-sm75-reference.h
//
// Header-only scalar fp32 reference implementation of
// `fattn_per_slot_kv_sm75` (specs/deltanet/fattn-per-slot-kv-sm75.md).
//
// Purpose: ground-truth oracle for kernel byte-identity (or
// ≤ 1 fp16 ULP) binding. Mirrors the kernel's compute structure:
//
//   - Per-slot K-loop bound (uses slot_seq_lens)
//   - KV_BLOCK_SIZE-chunked iteration
//   - Per-(slot, head, parallel-block) online softmax with fp32 accumulators
//   - Final normalize + cast to half
//   - Separate combine across parallel_blocks via flash_attn_combine_results-equivalent
//
// Two binding modes:
//   STRICT  — bit-equality of the final output halfs (requires matching
//             the kernel's mma.sync.m16n8k8 internal reduction order;
//             not always achievable for a scalar reference)
//   COSINE  — cosine ≥ 0.9999 AND NMSE ≤ 1e-4 (default; the kernel's
//             output and the reference's output should agree at fp16
//             precision per the closure binding)
//
// Both modes are checked simultaneously per the DFlash T3 precedent.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <limits>

namespace fattn_per_slot_kv_sm75 {

// Half precision helpers (host-side fp16 conversion via uint16 bit-cast).
struct Half {
    uint16_t bits;
};

static inline float half_to_float(Half h) {
    // Standard IEEE 754 fp16 → fp32 conversion.
    uint32_t x = h.bits;
    uint32_t sign = (x >> 15) & 0x1;
    uint32_t exp_ = (x >> 10) & 0x1f;
    uint32_t mant = x & 0x3ff;
    uint32_t fbits;
    if (exp_ == 0) {
        if (mant == 0) {
            fbits = sign << 31;
        } else {
            // Subnormal — normalise.
            int e = 0;
            while ((mant & 0x400) == 0) { mant <<= 1; e++; }
            mant &= 0x3ff;
            uint32_t exp32 = (127 - 15 - e + 1);
            fbits = (sign << 31) | (exp32 << 23) | (mant << 13);
        }
    } else if (exp_ == 0x1f) {
        // Inf or NaN.
        fbits = (sign << 31) | (0xff << 23) | (mant << 13);
    } else {
        uint32_t exp32 = exp_ - 15 + 127;
        fbits = (sign << 31) | (exp32 << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &fbits, sizeof(f));
    return f;
}

static inline Half float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 31) & 0x1;
    int32_t  exp_ = ((x >> 23) & 0xff) - 127;
    uint32_t mant = x & 0x7fffff;
    uint16_t bits;
    if (exp_ > 15) {
        // Overflow → inf (preserving sign).
        bits = (sign << 15) | (0x1f << 10);
    } else if (exp_ < -14) {
        // Underflow → subnormal or zero.
        if (exp_ < -25) {
            bits = sign << 15;
        } else {
            // Subnormal — round half to even.
            mant |= 0x800000; // hidden bit
            uint32_t shift = (uint32_t)(-14 - exp_) + 13;
            uint32_t round_bit = (mant >> (shift - 1)) & 1;
            uint32_t sticky = (mant & ((1u << (shift - 1)) - 1)) != 0;
            uint32_t mant_h = mant >> shift;
            if (round_bit && (sticky || (mant_h & 1))) mant_h++;
            bits = (sign << 15) | (uint16_t)mant_h;
        }
    } else {
        // Normal — round half to even.
        uint32_t round_bit = (mant >> 12) & 1;
        uint32_t sticky    = (mant & 0xfff) != 0;
        uint32_t mant_h    = mant >> 13;
        if (round_bit && (sticky || (mant_h & 1))) {
            mant_h++;
            if (mant_h == 0x400) {
                mant_h = 0;
                exp_++;
                if (exp_ > 15) {
                    bits = (sign << 15) | (0x1f << 10);
                    return Half{bits};
                }
            }
        }
        bits = (sign << 15) | (((uint16_t)(exp_ + 15)) << 10) | (uint16_t)mant_h;
    }
    return Half{bits};
}

// Inputs/outputs use the same layout as the kernel signature in §3 of
// the spec:
//
//   Q:               [HEAD_DIM_Q, n_tokens, n_heads_q, n_seqs]   (column-major)
//   K_cache:         [HEAD_DIM_Q, n_kv_max,  n_kv_heads, n_seqs]
//   V_cache:         [HEAD_DIM_V, n_kv_max,  n_kv_heads, n_seqs]
//   mask:            [n_kv_max, n_tokens]                         (per-Q-row mask)
//   slot_seq_lens:   [n_seqs]                                     (per-slot valid n_kv)
//   dst_partial:     [HEAD_DIM_V, parallel_blocks, n_tokens, n_heads_q, n_seqs]
//   dst_meta:        [2, parallel_blocks, n_tokens, n_heads_q, n_seqs]   (float2 = max, rowsum)
//
// HEAD_DIM_Q and HEAD_DIM_V are compile-time constants for the production
// (256, 128) tuple per OQ-4. Other tuples are not implemented in this oracle.

struct AttnConfig {
    int head_dim_q;          // 256
    int head_dim_v;          // 128
    int kv_block_size;       // 32 or 64
    int n_tokens;
    int n_heads_q;
    int n_kv_heads;          // gqa: n_heads_q / n_kv_heads gives gqa_ratio
    int n_seqs;
    int n_kv_max;            // K cache's ne[1]
    int parallel_blocks;     // global parallel_blocks (kernel uses per-slot)
    float scale;
    float softcap;           // 0 = no softcap
    bool use_softcap;
};

// Compute reference output for one (slot, query_row, head) tuple at a specific parallel-block index `ip`.
// Returns max, rowsum, and writes HEAD_DIM_V floats to `out_partial`.
static inline void reference_one_cta(
    const AttnConfig & cfg,
    const Half * Q,
    const Half * K_cache,
    const Half * V_cache,
    const Half * mask,
    const int32_t * slot_seq_lens,
    int slot,
    int q_row,
    int head,
    int ip,
    int parallel_blocks_for_slot,
    float * out_partial,         // [head_dim_v]
    float * out_max,
    float * out_rowsum
) {
    const int Dq = cfg.head_dim_q;
    const int Dv = cfg.head_dim_v;
    const int KVB = cfg.kv_block_size;
    const int gqa = cfg.n_heads_q / cfg.n_kv_heads;
    const int kv_head = head / gqa;
    const int n_kv_slot = slot_seq_lens[slot];

    // Per-slot K range for this CTA.
    const int k_chunk_size = (n_kv_slot + parallel_blocks_for_slot - 1) / parallel_blocks_for_slot;
    const int k_start = ip * k_chunk_size;
    const int k_end   = std::min((ip + 1) * k_chunk_size, n_kv_slot);

    if (k_start >= n_kv_slot) {
        *out_max = -std::numeric_limits<float>::infinity();
        *out_rowsum = 0.0f;
        std::memset(out_partial, 0, Dv * sizeof(float));
        return;
    }

    // Q row pointer.
    auto Q_off = [&](int d) {
        // Column-major: ne[0]=Dq, ne[1]=n_tokens, ne[2]=n_heads_q, ne[3]=n_seqs
        return (((size_t)slot * cfg.n_heads_q + head) * cfg.n_tokens + q_row) * Dq + d;
    };
    auto K_off = [&](int d, int k) {
        // [Dq, n_kv_max, n_kv_heads, n_seqs]
        return (((size_t)slot * cfg.n_kv_heads + kv_head) * cfg.n_kv_max + k) * Dq + d;
    };
    auto V_off = [&](int d, int k) {
        // [Dv, n_kv_max, n_kv_heads, n_seqs]
        return (((size_t)slot * cfg.n_kv_heads + kv_head) * cfg.n_kv_max + k) * Dv + d;
    };
    auto mask_off = [&](int k) {
        return (size_t)q_row * cfg.n_kv_max + k;
    };

    // Online-softmax state: per-Q-row max, rowsum, VKQ accumulator (fp32).
    float kqmax = -std::numeric_limits<float>::infinity();
    float kqrowsum = 0.0f;
    std::vector<float> VKQ(Dv, 0.0f);

    // Process K in KV_BLOCK_SIZE chunks.
    for (int kb = k_start; kb < k_end; kb += KVB) {
        const int kb_end = std::min(kb + KVB, k_end);

        // Compute KQ row: KQ[k] = scale * sum_d (Q[d] * K[d, k]).
        std::vector<float> KQ(KVB, -std::numeric_limits<float>::infinity());
        for (int k = kb; k < kb_end; k++) {
            float acc = 0.0f;
            for (int d = 0; d < Dq; d++) {
                acc += half_to_float(Q[Q_off(d)]) * half_to_float(K_cache[K_off(d, k)]);
            }
            acc *= cfg.scale;
            if (cfg.use_softcap) {
                acc = cfg.softcap * std::tanh(acc / cfg.softcap);
            }
            acc += half_to_float(mask[mask_off(k)]);  // alibi/padding mask
            KQ[k - kb] = acc;
        }

        // Find new max over this chunk + running max.
        float new_max = kqmax;
        for (int k = kb; k < kb_end; k++) {
            if (KQ[k - kb] > new_max) new_max = KQ[k - kb];
        }
        if (new_max == -std::numeric_limits<float>::infinity()) {
            // All masked; nothing to add.
            continue;
        }

        // Scale running state by exp(diff) of old max vs new max.
        const float scale_factor = (kqmax == -std::numeric_limits<float>::infinity())
                                 ? 0.0f
                                 : std::exp(kqmax - new_max);
        kqrowsum *= scale_factor;
        for (int d = 0; d < Dv; d++) {
            VKQ[d] *= scale_factor;
        }
        kqmax = new_max;

        // Softmax weights for this chunk.
        std::vector<float> SM(KVB, 0.0f);
        for (int k = kb; k < kb_end; k++) {
            SM[k - kb] = std::exp(KQ[k - kb] - new_max);
            kqrowsum += SM[k - kb];
        }

        // Accumulate V contribution.
        for (int k = kb; k < kb_end; k++) {
            const float sm_k = SM[k - kb];
            if (sm_k == 0.0f) continue;
            for (int d = 0; d < Dv; d++) {
                VKQ[d] += sm_k * half_to_float(V_cache[V_off(d, k)]);
            }
        }
    }

    // Write partial result (un-normalised; combine kernel finishes).
    *out_max = kqmax;
    *out_rowsum = kqrowsum;
    for (int d = 0; d < Dv; d++) {
        out_partial[d] = VKQ[d];
    }
}

// Compute the full reference output across all (slot, q_row, head, ip)
// CTAs, then run the combine step to produce the final per-(slot, q_row,
// head, d) output. Output layout matches what flash_attn_combine_results
// would write: out[Dv, n_tokens, n_heads_q, n_seqs].
//
// `parallel_blocks_per_slot` is an array of length n_seqs.
static inline void reference_full(
    const AttnConfig & cfg,
    const Half * Q,
    const Half * K_cache,
    const Half * V_cache,
    const Half * mask,
    const int32_t * slot_seq_lens,
    const int32_t * parallel_blocks_per_slot,
    float * out_final  // [Dv, n_tokens, n_heads_q, n_seqs]  fp32
) {
    const int Dv = cfg.head_dim_v;

    // Storage for partials and meta across all CTAs.
    std::vector<float> dst_partial;
    std::vector<float> dst_max;
    std::vector<float> dst_rowsum;

    auto cta_index = [&](int slot, int q_row, int head, int ip) {
        const int pb = parallel_blocks_per_slot[slot];
        return ((slot * cfg.n_heads_q + head) * cfg.n_tokens + q_row) * pb + ip;
    };

    // Pre-size — assumes uniform parallel_blocks for simplicity here.
    // In reality each slot may have a different pb; we size by max.
    int max_pb = 1;
    for (int s = 0; s < cfg.n_seqs; s++) {
        if (parallel_blocks_per_slot[s] > max_pb) max_pb = parallel_blocks_per_slot[s];
    }
    const size_t n_cta_max = (size_t)cfg.n_seqs * cfg.n_heads_q * cfg.n_tokens * max_pb;
    dst_partial.assign(n_cta_max * Dv, 0.0f);
    dst_max.assign(n_cta_max, -std::numeric_limits<float>::infinity());
    dst_rowsum.assign(n_cta_max, 0.0f);

    // Phase 1: compute each CTA's partial.
    for (int slot = 0; slot < cfg.n_seqs; slot++) {
        const int pb = parallel_blocks_per_slot[slot];
        for (int head = 0; head < cfg.n_heads_q; head++) {
            for (int q_row = 0; q_row < cfg.n_tokens; q_row++) {
                for (int ip = 0; ip < pb; ip++) {
                    const size_t idx = ((size_t)slot * cfg.n_heads_q + head) * cfg.n_tokens + q_row;
                    const size_t cta_idx = idx * max_pb + ip;
                    reference_one_cta(
                        cfg, Q, K_cache, V_cache, mask, slot_seq_lens,
                        slot, q_row, head, ip, pb,
                        dst_partial.data() + cta_idx * Dv,
                        &dst_max[cta_idx],
                        &dst_rowsum[cta_idx]
                    );
                }
            }
        }
    }

    // Phase 2: combine across parallel_blocks per (slot, q_row, head).
    // Mirrors flash_attn_combine_results: iterate ip = 0..pb-1 in fixed order.
    for (int slot = 0; slot < cfg.n_seqs; slot++) {
        const int pb = parallel_blocks_per_slot[slot];
        for (int head = 0; head < cfg.n_heads_q; head++) {
            for (int q_row = 0; q_row < cfg.n_tokens; q_row++) {
                const size_t base = ((size_t)slot * cfg.n_heads_q + head) * cfg.n_tokens + q_row;

                // Find combined max.
                float kqmax = -std::numeric_limits<float>::infinity();
                for (int ip = 0; ip < pb; ip++) {
                    const float m = dst_max[base * max_pb + ip];
                    if (m > kqmax) kqmax = m;
                }

                // Combine.
                std::vector<float> num(Dv, 0.0f);
                float den = 0.0f;
                for (int ip = 0; ip < pb; ip++) {
                    const float m = dst_max[base * max_pb + ip];
                    const float sum = dst_rowsum[base * max_pb + ip];
                    if (m == -std::numeric_limits<float>::infinity()) continue;
                    const float w = std::exp(m - kqmax);
                    for (int d = 0; d < Dv; d++) {
                        num[d] += w * dst_partial[(base * max_pb + ip) * Dv + d];
                    }
                    den += w * sum;
                }

                if (den == 0.0f) {
                    for (int d = 0; d < Dv; d++) {
                        out_final[base * Dv + d] = 0.0f;
                    }
                } else {
                    for (int d = 0; d < Dv; d++) {
                        out_final[base * Dv + d] = num[d] / den;
                    }
                }
            }
        }
    }
}

// Helpers used by the unit test for tolerance binding.

struct CompareResult {
    int n_total;
    int n_exact_byte_match;     // ULP-exact at fp16 storage
    int n_close;                 // |diff| / max(|a|,|b|,1e-6) ≤ 1e-3
    float max_abs_diff;
    float mean_abs_diff;
    float cosine;
    float nmse;
};

static inline CompareResult compare(const float * a, const float * b, size_t n) {
    CompareResult r{};
    r.n_total = (int)n;
    double sum_a2 = 0.0, sum_b2 = 0.0, sum_ab = 0.0, sum_diff2 = 0.0;
    double sum_diff_abs = 0.0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float da = a[i], db = b[i];
        const float diff = std::fabs(da - db);
        if (diff == 0.0f) r.n_exact_byte_match++;
        const float scale = std::max({std::fabs(da), std::fabs(db), 1e-6f});
        if (diff / scale <= 1e-3f) r.n_close++;
        if (diff > max_diff) max_diff = diff;
        sum_diff_abs += diff;
        sum_a2 += (double)da * da;
        sum_b2 += (double)db * db;
        sum_ab += (double)da * db;
        sum_diff2 += (double)diff * diff;
    }
    r.max_abs_diff = max_diff;
    r.mean_abs_diff = (float)(sum_diff_abs / std::max((size_t)1, n));
    const double denom = std::sqrt(sum_a2 * sum_b2);
    r.cosine = denom > 0.0 ? (float)(sum_ab / denom) : 1.0f;
    r.nmse = (float)(sum_diff2 / std::max(1e-12, sum_a2));
    return r;
}

}  // namespace fattn_per_slot_kv_sm75
