// test-fattn-per-slot-kv-split-sm75.cpp
//
// PHASE_FATTN_KV_SPLIT_SM75 step 3 — unit tests for the KV-split decode
// kernels (flash_attn_per_slot_kv_split_partial + _merge) against an f64
// scalar reference, plus the determinism/invariance claims the kernel's
// header makes:
//
//   Scenario A: f64-reference accuracy (q4_0 + f16 KV) — cosine ≥ 0.9999,
//               NMSE ≤ 1e-4 (binding per the legacy test's closure §7).
//   Scenario B: rep-determinism — two identical launches, byte-identical.
//   Scenario C: padding / chunk-count invariance — same logical KV content
//               and valid length L, ne11 padded by +1024 and +2048 (extra
//               fully-masked chunks). Byte-identical ⇒ the merge's −INF
//               fold really is a bit-exact no-op (straddle L values bind
//               the CHUNK=1024 and BLOCK=64 boundaries and the ILP-4 tail).
//   Scenario D: block-table permutation invariance — same logical KV,
//               physically permuted pool blocks. Byte-identical ⇒ physical
//               placement cannot affect the fp32 chain (the TRACE-6 class
//               of hazard is structurally gone under paged addressing).
//   Scenario E: batch invariance — a slot extracted from a mixed batch
//               (different n_seqs, different ne11, different grid) must
//               produce byte-identical output when run alone.
//
// Env (optional):
//   FATTN_TEST_QUICK=1     — abbreviated sweep
//   FATTN_TEST_FAIL_FAST=1 — bail on first failure
//   FATTN_TEST_SEED=<int>  — override base seed
//
// Exit: 0 = pass, 1 = fail.

#include "fattn-per-slot-kv-sm75-reference.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>

using fattn_per_slot_kv_sm75::Half;
using fattn_per_slot_kv_sm75::half_to_float;
using fattn_per_slot_kv_sm75::float_to_half;
using fattn_per_slot_kv_sm75::CompareResult;
using fattn_per_slot_kv_sm75::compare;

// Implemented in ggml/src/ggml-cuda/fattn-per-slot-kv-split-sm75.cu.
extern "C" int fattn_per_slot_kv_split_sm75_test_launch(
        const float    * Q_h,
        const void     * K_pool_h,
        const void     * V_pool_h,
        const uint16_t * mask_h,
        const int32_t  * block_table_h,
        int n_blocks_pool,
        int n_blocks_per_seq,
        int n_heads_q,
        int n_kv_heads,
        int n_seqs,
        int ne11,
        int kv_is_q4_0,
        float scale,
        float * dst_h);

// ---------------------------------------------------------------------------
// Paged-pool layout constants (must match the kernel's kv_bases lambda).
// ---------------------------------------------------------------------------

static constexpr int D          = 256;  // Dk == Dv
static constexpr int BLOCK_TOK  = 64;   // paged block size (tokens)
static constexpr int CHUNK_TOK  = 1024; // PSKV_SPLIT_CHUNK_TOKENS

struct BlockQ4_0 {                      // host mirror of ggml block_q4_0
    uint16_t d;                         // fp16 scale (bits)
    uint8_t  qs[16];                    // 32 packed nibbles
};
static_assert(sizeof(BlockQ4_0) == 18, "block_q4_0 host mirror must be 18 bytes");

struct PagedPool {
    bool                 q4_0;
    int                  n_kv_heads;
    int                  n_blocks;
    size_t               nb11;        // bytes per row
    size_t               paged_nb12;  // bytes per (block, kv-head) slab
    size_t               paged_nb13;  // bytes per block
    std::vector<uint8_t> bytes;

    void init(bool q4, int n_kvh, int n_blk) {
        q4_0       = q4;
        n_kv_heads = n_kvh;
        n_blocks   = n_blk;
        nb11       = q4 ? (size_t)(D / 32) * sizeof(BlockQ4_0) : (size_t)D * 2;
        paged_nb12 = nb11 * BLOCK_TOK;
        paged_nb13 = paged_nb12 * n_kvh;
        bytes.assign((size_t)n_blk * paged_nb13, 0);
    }
    uint8_t       * row(int bid, int kvh, int off)       { return bytes.data() + (size_t)bid*paged_nb13 + (size_t)kvh*paged_nb12 + (size_t)off*nb11; }
    const uint8_t * row(int bid, int kvh, int off) const { return bytes.data() + (size_t)bid*paged_nb13 + (size_t)kvh*paged_nb12 + (size_t)off*nb11; }

    void fill_random(std::mt19937 & rng) {
        if (q4_0) {
            std::uniform_int_distribution<int> byte_dist(0, 255);
            std::uniform_real_distribution<float> d_dist(0.002f, 0.04f);
            for (int b = 0; b < n_blocks; b++) {
                for (int h = 0; h < n_kv_heads; h++) {
                    for (int t = 0; t < BLOCK_TOK; t++) {
                        BlockQ4_0 * blks = (BlockQ4_0 *) row(b, h, t);
                        for (int j = 0; j < D/32; j++) {
                            blks[j].d = float_to_half(d_dist(rng)).bits;
                            for (int q = 0; q < 16; q++) blks[j].qs[q] = (uint8_t) byte_dist(rng);
                        }
                    }
                }
            }
        } else {
            std::uniform_real_distribution<float> v_dist(-1.0f, 1.0f);
            for (int b = 0; b < n_blocks; b++) {
                for (int h = 0; h < n_kv_heads; h++) {
                    for (int t = 0; t < BLOCK_TOK; t++) {
                        uint16_t * r = (uint16_t *) row(b, h, t);
                        for (int d = 0; d < D; d++) r[d] = float_to_half(v_dist(rng)).bits;
                    }
                }
            }
        }
    }

    // Logical element (k, d) for a sequence with the given block table, in f64.
    double dequant(const int32_t * bt, int kvh, int k, int d) const {
        const int bid = bt[k / BLOCK_TOK];
        const int off = k % BLOCK_TOK;
        const uint8_t * r = row(bid, kvh, off);
        if (q4_0) {
            const BlockQ4_0 * blk = (const BlockQ4_0 *) r + d / 32;
            const int o32   = d % 32;
            const int byte  = blk->qs[o32 & 15];
            const int shift = o32 / 16;
            const int nib   = (byte >> (shift * 4)) & 0xF;
            Half hd; hd.bits = blk->d;
            return (double) half_to_float(hd) * (double)(nib - 8);
        }
        Half hv; hv.bits = ((const uint16_t *) r)[d];
        return (double) half_to_float(hv);
    }
};

// ---------------------------------------------------------------------------
// Test configuration: one batch (Q, paged K/V, mask, block tables).
// ---------------------------------------------------------------------------

struct SplitConfig {
    bool q4_0;
    int  n_heads_q;
    int  n_kv_heads;
    int  n_seqs;
    int  ne11;                       // padded logical KV length (global)
    std::vector<int> seq_len;        // valid length per seq (≤ ne11)
    int  n_blocks_per_seq;

    PagedPool             K, V;
    std::vector<float>    Q;         // [256 * n_heads_q * n_seqs]
    std::vector<uint16_t> mask;      // f16 bits [ne11 * n_seqs]
    std::vector<int32_t>  bt;        // [n_seqs * n_blocks_per_seq]

    float scale() const { return 1.0f / std::sqrt((float) D); }

    void build(std::mt19937 & rng) {
        n_blocks_per_seq = (ne11 + BLOCK_TOK - 1) / BLOCK_TOK;
        const int n_blocks_pool = n_seqs * n_blocks_per_seq;
        K.init(q4_0, n_kv_heads, n_blocks_pool);
        V.init(q4_0, n_kv_heads, n_blocks_pool);
        K.fill_random(rng);
        V.fill_random(rng);

        Q.resize((size_t)D * n_heads_q * n_seqs);
        std::uniform_real_distribution<float> q_dist(-1.0f, 1.0f);
        for (float & v : Q) v = q_dist(rng);

        const uint16_t NEG_INF_F16 = 0xFC00;
        mask.assign((size_t)ne11 * n_seqs, NEG_INF_F16);
        for (int s = 0; s < n_seqs; s++) {
            for (int k = 0; k < seq_len[s]; k++) {
                mask[(size_t)s*ne11 + k] = 0; // visible (decode: all history)
            }
        }

        bt.resize((size_t)n_seqs * n_blocks_per_seq);
        for (int s = 0; s < n_seqs; s++) {
            for (int b = 0; b < n_blocks_per_seq; b++) {
                bt[(size_t)s*n_blocks_per_seq + b] = s*n_blocks_per_seq + b;
            }
        }
    }

    int run(std::vector<float> & out) const {
        out.assign((size_t)D * n_heads_q * n_seqs, 0.0f);
        return fattn_per_slot_kv_split_sm75_test_launch(
            Q.data(), K.bytes.data(), V.bytes.data(), mask.data(), bt.data(),
            K.n_blocks, n_blocks_per_seq, n_heads_q, n_kv_heads, n_seqs,
            ne11, q4_0 ? 1 : 0, scale(), out.data());
    }
};

// ---------------------------------------------------------------------------
// f64 scalar reference (full two-pass softmax, masked positions excluded).
// ---------------------------------------------------------------------------

static void reference_f64(const SplitConfig & c, std::vector<float> & out) {
    out.assign((size_t)D * c.n_heads_q * c.n_seqs, 0.0f);
    const int gqa = c.n_heads_q / c.n_kv_heads;
    const double scale = (double) c.scale();

    std::vector<double> s(c.ne11);
    for (int seq = 0; seq < c.n_seqs; seq++) {
        const int32_t * bt_seq = c.bt.data() + (size_t)seq * c.n_blocks_per_seq;
        for (int h = 0; h < c.n_heads_q; h++) {
            const int kvh = h / gqa;
            const float * Qh = c.Q.data() + ((size_t)seq*c.n_heads_q + h) * D;

            double m = -1e300;
            for (int k = 0; k < c.ne11; k++) {
                Half mk; mk.bits = c.mask[(size_t)seq*c.ne11 + k];
                const float mf = half_to_float(mk);
                if (std::isinf(mf) && mf < 0) { s[k] = -1e300; continue; }
                double dot = 0.0;
                for (int d = 0; d < D; d++) {
                    dot += c.K.dequant(bt_seq, kvh, k, d) * (double) Qh[d];
                }
                s[k] = scale * dot + (double) mf;
                if (s[k] > m) m = s[k];
            }
            double l = 0.0;
            std::vector<double> acc(D, 0.0);
            for (int k = 0; k < c.ne11; k++) {
                if (s[k] <= -1e299) continue;
                const double p = std::exp(s[k] - m);
                l += p;
                for (int d = 0; d < D; d++) {
                    acc[d] += p * c.V.dequant(bt_seq, kvh, k, d);
                }
            }
            float * o = out.data() + ((size_t)seq*c.n_heads_q + h) * D;
            for (int d = 0; d < D; d++) o[d] = (float)(acc[d] / l);
        }
    }
}

// ---------------------------------------------------------------------------
// Scenario helpers
// ---------------------------------------------------------------------------

static int g_fail = 0;
static bool g_fail_fast = false;

static bool report(const char * scen, const std::string & label, bool ok, const char * extra = "") {
    if (!ok) {
        fprintf(stderr, "[split-sm75] FAIL %s %s %s\n", scen, label.c_str(), extra);
        g_fail++;
    }
    return ok;
}

static bool bytes_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

// A: f64 reference; B: rep-determinism.
static void run_accuracy_and_rep(const SplitConfig & c, const std::string & label) {
    std::vector<float> out1, out2, ref;
    if (!report("launch", label, c.run(out1) == 0)) return;
    if (!report("launch", label, c.run(out2) == 0)) return;

    report("B:rep_determinism", label, bytes_equal(out1, out2));

    reference_f64(c, ref);
    const CompareResult cmp = compare(ref.data(), out1.data(), out1.size());
    char extra[160];
    snprintf(extra, sizeof(extra), "(cosine=%.7f nmse=%g max_abs=%g)",
             cmp.cosine, cmp.nmse, cmp.max_abs_diff);
    report("A:f64_reference", label, cmp.cosine >= 0.9999f && cmp.nmse <= 1e-4f, extra);
}

// C: padding/chunk-count invariance: extend ne11 by `pad` fully-masked tokens.
static void run_padding_invariance(const SplitConfig & c, int pad, const std::string & label) {
    std::vector<float> base;
    if (!report("launch", label, c.run(base) == 0)) return;

    SplitConfig p = c;             // copies pools, Q, mask, bt
    p.ne11 = c.ne11 + pad;
    p.n_blocks_per_seq = (p.ne11 + BLOCK_TOK - 1) / BLOCK_TOK;

    const uint16_t NEG_INF_F16 = 0xFC00;
    p.mask.assign((size_t)p.ne11 * p.n_seqs, NEG_INF_F16);
    for (int s = 0; s < p.n_seqs; s++) {
        std::memcpy(p.mask.data() + (size_t)s*p.ne11,
                    c.mask.data() + (size_t)s*c.ne11, (size_t)c.ne11 * 2);
    }
    // Extended block-table entries point at block 0 — content is irrelevant
    // (fully masked) but the address must be a valid pool block.
    p.bt.assign((size_t)p.n_seqs * p.n_blocks_per_seq, 0);
    for (int s = 0; s < p.n_seqs; s++) {
        std::memcpy(p.bt.data() + (size_t)s*p.n_blocks_per_seq,
                    c.bt.data() + (size_t)s*c.n_blocks_per_seq,
                    (size_t)c.n_blocks_per_seq * sizeof(int32_t));
    }

    std::vector<float> padded;
    if (!report("launch", label, p.run(padded) == 0)) return;
    char extra[64];
    snprintf(extra, sizeof(extra), "(pad=+%d, n_chunks %d->%d)", pad,
             (c.ne11 + CHUNK_TOK - 1)/CHUNK_TOK, (p.ne11 + CHUNK_TOK - 1)/CHUNK_TOK);
    report("C:padding_invariance", label, bytes_equal(base, padded), extra);
}

// D: physical block permutation invariance.
static void run_permutation_invariance(const SplitConfig & c, std::mt19937 & rng, const std::string & label) {
    std::vector<float> base;
    if (!report("launch", label, c.run(base) == 0)) return;

    std::vector<int> perm(c.K.n_blocks);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    SplitConfig p = c;
    for (int b = 0; b < c.K.n_blocks; b++) {
        std::memcpy(p.K.bytes.data() + (size_t)perm[b]*c.K.paged_nb13,
                    c.K.bytes.data() + (size_t)b*c.K.paged_nb13, c.K.paged_nb13);
        std::memcpy(p.V.bytes.data() + (size_t)perm[b]*c.V.paged_nb13,
                    c.V.bytes.data() + (size_t)b*c.V.paged_nb13, c.V.paged_nb13);
    }
    for (int32_t & e : p.bt) e = perm[e];

    std::vector<float> shuffled;
    if (!report("launch", label, p.run(shuffled) == 0)) return;
    report("D:block_permutation", label, bytes_equal(base, shuffled));
}

// E: batch invariance — each slot of a mixed batch, re-run alone.
static void run_batch_invariance(const SplitConfig & c, const std::string & label) {
    std::vector<float> batch;
    if (!report("launch", label, c.run(batch) == 0)) return;

    for (int s = 0; s < c.n_seqs; s++) {
        SplitConfig solo;
        solo.q4_0       = c.q4_0;
        solo.n_heads_q  = c.n_heads_q;
        solo.n_kv_heads = c.n_kv_heads;
        solo.n_seqs     = 1;
        solo.seq_len    = { c.seq_len[s] };
        solo.ne11       = ((c.seq_len[s] + BLOCK_TOK - 1) / BLOCK_TOK) * BLOCK_TOK;
        solo.n_blocks_per_seq = solo.ne11 / BLOCK_TOK;

        // Reuse the batch pools verbatim; the solo block table is the prefix
        // of seq s's table (physical placement deliberately unchanged —
        // Scenario D covers placement).
        solo.K = c.K;
        solo.V = c.V;
        solo.Q.assign(c.Q.begin() + (size_t)s*c.n_heads_q*D,
                      c.Q.begin() + (size_t)(s+1)*c.n_heads_q*D);
        const uint16_t NEG_INF_F16 = 0xFC00;
        solo.mask.assign((size_t)solo.ne11, NEG_INF_F16);
        for (int k = 0; k < c.seq_len[s]; k++) solo.mask[k] = 0;
        solo.bt.assign(c.bt.begin() + (size_t)s*c.n_blocks_per_seq,
                       c.bt.begin() + (size_t)s*c.n_blocks_per_seq + solo.n_blocks_per_seq);

        std::vector<float> alone;
        char lab[160];
        snprintf(lab, sizeof(lab), "%s slot=%d L=%d ne11 %d->%d", label.c_str(), s,
                 c.seq_len[s], c.ne11, solo.ne11);
        if (!report("launch", lab, solo.run(alone) == 0)) continue;

        std::vector<float> slot_slice(batch.begin() + (size_t)s*c.n_heads_q*D,
                                      batch.begin() + (size_t)(s+1)*c.n_heads_q*D);
        report("E:batch_invariance", lab, bytes_equal(slot_slice, alone));
    }
}

// ---------------------------------------------------------------------------
// Main sweep
// ---------------------------------------------------------------------------

int main() {
    const bool quick = std::getenv("FATTN_TEST_QUICK") != nullptr;
    g_fail_fast      = std::getenv("FATTN_TEST_FAIL_FAST") != nullptr;
    const int seed   = std::getenv("FATTN_TEST_SEED") ? std::atoi(std::getenv("FATTN_TEST_SEED")) : 42;

    // Straddle lengths: BLOCK=64, ILP_W=4 tail, CHUNK=1024 boundaries.
    const std::vector<int> L_full  = {1, 3, 5, 63, 64, 65, 127, 1023, 1024, 1025,
                                      2047, 2048, 2049, 3000, 4095, 4096, 4097};
    const std::vector<int> L_quick = {3, 64, 1023, 1024, 1025, 2048};
    const std::vector<int> & Ls = quick ? L_quick : L_full;

    int n_cfg = 0;
    for (int q4 = 1; q4 >= 0; q4--) {
        std::mt19937 rng(seed + q4);

        // --- single-seq straddle sweep: A, B, C(+1024, +2048), D ---
        for (int L : Ls) {
            SplitConfig c;
            c.q4_0       = q4 != 0;
            c.n_heads_q  = 24;
            c.n_kv_heads = 4;
            c.n_seqs     = 1;
            c.seq_len    = { L };
            c.ne11       = ((L + BLOCK_TOK - 1) / BLOCK_TOK) * BLOCK_TOK;
            c.build(rng);

            char label[96];
            snprintf(label, sizeof(label), "%s L=%d ne11=%d", q4 ? "q4_0" : "f16", L, c.ne11);
            run_accuracy_and_rep(c, label);
            run_padding_invariance(c, 1024, label);
            run_padding_invariance(c, 2048, label);
            run_permutation_invariance(c, rng, label);
            n_cfg++;
            if (g_fail_fast && g_fail) goto done;
        }

        // --- mixed batches: A, B, E ---
        // The all-single-chunk n_seqs=8 batch (every L <= 1024, ne11 <= 1024)
        // exercises the SPREAD launch layout at the production NP=8 decode
        // shape — the cell where the 2026-06-07 post-flip NPC matrix diverged.
        {
            const std::vector<std::vector<int>> batches = quick
                ? std::vector<std::vector<int>>{{1500, 70}}
                : std::vector<std::vector<int>>{
                      {1500, 70},
                      {2049, 1024, 63, 4096},
                      {4097, 3000, 2048, 1025, 1024, 1023, 65, 1},
                      {824, 612, 1024, 333, 95, 1001, 730, 64},
                      {1023, 1024, 960, 64, 512, 896, 128, 700}};
            for (const auto & lens : batches) {
                SplitConfig c;
                c.q4_0       = q4 != 0;
                c.n_heads_q  = 24;
                c.n_kv_heads = 4;
                c.n_seqs     = (int) lens.size();
                c.seq_len    = lens;
                const int maxL = *std::max_element(lens.begin(), lens.end());
                c.ne11       = ((maxL + BLOCK_TOK - 1) / BLOCK_TOK) * BLOCK_TOK;
                c.build(rng);

                char label[96];
                snprintf(label, sizeof(label), "%s batch n_seqs=%d ne11=%d",
                         q4 ? "q4_0" : "f16", c.n_seqs, c.ne11);
                run_accuracy_and_rep(c, label);
                run_batch_invariance(c, label);
                // Crosses the spread (single-chunk) and grouped (padded
                // multi-chunk) launch layouts at the full batch shape.
                run_padding_invariance(c, 1024, label);
                n_cfg++;
                if (g_fail_fast && g_fail) goto done;
            }
        }

        // --- per-GPU head shape (12 q-heads / 2 kv-heads — what each GPU of
        // the 2-way graph split actually runs), all-single-chunk n_seqs=8 ---
        {
            SplitConfig c;
            c.q4_0       = q4 != 0;
            c.n_heads_q  = 12;
            c.n_kv_heads = 2;
            c.n_seqs     = 8;
            c.seq_len    = {824, 612, 1024, 333, 95, 1001, 730, 64};
            c.ne11       = 1024;
            c.build(rng);
            char label[96];
            snprintf(label, sizeof(label), "%s perGPU 12/2 n_seqs=8", q4 ? "q4_0" : "f16");
            run_accuracy_and_rep(c, label);
            run_batch_invariance(c, label);
            run_padding_invariance(c, 1024, label);
            n_cfg++;
            if (g_fail_fast && g_fail) goto done;
        }

        // --- gqa=1 sanity (block (32,1,1) launch shape) ---
        {
            SplitConfig c;
            c.q4_0       = q4 != 0;
            c.n_heads_q  = 4;
            c.n_kv_heads = 4;
            c.n_seqs     = 1;
            c.seq_len    = { 1500 };
            c.ne11       = ((1500 + BLOCK_TOK - 1) / BLOCK_TOK) * BLOCK_TOK;
            c.build(rng);
            char label[96];
            snprintf(label, sizeof(label), "%s gqa=1 L=1500", q4 ? "q4_0" : "f16");
            run_accuracy_and_rep(c, label);
            n_cfg++;
            if (g_fail_fast && g_fail) goto done;
        }
    }

done:
    fprintf(stderr, "[split-sm75] %d configs%s, %d scenario failures\n",
            n_cfg, quick ? " (QUICK)" : "", g_fail);
    return g_fail ? 1 : 0;
}
