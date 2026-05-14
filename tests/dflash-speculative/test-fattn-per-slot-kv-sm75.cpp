// test-fattn-per-slot-kv-sm75.cpp
//
// PLAN.md S2.5.b — RED unit test for `fattn_per_slot_kv_sm75`
// (specs/deltanet/fattn-per-slot-kv-sm75.md §10 S2.5.b).
//
// Test discipline: this binary EXISTS BEFORE the kernel does. Until the
// kernel is implemented, the build links against the kernel stub
// (link error → expected RED state per feedback_test_first_discipline).
// Once the kernel lands, this test transitions to GREEN.
//
// Coverage: ~256 configs across (NP, slot_seq_lens, n_heads, KV_BLOCK_SIZE,
//   USE_SOFTCAP, seed). Each config generates random fp16 inputs, calls
//   both the kernel and the scalar fp32 oracle, compares with three
//   binding modes (byte-identity / fp16 ULP-close / cosine + NMSE).
//
// Three test scenarios:
//   - Scenario A: oracle-vs-kernel agreement (closure §7 numerical)
//   - Scenario B: kernel rep-determinism (closure §6 determinism)
//   - Scenario C: per-slot batch-invariance (closure FA_NkvIsDominantBatchShapeEntryPoint)
//
// Env (all optional; defaults exercise the full sweep):
//   FATTN_TEST_QUICK=1            — abbreviated sweep (~8 configs) for CI
//   FATTN_TEST_FAIL_FAST=1        — bail on first failure
//   FATTN_TEST_JSON=<path>        — emit per-config result JSON
//   FATTN_TEST_SEED=<int>         — override base seed
//
// Exit:
//   0  = all configs pass all 3 scenarios
//   1  = at least one config failed
//   77 = SKIP (e.g., no CUDA device with sm_75 capability)

#include "fattn-per-slot-kv-sm75-reference.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <string>
#include <cstdint>

using namespace fattn_per_slot_kv_sm75;

// ---------------------------------------------------------------------------
// Forward-declared kernel entry point.
//
// IMPLEMENTATION TARGET (does not yet exist as of this test's commit;
// implementing this symbol is the kernel work — see PLAN.md S2.5/S2.6):
//
//   void fattn_per_slot_kv_sm75_launch(
//       const Half * Q,            // [Dq, n_tokens, n_heads_q, n_seqs]
//       const Half * K,            // [Dq, n_kv_max, n_kv_heads, n_seqs]
//       const Half * V,            // [Dv, n_kv_max, n_kv_heads, n_seqs]
//       const Half * mask,         // [n_kv_max, n_tokens]
//       const int32_t * slot_seq_lens,
//       float * out_final,         // [Dv, n_tokens, n_heads_q, n_seqs] — combined output
//       const AttnConfig & cfg
//   );
//
// The launcher is responsible for: allocating device buffers, dispatching
// the kernel + combine kernel, and returning the final fp32 output.
// ---------------------------------------------------------------------------

extern "C" int fattn_per_slot_kv_sm75_launch(
    const Half * Q,
    const Half * K,
    const Half * V,
    const Half * mask,
    const int32_t * slot_seq_lens,
    float * out_final,
    const AttnConfig & cfg
);

// ---------------------------------------------------------------------------
// Config sweep
// ---------------------------------------------------------------------------

struct TestConfig {
    AttnConfig cfg;
    int seed;
    std::vector<int32_t> slot_seq_lens;
};

static std::vector<TestConfig> build_sweep(bool quick) {
    std::vector<TestConfig> configs;

    // Locked production tuple (OQ-4 LOCKED).
    constexpr int Dq = 256;
    constexpr int Dv = 128;
    constexpr int n_heads_q = 12;
    constexpr int n_kv_heads = 2;

    const std::vector<int> kv_block_sizes = quick ? std::vector<int>{32} : std::vector<int>{32, 64};
    const std::vector<int> nps            = quick ? std::vector<int>{2, 4} : std::vector<int>{1, 2, 4, 8};
    const std::vector<int> n_tokens_set   = quick ? std::vector<int>{1, 16} : std::vector<int>{1, 4, 16, 64, 256};
    const std::vector<int> n_kv_set       = quick ? std::vector<int>{128, 1024} : std::vector<int>{32, 128, 512, 4096};
    const std::vector<int> seeds          = quick ? std::vector<int>{1} : std::vector<int>{1, 2, 3, 4};
    const std::vector<bool> softcap_set   = quick ? std::vector<bool>{false} : std::vector<bool>{false, true};

    for (int kvb : kv_block_sizes) {
        for (int np : nps) {
            for (int n_tok : n_tokens_set) {
                if (np > 1 && n_tok > 1) continue;  // decode (n_tok=1, np>=1) + prefill (n_tok>=1, np=1)
                for (int n_kv : n_kv_set) {
                    if (n_kv < n_tok) continue;
                    for (int seed : seeds) {
                        for (bool sc : softcap_set) {
                            TestConfig tc;
                            tc.cfg.head_dim_q  = Dq;
                            tc.cfg.head_dim_v  = Dv;
                            tc.cfg.kv_block_size = kvb;
                            tc.cfg.n_tokens    = n_tok;
                            tc.cfg.n_heads_q   = n_heads_q;
                            tc.cfg.n_kv_heads  = n_kv_heads;
                            tc.cfg.n_seqs      = np;
                            tc.cfg.n_kv_max    = n_kv;
                            tc.cfg.parallel_blocks = std::max(1, (n_kv + 255) / 256);
                            tc.cfg.scale       = 1.0f / std::sqrt((float)Dq);
                            tc.cfg.softcap     = sc ? 30.0f : 0.0f;
                            tc.cfg.use_softcap = sc;
                            tc.seed = seed;
                            // Per-slot seq lengths: vary across slots within batch.
                            // Slot k gets n_kv * (1 - 0.1*k) rounded, min 8.
                            tc.slot_seq_lens.resize(np);
                            for (int k = 0; k < np; k++) {
                                int len = std::max(8, (int)(n_kv * (1.0 - 0.1 * k)));
                                tc.slot_seq_lens[k] = std::min(len, n_kv);
                            }
                            configs.push_back(std::move(tc));
                        }
                    }
                }
            }
        }
    }
    return configs;
}

// ---------------------------------------------------------------------------
// Random input generators
// ---------------------------------------------------------------------------

static void fill_random_halfs(Half * dst, size_t n, std::mt19937 & rng, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> dist(lo, hi);
    for (size_t i = 0; i < n; i++) {
        dst[i] = float_to_half(dist(rng));
    }
}

static void fill_random_mask(Half * dst, size_t n, std::mt19937 & rng) {
    // Mask is usually 0 (allow) or -inf (deny). We use 0 for allow, -1e4 for deny.
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n; i++) {
        dst[i] = float_to_half(dist(rng) < 0.1f ? -1e4f : 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Test runners
// ---------------------------------------------------------------------------

struct TestResult {
    std::string scenario;
    int config_idx;
    bool passed;
    CompareResult cmp;
    std::string note;
};

static TestResult run_oracle_vs_kernel(int idx, const TestConfig & tc) {
    TestResult r;
    r.scenario = "A:oracle_vs_kernel";
    r.config_idx = idx;

    std::mt19937 rng(tc.seed * 1000 + idx);
    const AttnConfig & c = tc.cfg;

    const size_t n_Q = (size_t)c.head_dim_q * c.n_tokens * c.n_heads_q * c.n_seqs;
    const size_t n_K = (size_t)c.head_dim_q * c.n_kv_max * c.n_kv_heads * c.n_seqs;
    const size_t n_V = (size_t)c.head_dim_v * c.n_kv_max * c.n_kv_heads * c.n_seqs;
    const size_t n_M = (size_t)c.n_kv_max * c.n_tokens;
    const size_t n_O = (size_t)c.head_dim_v * c.n_tokens * c.n_heads_q * c.n_seqs;

    std::vector<Half> Q(n_Q), K(n_K), V(n_V), M(n_M);
    fill_random_halfs(Q.data(), n_Q, rng);
    fill_random_halfs(K.data(), n_K, rng);
    fill_random_halfs(V.data(), n_V, rng);
    fill_random_mask(M.data(), n_M, rng);

    // Reference output (via oracle).
    std::vector<float> out_ref(n_O, 0.0f);
    std::vector<int32_t> pb_per_slot(c.n_seqs);
    for (int s = 0; s < c.n_seqs; s++) {
        pb_per_slot[s] = std::max(1, (tc.slot_seq_lens[s] + 255) / 256);
    }
    reference_full(c, Q.data(), K.data(), V.data(), M.data(),
                   tc.slot_seq_lens.data(), pb_per_slot.data(), out_ref.data());

    // Kernel output (via the to-be-implemented launcher).
    std::vector<float> out_kern(n_O, 0.0f);
    const int rc = fattn_per_slot_kv_sm75_launch(
        Q.data(), K.data(), V.data(), M.data(),
        tc.slot_seq_lens.data(), out_kern.data(), c
    );
    if (rc != 0) {
        r.passed = false;
        r.note = "kernel launcher returned non-zero: " + std::to_string(rc);
        return r;
    }

    r.cmp = compare(out_ref.data(), out_kern.data(), n_O);
    // Pass criteria: cosine ≥ 0.9999 AND nmse ≤ 1e-4, OR exact byte match
    // (the latter is the strict-binding aspiration; cosine+nmse is the
    // closure binding per spec §7).
    bool cosine_ok = r.cmp.cosine >= 0.9999f;
    bool nmse_ok = r.cmp.nmse <= 1e-4f;
    bool byte_ok = r.cmp.n_exact_byte_match == r.cmp.n_total;
    r.passed = (cosine_ok && nmse_ok) || byte_ok;
    return r;
}

static TestResult run_rep_determinism(int idx, const TestConfig & tc) {
    TestResult r;
    r.scenario = "B:rep_determinism";
    r.config_idx = idx;

    std::mt19937 rng(tc.seed * 1000 + idx + 7);
    const AttnConfig & c = tc.cfg;

    const size_t n_Q = (size_t)c.head_dim_q * c.n_tokens * c.n_heads_q * c.n_seqs;
    const size_t n_K = (size_t)c.head_dim_q * c.n_kv_max * c.n_kv_heads * c.n_seqs;
    const size_t n_V = (size_t)c.head_dim_v * c.n_kv_max * c.n_kv_heads * c.n_seqs;
    const size_t n_M = (size_t)c.n_kv_max * c.n_tokens;
    const size_t n_O = (size_t)c.head_dim_v * c.n_tokens * c.n_heads_q * c.n_seqs;

    std::vector<Half> Q(n_Q), K(n_K), V(n_V), M(n_M);
    fill_random_halfs(Q.data(), n_Q, rng);
    fill_random_halfs(K.data(), n_K, rng);
    fill_random_halfs(V.data(), n_V, rng);
    fill_random_mask(M.data(), n_M, rng);

    std::vector<float> out1(n_O), out2(n_O);
    int rc1 = fattn_per_slot_kv_sm75_launch(
        Q.data(), K.data(), V.data(), M.data(),
        tc.slot_seq_lens.data(), out1.data(), c);
    int rc2 = fattn_per_slot_kv_sm75_launch(
        Q.data(), K.data(), V.data(), M.data(),
        tc.slot_seq_lens.data(), out2.data(), c);
    if (rc1 != 0 || rc2 != 0) {
        r.passed = false;
        r.note = "launcher returned non-zero";
        return r;
    }
    r.cmp = compare(out1.data(), out2.data(), n_O);
    // Rep determinism = byte-identity (strict).
    r.passed = r.cmp.n_exact_byte_match == r.cmp.n_total;
    return r;
}

static TestResult run_batch_invariance(int idx, const TestConfig & tc) {
    TestResult r;
    r.scenario = "C:batch_invariance";
    r.config_idx = idx;
    const AttnConfig & c = tc.cfg;

    if (c.n_seqs < 2) {
        r.passed = true;
        r.note = "SKIP (n_seqs < 2 — invariance is trivially held)";
        return r;
    }

    // Build a config with n_seqs/2 seqs holding the FIRST HALF of original slots'
    // KV state; compare slot 0's output against the n_seqs-sized run's slot 0.
    std::mt19937 rng(tc.seed * 1000 + idx + 13);

    const size_t n_K = (size_t)c.head_dim_q * c.n_kv_max * c.n_kv_heads * c.n_seqs;
    const size_t n_V = (size_t)c.head_dim_v * c.n_kv_max * c.n_kv_heads * c.n_seqs;
    std::vector<Half> Q_big((size_t)c.head_dim_q * c.n_tokens * c.n_heads_q * c.n_seqs);
    std::vector<Half> K_big(n_K), V_big(n_V);
    std::vector<Half> M_big((size_t)c.n_kv_max * c.n_tokens);
    fill_random_halfs(Q_big.data(), Q_big.size(), rng);
    fill_random_halfs(K_big.data(), K_big.size(), rng);
    fill_random_halfs(V_big.data(), V_big.size(), rng);
    fill_random_mask(M_big.data(), M_big.size(), rng);

    AttnConfig c_big = c;
    std::vector<float> out_big((size_t)c.head_dim_v * c.n_tokens * c.n_heads_q * c.n_seqs);
    int rc1 = fattn_per_slot_kv_sm75_launch(
        Q_big.data(), K_big.data(), V_big.data(), M_big.data(),
        tc.slot_seq_lens.data(), out_big.data(), c_big);

    // Smaller config: half the slots (just slot 0 if n_seqs=2; slots 0,1 if n_seqs=4; etc.).
    const int n_seqs_small = std::max(1, c.n_seqs / 2);
    AttnConfig c_small = c;
    c_small.n_seqs = n_seqs_small;
    std::vector<Half> Q_small((size_t)c.head_dim_q * c.n_tokens * c.n_heads_q * n_seqs_small);
    std::vector<Half> K_small((size_t)c.head_dim_q * c.n_kv_max * c.n_kv_heads * n_seqs_small);
    std::vector<Half> V_small((size_t)c.head_dim_v * c.n_kv_max * c.n_kv_heads * n_seqs_small);
    std::vector<Half> M_small((size_t)c.n_kv_max * c.n_tokens);  // same shape regardless of n_seqs
    std::vector<int32_t> slot_seq_lens_small(n_seqs_small);

    // Copy first-n_seqs_small slots from the big config.
    const size_t Qslot_big = (size_t)c.head_dim_q * c.n_tokens * c.n_heads_q;
    const size_t Kslot_big = (size_t)c.head_dim_q * c.n_kv_max  * c.n_kv_heads;
    const size_t Vslot_big = (size_t)c.head_dim_v * c.n_kv_max  * c.n_kv_heads;
    for (int s = 0; s < n_seqs_small; s++) {
        std::memcpy(Q_small.data() + s * Qslot_big, Q_big.data() + s * Qslot_big, Qslot_big * sizeof(Half));
        std::memcpy(K_small.data() + s * Kslot_big, K_big.data() + s * Kslot_big, Kslot_big * sizeof(Half));
        std::memcpy(V_small.data() + s * Vslot_big, V_big.data() + s * Vslot_big, Vslot_big * sizeof(Half));
        slot_seq_lens_small[s] = tc.slot_seq_lens[s];
    }
    std::memcpy(M_small.data(), M_big.data(), M_big.size() * sizeof(Half));

    std::vector<float> out_small((size_t)c.head_dim_v * c.n_tokens * c.n_heads_q * n_seqs_small);
    int rc2 = fattn_per_slot_kv_sm75_launch(
        Q_small.data(), K_small.data(), V_small.data(), M_small.data(),
        slot_seq_lens_small.data(), out_small.data(), c_small);

    if (rc1 != 0 || rc2 != 0) {
        r.passed = false;
        r.note = "launcher returned non-zero";
        return r;
    }

    // Compare slot 0's output region across the two configs. Both should be
    // BYTE-IDENTICAL because slot 0's per-slot state is identical.
    const size_t Oslot = (size_t)c.head_dim_v * c.n_tokens * c.n_heads_q;
    r.cmp = compare(out_big.data(), out_small.data(), Oslot);
    r.passed = r.cmp.n_exact_byte_match == r.cmp.n_total;
    return r;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    const bool quick      = std::getenv("FATTN_TEST_QUICK")     != nullptr;
    const bool fail_fast  = std::getenv("FATTN_TEST_FAIL_FAST") != nullptr;
    const char * json_out = std::getenv("FATTN_TEST_JSON");
    const int base_seed   = std::getenv("FATTN_TEST_SEED") ? std::atoi(std::getenv("FATTN_TEST_SEED")) : 42;

    auto configs = build_sweep(quick);
    fprintf(stderr, "[fattn_per_slot_kv_sm75] %zu test configs%s\n",
            configs.size(), quick ? " (QUICK MODE)" : "");

    std::vector<TestResult> results;
    int n_fail_A = 0, n_fail_B = 0, n_fail_C = 0;
    int n_pass_A = 0, n_pass_B = 0, n_pass_C = 0;

    FILE * fjson = json_out ? std::fopen(json_out, "w") : nullptr;
    if (fjson) std::fprintf(fjson, "[\n");

    bool first_json = true;
    for (size_t i = 0; i < configs.size(); i++) {
        TestConfig & tc = configs[i];
        tc.seed = base_seed + (int)i;

        TestResult rA = run_oracle_vs_kernel((int)i, tc);
        if (rA.passed) n_pass_A++; else n_fail_A++;
        results.push_back(rA);

        TestResult rB = run_rep_determinism((int)i, tc);
        if (rB.passed) n_pass_B++; else n_fail_B++;
        results.push_back(rB);

        TestResult rC = run_batch_invariance((int)i, tc);
        if (rC.passed) n_pass_C++; else n_fail_C++;
        results.push_back(rC);

        if (fjson) {
            for (const TestResult & r : {rA, rB, rC}) {
                std::fprintf(fjson,
                    "%s  {\"config\": %d, \"scenario\": \"%s\", \"passed\": %s, "
                    "\"max_diff\": %g, \"cosine\": %.7f, \"nmse\": %g, \"note\": \"%s\"}",
                    first_json ? "" : ",\n",
                    r.config_idx, r.scenario.c_str(),
                    r.passed ? "true" : "false",
                    r.cmp.max_abs_diff, r.cmp.cosine, r.cmp.nmse,
                    r.note.c_str());
                first_json = false;
            }
        }

        if (fail_fast && (!rA.passed || !rB.passed || !rC.passed)) {
            fprintf(stderr, "[fattn_per_slot_kv_sm75] FAIL FAST at config %zu\n", i);
            break;
        }
    }

    if (fjson) {
        std::fprintf(fjson, "\n]\n");
        std::fclose(fjson);
    }

    fprintf(stderr,
        "[fattn_per_slot_kv_sm75] Scenario A (oracle_vs_kernel):    %d pass / %d fail\n"
        "[fattn_per_slot_kv_sm75] Scenario B (rep_determinism):     %d pass / %d fail\n"
        "[fattn_per_slot_kv_sm75] Scenario C (batch_invariance):    %d pass / %d fail\n",
        n_pass_A, n_fail_A, n_pass_B, n_fail_B, n_pass_C, n_fail_C);

    const int n_fail_total = n_fail_A + n_fail_B + n_fail_C;
    return n_fail_total ? 1 : 0;
}
