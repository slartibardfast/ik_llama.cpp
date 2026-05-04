// Phase 1.A — CPU + AVX SIMD tests for GGML_TYPE_Q4_0_AR16.
//
// Drives the production type-traits dispatch (the same path
// ggml_compute_forward_mul_mat takes), with an *independent* scalar
// reference for vec_dot — see project memory feedback note
// `feedback_simd_needs_independent_reference`.

#include "ggml.h"

#undef NDEBUG
#include <assert.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

static int g_failures = 0;

static const char * ok_str(bool ok) { return ok ? "ok" : "FAILED"; }

// Independent fp32 scalar reference dot.
static float scalar_dot(const float * a, const float * b, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; ++i) acc += (double) a[i] * (double) b[i];
    return (float) acc;
}

// --- Test 1: Round-trip (quant -> dequant) ----------------------------------
static void test_round_trip() {
    const int n = 512;
    const int n_blocks = n / 16;

    std::mt19937 rng(0xC0FFEEu);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) x[i] = dist(rng);

    auto qfns = ggml_internal_get_type_traits(GGML_TYPE_Q4_0_AR16);
    assert(qfns.blck_size == 16 && qfns.type_size == 10);

    std::vector<uint8_t> packed(n_blocks * 10);
    qfns.from_float(x.data(), packed.data(), n);

    std::vector<float> y(n);
    qfns.to_float(packed.data(), y.data(), n);

    bool ok = true;
    float max_rel = 0.0f;
    for (int bi = 0; bi < n_blocks; ++bi) {
        float bmax = 0.0f;
        for (int j = 0; j < 16; ++j) bmax = std::max(bmax, std::fabs(x[bi*16 + j]));
        const float quantum = bmax / 8.0f;
        // Per-element error budget: one INT4 quantum (round-to-nearest)
        // plus the fp16-of-scale rounding slop. fp16 has 10 mantissa bits,
        // so |fp16(d) - d| <= d * 2^-11; over a code in [-8,7] that
        // contributes at most 8 * d * 2^-11 = bmax / 256 of extra drift.
        const float tol = quantum + bmax * (1.0f / 256.0f) + 1e-6f;
        for (int j = 0; j < 16; ++j) {
            const float err = std::fabs(x[bi*16 + j] - y[bi*16 + j]);
            if (err > tol) ok = false;
            if (quantum > 0) max_rel = std::max(max_rel, err / quantum);
        }
    }
    if (!ok) g_failures++;
    printf("[round-trip]      max_err_in_quanta=%.4f  -> %s\n", max_rel, ok_str(ok));
}

// --- Test 2: Determinism ----------------------------------------------------
static void test_determinism() {
    const int n = 512;
    std::mt19937 rng(0xD37E2u);
    std::uniform_real_distribution<float> dist(-2.5f, 2.5f);
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) x[i] = dist(rng);

    auto qfns = ggml_internal_get_type_traits(GGML_TYPE_Q4_0_AR16);
    std::vector<uint8_t> a(n / 16 * 10), b(n / 16 * 10);
    qfns.from_float(x.data(), a.data(), n);
    qfns.from_float(x.data(), b.data(), n);
    const bool ok = std::memcmp(a.data(), b.data(), a.size()) == 0;
    if (!ok) g_failures++;
    printf("[determinism]     bytes_equal=%s             -> %s\n",
           ok ? "yes" : "no", ok_str(ok));
}

// --- Test 3: Col-perm equivalence (16-aligned) — THE critical invariant -----
static void test_col_perm_equivalence() {
    const int n_blocks = 32;
    const int n = n_blocks * 16;

    std::mt19937 rng(0x5EED5u);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) x[i] = dist(rng);

    std::vector<int> perm(n_blocks);
    for (int i = 0; i < n_blocks; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);

    auto qfns = ggml_internal_get_type_traits(GGML_TYPE_Q4_0_AR16);

    std::vector<uint8_t> packed(n_blocks * 10);
    qfns.from_float(x.data(), packed.data(), n);

    // Path A: permute blocks at the byte level, then dequantize.
    std::vector<uint8_t> packed_perm(n_blocks * 10);
    for (int new_i = 0; new_i < n_blocks; ++new_i) {
        std::memcpy(&packed_perm[new_i * 10], &packed[perm[new_i] * 10], 10);
    }
    std::vector<float> A(n);
    qfns.to_float(packed_perm.data(), A.data(), n);

    // Path B: dequantize, then permute the 16-element column chunks.
    std::vector<float> direct(n);
    qfns.to_float(packed.data(), direct.data(), n);
    std::vector<float> B(n);
    for (int new_i = 0; new_i < n_blocks; ++new_i) {
        std::memcpy(&B[new_i * 16], &direct[perm[new_i] * 16], 16 * sizeof(float));
    }

    const bool ok = std::memcmp(A.data(), B.data(), n * sizeof(float)) == 0;
    if (!ok) g_failures++;
    printf("[col-perm-16]     bit_exact=%s              -> %s\n",
           ok ? "yes" : "no", ok_str(ok));
}

// --- Test 4: vec_dot vs independent scalar reference ------------------------
static void test_vec_dot() {
    const int n = 512;  // multiple of 32 (Q8_0 block size)
    std::mt19937 rng(0xBEE5u);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    std::vector<float> w(n), q(n);
    for (int i = 0; i < n; ++i) w[i] = dist(rng);
    for (int i = 0; i < n; ++i) q[i] = dist(rng);

    auto ar16 = ggml_internal_get_type_traits(GGML_TYPE_Q4_0_AR16);
    auto q8   = ggml_internal_get_type_traits(GGML_TYPE_Q8_0);

    std::vector<uint8_t> wq(n / 16 * 10);
    ar16.from_float(w.data(), wq.data(), n);

    const size_t q8_bytes = (size_t)(n / 32) * (size_t) q8.type_size;
    std::vector<uint8_t> qq(q8_bytes);
    q8.from_float(q.data(), qq.data(), n);

    // Independent reference: dequantize both sides, take fp32 dot.
    std::vector<float> w_dq(n), q_dq(n);
    ar16.to_float(wq.data(), w_dq.data(), n);
    q8.to_float(qq.data(), q_dq.data(), n);
    const float ref = scalar_dot(w_dq.data(), q_dq.data(), n);

    float got = 0.0f;
    ar16.vec_dot(n, &got, 0, wq.data(), 0, qq.data(), 0, 1);

    const float abs_tol = 1e-4f;
    const float rel_tol = 1e-3f;
    const float abs_err = std::fabs(got - ref);
    const float rel_err = std::fabs(ref) > 0 ? abs_err / std::fabs(ref) : abs_err;
    const bool ok = (abs_err <= abs_tol) || (rel_err <= rel_tol);
    if (!ok) g_failures++;
    printf("[vec_dot]         ref=%.6f got=%.6f abs=%.2e rel=%.2e -> %s\n",
           ref, got, abs_err, rel_err, ok_str(ok));
}

int main() {
    // ggml_init populates the fp16<->fp32 lookup table that
    // GGML_FP16_TO_FP32 dispatches through on x86. Without it, every
    // dequant returns 0 (silent zeros, no error).
    struct ggml_init_params ip = { /*.mem_size =*/ 1024,
                                   /*.mem_buffer =*/ nullptr,
                                   /*.no_alloc =*/ true };
    struct ggml_context * ctx = ggml_init(ip);

    test_round_trip();
    test_determinism();
    test_col_perm_equivalence();
    test_vec_dot();

    ggml_free(ctx);

    if (g_failures == 0) {
        printf("\nALL OK\n");
        return 0;
    }
    printf("\n%d test(s) FAILED\n", g_failures);
    return 1;
}
