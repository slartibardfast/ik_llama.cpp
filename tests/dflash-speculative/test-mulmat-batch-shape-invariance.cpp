// test-mulmat-batch-shape-invariance.cpp
//
// L5 — kernel-direct test that ggml_mul_mat with a quantised weight
// (Q4_0 / Q4_0_AR16) and F32 input produces a byte-identical row 1
// at n_tokens=2 to a separate run at n_tokens=1 with the same input
// vector. If this FAILS, MMQ kernel dispatch (mmq_x tile size or
// internal K-reduction) introduces batch-shape variance — explains
// the layer 0 row 1 divergence bound by L4.
//
// Setup: synthesise a small quantised weight matrix [K, N_OUT] of
// type Q4_0 (block_q4_0 = 32 weights per block), pack it on device.
// Generate F32 input of [K, n_tokens]. Run mul_mat for n_tokens=1
// and n_tokens=2, then compare result row 1 of n=2 to row 0 of n=1
// (both should be weight @ input_col_1).
//
// We use a single-CUDA-device backend, no model load.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Production-ish dims: K = 5120 (n_embd of Qwen 3.6 27B), N_OUT = 8192
// (qkv combined). Q4_0 block size = 32.
static constexpr int K     = 5120;
static constexpr int N_OUT = 8192;
static constexpr int QK    = 32;  // Q4_0 block elements
static constexpr int N_BLOCKS_PER_ROW = K / QK;
static_assert(K % QK == 0, "K must be multiple of QK");

// Run mul_mat(weight[N_OUT, K] Q4_0, input[K, n_tokens] F32) -> dst[N_OUT, n_tokens] F32.
// Returns true on success and writes the full dst into `out_dst`.
static bool run_mul_mat(ggml_backend_t backend,
                        int n_tokens,
                        const std::vector<uint8_t> & weight_q4_0_bytes,
                        const std::vector<float> & input_f32,
                        std::vector<float> & out_dst) {
    static const size_t mem_size = 16 * 1024 * 1024;
    ggml_init_params params = { mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    // Weight as Q4_0: ne=[K, N_OUT], rows are N_OUT, cols are K.
    // In ggml convention, src0 has ne[0]=K (inner), ne[1]=N_OUT (outer).
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N_OUT);
    // Input as F32: ne=[K, n_tokens].
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n_tokens);
    // dst = w @ x  →  ne=[N_OUT, n_tokens].
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_set_name(y, n_tokens == 1 ? "mul_mat_n1" : "mul_mat_nN");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }

    ggml_backend_tensor_set(w, weight_q4_0_bytes.data(), 0, weight_q4_0_bytes.size());
    ggml_backend_tensor_set(x, input_f32.data(),         0, input_f32.size() * sizeof(float));

    const auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph_compute failed status=%d at n_tokens=%d\n", (int) status, n_tokens);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    const size_t n_floats = (size_t) ggml_nelements(y);
    out_dst.assign(n_floats, 0.0f);
    ggml_backend_tensor_get(y, out_dst.data(), 0, n_floats * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed; SKIP\n");
        return 77;
    }

    // Synthesise a deterministic F32 weight matrix and quantise to Q4_0.
    std::vector<float> w_f32((size_t) K * N_OUT);
    {
        std::mt19937_64 rng(0xF1ULL);
        for (size_t i = 0; i < w_f32.size(); ++i) {
            const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
            w_f32[i] = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 0.1f;
        }
    }
    // Quantise to Q4_0. Use the host-side quantizer; compute row size
    // via ggml_type_size / ggml_blck_size to avoid pulling in the
    // internal block struct header.
    const size_t row_bytes = (size_t)(K / ggml_blck_size(GGML_TYPE_Q4_0))
                              * ggml_type_size(GGML_TYPE_Q4_0);
    std::vector<uint8_t> w_q4_0((size_t) N_OUT * row_bytes, 0);
    for (int r = 0; r < N_OUT; ++r) {
        const size_t off_in  = (size_t) r * K;
        const size_t off_out = (size_t) r * row_bytes;
        const int64_t produced = ggml_quantize_chunk(
            GGML_TYPE_Q4_0,
            w_f32.data() + off_in,
            w_q4_0.data() + off_out,
            /*start_row=*/0, /*nrows=*/1, /*n_per_row=*/K,
            /*imatrix=*/nullptr,
            /*user_data=*/nullptr);
        if (produced <= 0) {
            fprintf(stderr, "ggml_quantize_chunk failed row=%d\n", r);
            ggml_backend_free(backend);
            return 1;
        }
    }
    fprintf(stderr, "[L5] quantised %d-row × %d-col weight to Q4_0\n", N_OUT, K);

    // Synthesise an 8-column F32 input. Compare:
    //   y_n1 = w @ x_n1 where x_n1 = column 1 of x_n8 (single col)
    //   y_n2 col 1 = w @ x_n2 col 1 where x_n2 = first 2 cols of x_n8
    //   y_n8 col 1 = w @ x_n8 col 1
    // All three should produce IDENTICAL bytes at col 1 of the result
    // if the kernel is batch-shape invariant.
    std::vector<float> x_n8((size_t) K * 8);
    {
        std::mt19937_64 rng(0xF2ULL);
        for (size_t i = 0; i < x_n8.size(); ++i) {
            const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
            x_n8[i] = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 1.0f;
        }
    }
    std::vector<float> x_n1(x_n8.begin() + K, x_n8.begin() + 2 * K);
    std::vector<float> x_n2(x_n8.begin(),     x_n8.begin() + 2 * K);

    std::vector<float> y_n1, y_n2, y_n8;
    if (!run_mul_mat(backend, 1, w_q4_0, x_n1, y_n1)) { ggml_backend_free(backend); return 1; }
    if (!run_mul_mat(backend, 2, w_q4_0, x_n2, y_n2)) { ggml_backend_free(backend); return 1; }
    if (!run_mul_mat(backend, 8, w_q4_0, x_n8, y_n8)) { ggml_backend_free(backend); return 1; }
    ggml_backend_free(backend);

    if ((int) y_n1.size() != N_OUT || (int) y_n2.size() != 2 * N_OUT ||
        (int) y_n8.size() != 8 * N_OUT) {
        fprintf(stderr, "[FAIL] unexpected y sizes: y_n1=%zu y_n2=%zu y_n8=%zu\n",
                y_n1.size(), y_n2.size(), y_n8.size());
        return 1;
    }

    auto count_diffs = [](const float * a, const float * b, int n, int & first_diff, float & max_abs) {
        int n_diff = 0;
        first_diff = -1;
        max_abs = 0.0f;
        for (int i = 0; i < n; ++i) {
            uint32_t ai, bi;
            std::memcpy(&ai, &a[i], 4); std::memcpy(&bi, &b[i], 4);
            if (ai != bi) {
                if (first_diff < 0) first_diff = i;
                ++n_diff;
                const float d = std::abs(a[i] - b[i]);
                if (d > max_abs) max_abs = d;
            }
        }
        return n_diff;
    };

    // Run y_n1' = w @ col_0_of_x_n8 separately to test the existing
    // FATTN_SHAPE_INVARIANT claim of "row-0 (= col-0) invariance".
    std::vector<float> x_n1_c0(x_n8.begin(), x_n8.begin() + K);

    int first_diff_ab, first_diff_ac, first_diff_bc;
    float max_abs_ab, max_abs_ac, max_abs_bc;

    // col 1 comparison (path A's input).
    const float * a = y_n1.data();
    const float * b = y_n2.data() + N_OUT;       // col 1 of n=2
    const float * c = y_n8.data() + N_OUT;       // col 1 of n=8
    const int diff_ab = count_diffs(a, b, N_OUT, first_diff_ab, max_abs_ab);
    const int diff_ac = count_diffs(a, c, N_OUT, first_diff_ac, max_abs_ac);
    const int diff_bc = count_diffs(b, c, N_OUT, first_diff_bc, max_abs_bc);
    fprintf(stderr, "y_n1(col1) vs y_n2 col 1: diff=%d/%d max|Δ|=%.3e\n",
            diff_ab, N_OUT, max_abs_ab);
    fprintf(stderr, "y_n1(col1) vs y_n8 col 1: diff=%d/%d max|Δ|=%.3e\n",
            diff_ac, N_OUT, max_abs_ac);
    fprintf(stderr, "y_n2 col 1 vs y_n8 col 1: diff=%d/%d max|Δ|=%.3e\n",
            diff_bc, N_OUT, max_abs_bc);

    // col 0 comparison — the FATTN_SHAPE_INVARIANT-guaranteed case.
    std::vector<float> y_n1_c0;
    {
        ggml_backend_t backend2 = ggml_backend_cuda_init(0, nullptr);
        run_mul_mat(backend2, 1, w_q4_0, x_n1_c0, y_n1_c0);
        ggml_backend_free(backend2);
    }
    const float * d = y_n1_c0.data();
    const float * e = y_n2.data();       // col 0 of n=2
    const float * f = y_n8.data();       // col 0 of n=8
    int first_diff_de, first_diff_df, first_diff_ef;
    float max_abs_de, max_abs_df, max_abs_ef;
    const int diff_de = count_diffs(d, e, N_OUT, first_diff_de, max_abs_de);
    const int diff_df = count_diffs(d, f, N_OUT, first_diff_df, max_abs_df);
    const int diff_ef = count_diffs(e, f, N_OUT, first_diff_ef, max_abs_ef);
    fprintf(stderr, "y_n1(col0) vs y_n2 col 0: diff=%d/%d max|Δ|=%.3e\n",
            diff_de, N_OUT, max_abs_de);
    fprintf(stderr, "y_n1(col0) vs y_n8 col 0: diff=%d/%d max|Δ|=%.3e\n",
            diff_df, N_OUT, max_abs_df);
    fprintf(stderr, "y_n2 col 0 vs y_n8 col 0: diff=%d/%d max|Δ|=%.3e\n",
            diff_ef, N_OUT, max_abs_ef);

    const bool col0_invariant = (diff_de == 0 && diff_df == 0 && diff_ef == 0);
    const bool col1_invariant = (diff_ab == 0 && diff_ac == 0 && diff_bc == 0);

    if (col0_invariant && col1_invariant) {
        printf("[PASS] mul_mat(Q4_0) fully batch-shape-invariant across all columns.\n");
        return 0;
    }
    if (col0_invariant && !col1_invariant) {
        printf("[FAIL-COL1] mul_mat(Q4_0) is shape-invariant ONLY at column 0 (the "
               "existing FATTN_SHAPE_INVARIANT guarantee). Columns > 0 diverge — this "
               "explains why NPC concurrent-multi-slot-single-token passes but "
               "single-slot-multi-token does NOT. The verify-batch decoder needs "
               "ALL columns invariant.\n");
        return 1;
    }
    printf("[FAIL] mul_mat(Q4_0) batch-shape-variant at multiple columns. "
           "col1: diff_ab=%d diff_bc=%d. col0: diff_de=%d diff_ef=%d.\n",
           diff_ab, diff_bc, diff_de, diff_ef);
    return 1;
}
