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

    // Synthesise a 2-column F32 input. We will compare row 1 of the
    // 2-column result to row 0 of a separate 1-column result that uses
    // the SAME column-1 vector.
    std::vector<float> x_n2((size_t) K * 2);
    {
        std::mt19937_64 rng(0xF2ULL);
        for (size_t i = 0; i < x_n2.size(); ++i) {
            const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
            x_n2[i] = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 1.0f;
        }
    }
    // x_n1 is just column 1 of x_n2.
    std::vector<float> x_n1(x_n2.begin() + K, x_n2.begin() + 2 * K);

    // Run mul_mat with n_tokens=1 and n_tokens=2.
    std::vector<float> y_n1, y_n2;
    if (!run_mul_mat(backend, 1, w_q4_0, x_n1, y_n1)) { ggml_backend_free(backend); return 1; }
    if (!run_mul_mat(backend, 2, w_q4_0, x_n2, y_n2)) { ggml_backend_free(backend); return 1; }
    ggml_backend_free(backend);

    // Compare y_n2's column 1 (size N_OUT, at offset N_OUT) to y_n1's column 0 (size N_OUT, at offset 0).
    if ((int) y_n1.size() != N_OUT || (int) y_n2.size() != 2 * N_OUT) {
        fprintf(stderr, "[FAIL] unexpected y sizes: y_n1=%zu y_n2=%zu\n",
                y_n1.size(), y_n2.size());
        return 1;
    }
    const float * a = y_n1.data();
    const float * b = y_n2.data() + N_OUT;  // column 1 of n=2
    int n_diff = 0;
    int first_diff = -1;
    float max_abs = 0.0f;
    for (int i = 0; i < N_OUT; ++i) {
        uint32_t ai, bi;
        std::memcpy(&ai, &a[i], 4); std::memcpy(&bi, &b[i], 4);
        if (ai != bi) {
            if (first_diff < 0) first_diff = i;
            ++n_diff;
            const float d = std::abs(a[i] - b[i]);
            if (d > max_abs) max_abs = d;
        }
    }
    fprintf(stderr, "first 8 floats of y_n1: ");
    for (int i = 0; i < 8; ++i) fprintf(stderr, " %+10.6f", a[i]);
    fprintf(stderr, "\nfirst 8 floats of y_n2 col 1: ");
    for (int i = 0; i < 8; ++i) fprintf(stderr, " %+10.6f", b[i]);
    fprintf(stderr, "\n");

    if (n_diff == 0) {
        printf("[PASS] mul_mat(Q4_0 weight) is byte-shape-invariant: y_n1 == y_n2 col 1, "
               "%d fp32 floats byte-identical. The L4 layer 0 divergence is NOT caused "
               "by the q4_0 mul_mat kernel.\n", N_OUT);
        return 0;
    }
    printf("[FAIL] mul_mat(Q4_0 weight) DIFFERS between n_tokens=1 and n_tokens=2 col 1: "
           "%d/%d floats differ (max |Δ|=%.3e, first diff at idx %d: %+.6f vs %+.6f). "
           "MMQ kernel introduces batch-shape variance — this is the L4 layer 0 root "
           "cause.\n",
           n_diff, N_OUT, max_abs, first_diff, a[first_diff], b[first_diff]);
    return 1;
}
