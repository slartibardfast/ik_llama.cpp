// test-mulmat-batch-shape-invariance.cpp
//
// Cross-shape MMQ byte-invariance regression gate.
//
// Asserts that for ggml_mul_mat(Q4_0 weight [K, N], F32 input [K, ne11])
// the output dst[:, j] is byte-identical to dst[:, 0] of a separate
// mul_mat with the same input column at ne11=1. Asserts this at every
// j ∈ [0..ne11) and for several (K, N) shape pairs that fire different
// MMQ tile dispatches (I=8 path vs general path).
//
// Background — why this test exists:
// The MMQ I=8 split-K kernel (mul_mat_q_split_k_i8 with mma_int_C_I8J8
// fragment) was byte-shape-invariant for OUTPUT column 0 only. Cols
// j>=1 in a multi-token same-slot batch produced different fp32 bits
// than the same input vector would produce as col 0 of a single-token
// dispatch. The kernel's NPC verification compared col 0 across
// concurrent multi-slot single-token dispatches — every slot was col 0
// of its own n=1 call, so the col-j>0 path was structurally untested.
// This test exercises that axis directly.
//
// Production crashed visibly with P0.A.3: DFlash CLI verify-batches at
// n_tokens=5 same-slot produced incoherent text. Fixed at mmq.cuh:5012
// by setting i8_shape_supported = false.
//
// If a future kernel re-enables I=8 or introduces an analogous
// fragment-FMA-order bug in a sibling MMQ tile, this test catches it
// before it ships.

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

static constexpr int QK = 32;  // Q4_0 block elements

// Run mul_mat(weight[N, K] Q4_0, input[K, n_tokens] F32) -> dst[N, n_tokens] F32.
static bool run_mul_mat(ggml_backend_t backend,
                        int K, int N, int n_tokens,
                        const std::vector<uint8_t> & weight_q4_0_bytes,
                        const std::vector<float> & input_f32,
                        std::vector<float> & out_dst) {
    static const size_t mem_size = 16 * 1024 * 1024;
    ggml_init_params params = { mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n_tokens);
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }

    ggml_backend_tensor_set(w, weight_q4_0_bytes.data(), 0, weight_q4_0_bytes.size());
    ggml_backend_tensor_set(x, input_f32.data(),         0, input_f32.size() * sizeof(float));

    const auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph_compute failed status=%d at K=%d N=%d n_tokens=%d\n",
                (int) status, K, N, n_tokens);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    out_dst.assign((size_t) ggml_nelements(y), 0.0f);
    ggml_backend_tensor_get(y, out_dst.data(), 0, out_dst.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// Quantise a row-major F32 weight matrix [N, K] to Q4_0.
static bool quantise_q4_0(int K, int N,
                          const std::vector<float> & w_f32,
                          std::vector<uint8_t> & w_q4_0) {
    const size_t row_bytes = (size_t)(K / ggml_blck_size(GGML_TYPE_Q4_0))
                              * ggml_type_size(GGML_TYPE_Q4_0);
    w_q4_0.assign((size_t) N * row_bytes, 0);
    for (int r = 0; r < N; ++r) {
        const int64_t produced = ggml_quantize_chunk(
            GGML_TYPE_Q4_0,
            w_f32.data() + (size_t) r * K,
            w_q4_0.data() + (size_t) r * row_bytes,
            /*start_row=*/0, /*nrows=*/1, /*n_per_row=*/K,
            /*imatrix=*/nullptr,
            /*user_data=*/nullptr);
        if (produced <= 0) {
            fprintf(stderr, "ggml_quantize_chunk failed at row=%d (K=%d N=%d)\n", r, K, N);
            return false;
        }
    }
    return true;
}

// Count fp32-bit differences between two N-length vectors.
static int count_bit_diffs(const float * a, const float * b, int n, float & max_abs) {
    int n_diff = 0;
    max_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        uint32_t ai, bi;
        std::memcpy(&ai, &a[i], 4); std::memcpy(&bi, &b[i], 4);
        if (ai != bi) {
            ++n_diff;
            const float d = std::abs(a[i] - b[i]);
            if (d > max_abs) max_abs = d;
        }
    }
    return n_diff;
}

struct ShapeCase {
    int K;
    int N;
    const char * label;
};

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed; SKIP\n");
        return 77;
    }

    // Shape sweep. The K=5120 N=8192 pair is the production qkv-combined
    // shape that surfaced P0.A.3; the smaller pair sweeps cheap variants
    // to catch tile-boundary regressions in sibling MMQ paths.
    const std::vector<ShapeCase> shapes = {
        { 5120, 8192, "prod-qkv-combined" },
        { 5120, 5120, "prod-model-dim" },
        { 2048, 2048, "small-square" },
    };

    // ne11 sweep — binding region. Production decode uses ne11 = 1
    // (greedy decode per slot) and ne11 in {2..8} for DFlash
    // verify-batches. ne11 = 16 is a margin above the production max.
    // ne11 = 32 was originally informational (the cross-mmq_x dispatch
    // bug produced ULP variance there) but became binding 2026-05-21
    // after split_k_factor was unified across all mmq_x — see
    // test-mulmat-mmq_x-dispatch-invariance.cpp for the dedicated
    // cross-tile gate.
    const std::vector<int> ne11_binding_sweep    = { 1, 2, 5, 8, 16, 32 };
    constexpr int N_INPUT_COLS = 32;
    static_assert(N_INPUT_COLS >= 32, "input cols must cover ne11_max");

    int total_fails = 0;
    int total_passes = 0;

    for (const auto & shape : shapes) {
        const int K = shape.K;
        const int N = shape.N;
        if (K % QK != 0) {
            fprintf(stderr, "[SKIP] K=%d not multiple of QK=%d for shape '%s'\n",
                    K, QK, shape.label);
            continue;
        }

        // Synthesise + quantise the weight matrix for this shape.
        std::vector<float> w_f32((size_t) K * N);
        std::mt19937_64 rng_w((uint64_t) 0xF1ULL ^ ((uint64_t) K << 16) ^ (uint64_t) N);
        for (size_t i = 0; i < w_f32.size(); ++i) {
            const uint32_t r = (uint32_t)(rng_w() & 0xffffffffULL);
            w_f32[i] = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 0.1f;
        }
        std::vector<uint8_t> w_q4_0;
        if (!quantise_q4_0(K, N, w_f32, w_q4_0)) {
            fprintf(stderr, "[FAIL] quantise failed for shape '%s'\n", shape.label);
            ggml_backend_free(backend);
            return 1;
        }

        // Synthesise N_INPUT_COLS independent input columns.
        std::vector<float> x_full((size_t) K * N_INPUT_COLS);
        std::mt19937_64 rng_x((uint64_t) 0xF2ULL ^ ((uint64_t) K << 16) ^ (uint64_t) N);
        for (size_t i = 0; i < x_full.size(); ++i) {
            const uint32_t r = (uint32_t)(rng_x() & 0xffffffffULL);
            x_full[i] = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 1.0f;
        }

        // Reference dispatch: run mul_mat at ne11=1 once per input column
        // we care about. Output col is the ground-truth for that input.
        std::vector<std::vector<float>> ref_per_col(N_INPUT_COLS);
        for (int c = 0; c < N_INPUT_COLS; ++c) {
            std::vector<float> x_one(x_full.begin() + (size_t) c * K,
                                     x_full.begin() + (size_t) (c + 1) * K);
            if (!run_mul_mat(backend, K, N, 1, w_q4_0, x_one, ref_per_col[c])) {
                fprintf(stderr, "[FAIL] reference n=1 dispatch failed for col=%d shape='%s'\n",
                        c, shape.label);
                ggml_backend_free(backend);
                return 1;
            }
        }

        auto run_and_compare = [&](int ne11, bool binding) {
            if (ne11 > N_INPUT_COLS) return;
            std::vector<float> x_batch(x_full.begin(),
                                       x_full.begin() + (size_t) ne11 * K);
            std::vector<float> y_batch;
            if (!run_mul_mat(backend, K, N, ne11, w_q4_0, x_batch, y_batch)) {
                fprintf(stderr, "[FAIL] dispatch failed for shape='%s' ne11=%d\n",
                        shape.label, ne11);
                if (binding) ++total_fails;
                return;
            }
            if ((int) y_batch.size() != ne11 * N) {
                fprintf(stderr, "[FAIL] y_batch size %zu != ne11*N = %d\n",
                        y_batch.size(), ne11 * N);
                if (binding) ++total_fails;
                return;
            }

            int n_cols_bad = 0;
            int n_cols_ok  = 0;
            float worst_abs = 0.0f;
            int worst_col = -1;
            for (int j = 0; j < ne11; ++j) {
                float max_abs = 0.0f;
                const int diffs = count_bit_diffs(ref_per_col[j].data(),
                                                  y_batch.data() + (size_t) j * N,
                                                  N, max_abs);
                if (diffs == 0) {
                    ++n_cols_ok;
                } else {
                    ++n_cols_bad;
                    if (max_abs > worst_abs) { worst_abs = max_abs; worst_col = j; }
                }
            }
            const char * tag = binding ? "binding" : "info   ";
            if (n_cols_bad == 0) {
                fprintf(stderr, "  [%s ok]   shape='%s' ne11=%2d : all %d cols byte-identical\n",
                        tag, shape.label, ne11, n_cols_ok);
                if (binding) ++total_passes;
            } else {
                fprintf(stderr, "  [%s %s] shape='%s' ne11=%2d : %d/%d cols diverge, worst col=%d max|Δ|=%.3e\n",
                        tag, binding ? "FAIL" : "diff", shape.label, ne11,
                        n_cols_bad, ne11, worst_col, worst_abs);
                if (binding) ++total_fails;
            }
        };

        for (int ne11 : ne11_binding_sweep)   run_and_compare(ne11, /*binding=*/true);
    }

    ggml_backend_free(backend);

    fprintf(stderr, "\n[summary] %d/%d shape×ne11 cases pass\n",
            total_passes, total_passes + total_fails);
    if (total_fails == 0) {
        printf("[PASS] mul_mat(Q4_0) batch-shape-invariant across all swept shapes "
               "and ne11 in {1,2,5,8,16,32}.\n");
        return 0;
    }
    printf("[FAIL] mul_mat(Q4_0) batch-shape-variant in %d case(s). "
           "MMQ kernel regression — see stderr.\n", total_fails);
    return 1;
}
