// test-mulmat-mmq_x-dispatch-invariance.cpp
//
// Cross-mmq_x dispatch byte-invariance regression gate.
//
// The MMQ kernel dispatcher in mmq.cuh:5168-5186 selects a compile-time
// mmq_x tile size based on ne11 (input batch tokens). Tiles in 8..128
// step 8 each instantiate a separate kernel template. The split_k_factor
// at mmq.cuh:4974 depends on mmq_x:
//
//     constexpr int split_k_factor = (mmq_x <= 16) ? 4 : 1;
//
// So ne11 ≤ 16 routes to `mul_mat_q_split_k<...,4>` (K dim split into
// 4 chunks, each computed in parallel, then summed in fp32 by a fixup
// pass), and ne11 > 16 routes to plain `mul_mat_q` (K summed in a
// single sequential pass). Mathematically equivalent; bit-different by
// ULP magnitude (~5e-6 at production K=5120).
//
// This test asserts that mul_mat output for the SAME input column at
// any ne11 is byte-identical to the output at ne11=1. Sweeps ne11
// across the dispatcher's tile boundary to catch any cross-tile
// reduction-order divergence.
//
// Current state (2026-05-21): FAILS on HEAD at ne11 ≥ 24 (every tile
// transition picks a kernel with split_k_factor=1). After fix that
// makes split_k_factor uniform across mmq_x, should PASS at every
// ne11.

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

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed; SKIP\n");
        return 77;
    }

    // Single production-relevant shape — qkv-combined dim from Qwen 3.6 27B.
    // The test asserts byte-identity at every ne11 in the sweep; testing
    // multiple shapes would multiply runtime without adding mechanism
    // coverage (the dispatcher routes purely on ne11).
    const int K = 5120;
    const int N = 8192;

    // ne11 sweep covering every dispatcher tile boundary from 8 to 128.
    // Each ne11 picks the smallest mmq_x in {8,16,24,32,40,48,56,64,...}
    // that satisfies the constraint, so this sweep exercises every
    // unique mmq_x template. The 256 and 512 cases land at mmq_x=128
    // (the max) with multiple xy-tiles; they bind the case where mmq_x
    // saturates but ne11 keeps growing.
    const std::vector<int> ne11_sweep = {
        1, 8, 16,                         // mmq_x ≤ 16 region (split-K factor=4)
        24, 32, 40, 48, 56, 64,           // mmq_x > 16 boundary cases
        72, 80, 96, 128,                  // remaining unique tiles
        256, 512                          // xy-tile-replication region
    };
    constexpr int N_INPUT_COLS_MAX = 512;

    fprintf(stderr, "[cross-mmq_x] K=%d N=%d, sweeping ne11 ∈ {",  K, N);
    for (size_t i = 0; i < ne11_sweep.size(); ++i) {
        fprintf(stderr, "%s%d", i ? "," : "", ne11_sweep[i]);
    }
    fprintf(stderr, "}\n");

    // Synthesise + quantise the weight matrix.
    std::vector<float> w_f32((size_t) K * N);
    {
        std::mt19937_64 rng(0xF1ULL);
        for (size_t i = 0; i < w_f32.size(); ++i) {
            const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
            w_f32[i] = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 0.1f;
        }
    }
    std::vector<uint8_t> w_q4_0;
    if (!quantise_q4_0(K, N, w_f32, w_q4_0)) {
        ggml_backend_free(backend); return 1;
    }
    fprintf(stderr, "[cross-mmq_x] quantised %d×%d weight to Q4_0\n", N, K);

    // Synthesise the maximum-width input matrix; smaller ne11 dispatches
    // see the leading prefix.
    std::vector<float> x_full((size_t) K * N_INPUT_COLS_MAX);
    {
        std::mt19937_64 rng(0xF2ULL);
        for (size_t i = 0; i < x_full.size(); ++i) {
            const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
            x_full[i] = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 1.0f;
        }
    }

    // Reference dispatches: one mul_mat at ne11=1 per input column. The
    // output is the ground-truth for that input column.
    // We only need references up to the max test column we'll inspect,
    // which is min(max(ne11_sweep), N_INPUT_COLS_MAX).
    int max_ne11 = 0;
    for (int v : ne11_sweep) max_ne11 = std::max(max_ne11, v);
    const int n_refs = std::min(max_ne11, N_INPUT_COLS_MAX);
    fprintf(stderr, "[cross-mmq_x] building %d single-column references...\n", n_refs);
    std::vector<std::vector<float>> ref_per_col(n_refs);
    for (int c = 0; c < n_refs; ++c) {
        std::vector<float> x_one(x_full.begin() + (size_t) c * K,
                                 x_full.begin() + (size_t) (c + 1) * K);
        if (!run_mul_mat(backend, K, N, 1, w_q4_0, x_one, ref_per_col[c])) {
            fprintf(stderr, "[FAIL] reference n=1 dispatch failed at col=%d\n", c);
            ggml_backend_free(backend);
            return 1;
        }
    }

    int total_passes = 0;
    int total_fails  = 0;

    for (int ne11 : ne11_sweep) {
        if (ne11 > N_INPUT_COLS_MAX) continue;
        std::vector<float> x_batch(x_full.begin(), x_full.begin() + (size_t) ne11 * K);
        std::vector<float> y_batch;
        if (!run_mul_mat(backend, K, N, ne11, w_q4_0, x_batch, y_batch)) {
            fprintf(stderr, "[FAIL] dispatch failed at ne11=%d\n", ne11);
            ++total_fails;
            continue;
        }
        if ((int) y_batch.size() != ne11 * N) {
            fprintf(stderr, "[FAIL] y_batch size %zu != ne11*N = %d\n",
                    y_batch.size(), ne11 * N);
            ++total_fails;
            continue;
        }

        int n_cols_bad = 0;
        float worst_abs = 0.0f;
        int worst_col = -1;
        for (int j = 0; j < ne11; ++j) {
            float max_abs = 0.0f;
            const int diffs = count_bit_diffs(ref_per_col[j].data(),
                                              y_batch.data() + (size_t) j * N,
                                              N, max_abs);
            if (diffs != 0) {
                ++n_cols_bad;
                if (max_abs > worst_abs) { worst_abs = max_abs; worst_col = j; }
            }
        }
        if (n_cols_bad == 0) {
            fprintf(stderr, "  [ok]   ne11=%4d : all %d cols byte-identical to ne11=1\n",
                    ne11, ne11);
            ++total_passes;
        } else {
            fprintf(stderr, "  [FAIL] ne11=%4d : %d/%d cols diverge, worst col=%d max|Δ|=%.3e\n",
                    ne11, n_cols_bad, ne11, worst_col, worst_abs);
            ++total_fails;
        }
    }

    ggml_backend_free(backend);

    fprintf(stderr, "\n[summary] %d/%d ne11 dispatches byte-identical to ne11=1\n",
            total_passes, total_passes + total_fails);
    if (total_fails == 0) {
        printf("[PASS] mul_mat(Q4_0) cross-mmq_x dispatch byte-invariant across the full sweep.\n");
        return 0;
    }
    printf("[FAIL] mul_mat(Q4_0) cross-mmq_x dispatch variant in %d case(s). "
           "Different mmq_x tiles produce different fp32 bits for the same input.\n",
           total_fails);
    return 1;
}
