// test-deltanet-save-all-steps-intermediate.cpp
//
// K1' kernel-direct intermediate-state equivalence test for the P0.A.3
// Suspect 4 (per_step_restore semantics) diagnosis. Extends K1
// (test-deltanet-save-all-steps-last-state) to verify that the kernel's
// per_step_ssm[k] for ALL k ∈ {0, ..., n_tokens-1} matches the state a
// fresh n_tokens=k+1 run would produce — not just the last step.
//
// Why this matters: K1 verified per_step_ssm[n-1] == final-state at
// n_tokens=5 (PASSED). per_step_restore reads per_step_ssm[step] for
// arbitrary step ∈ [0, accepted_step], so the WHOLE per-step buffer
// must be correct, not just the last entry. This test binds the kernel
// invariant that per_step_ssm[k] equals "state at the end of step k"
// in the canonical recurrence sense — i.e., the SAME state that a
// fresh run with n_tokens=k+1 and identical inputs at positions [0..k]
// would produce as its FINAL state.
//
// Production geometry (Qwen 3.6 27B DeltaNet verify-batch):
//   HEAD_DIM = 128
//   H_V      = 16
//   H_K      = 2   (gqa_ratio = 8)
//   N_TOKENS = 5
//   N_SEQS   = 1
//
// Algorithm:
//   1. Generate seeded inputs Q/K/V/G/Beta/state large enough for n=5.
//   2. Run A: n_tokens=5 with save_all_steps=true. Capture per_step[k]
//      for k ∈ {0, 1, 2, 3, 4}.
//   3. For each k ∈ {0, 1, 2, 3, 4}:
//      Run B_k: n_tokens=k+1 with save_all_steps=false. Use Q/K/V/G/Beta
//      tensors sized [_, k+1, _, _] populated from the SAME seeded source
//      at positions [0..k]. Same state input. Capture final_state.
//      Assert per_step[k] from A byte-equal to final_state from B_k.
//
// Predicted: PASS at every k. The kernel's recurrence body at
// delta-net.cu:117-184 is deterministic on state_local + inputs[t]; the
// state at end-of-step t doesn't depend on whether more iterations
// follow. If it FAILS at any k, the kernel writes per-step state at the
// wrong loop iteration (e.g., before the state_local update at line
// 155-159 rather than after), and per_step_restore reads a stale value.
//
// Returns: 0 = PASS at all k, 1 = FAIL at any k, 77 = SKIP (no CUDA).

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

static constexpr int HEAD_DIM   = 128;
static constexpr int H_V        = 16;
static constexpr int H_K        = 2;
static constexpr int N_TOKENS_5 = 5;
static constexpr int N_SEQS     = 1;

static void fill_seeded(std::vector<float> & buf, uint64_t seed, float scale) {
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i < buf.size(); ++i) {
        const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
        buf[i] = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * scale;
    }
}

// Copy the first n_tokens token slots of a `[D, N_TOKENS_5, H, n_seqs]`
// column-major buffer into a `[D, n_tokens, H, n_seqs]` column-major
// buffer.
//
// Source layout: contiguous, ne=[D, N_TOKENS_5, H, n_seqs], stride0=1,
//                stride1=D, stride2=D*N_TOKENS_5, stride3=D*N_TOKENS_5*H.
// Dest layout:   contiguous, ne=[D, n_tokens, H, n_seqs], stride0=1,
//                stride1=D, stride2=D*n_tokens, stride3=D*n_tokens*H.
//
// For each seq s, head h, token t in [0..n_tokens), copy D floats from
// src[..., t, h, s] to dst[..., t, h, s].
static void slice_first_n_tokens(const std::vector<float> & src,
                                 std::vector<float> & dst,
                                 int D, int H, int n_seqs, int n_tokens) {
    dst.assign((size_t)D * n_tokens * H * n_seqs, 0.0f);
    for (int s = 0; s < n_seqs; ++s) {
        for (int h = 0; h < H; ++h) {
            for (int t = 0; t < n_tokens; ++t) {
                const size_t src_off = (size_t)D *
                    ((size_t)t + (size_t)N_TOKENS_5 * ((size_t)h + (size_t)H * s));
                const size_t dst_off = (size_t)D *
                    ((size_t)t + (size_t)n_tokens * ((size_t)h + (size_t)H * s));
                std::memcpy(&dst[dst_off], &src[src_off], (size_t)D * sizeof(float));
            }
        }
    }
}

// G/beta layout note: the kernel reads `g_in[h + t*n_heads + s*n_tokens*n_heads]`
// and `beta_in[h + t*n_heads + s*n_tokens*n_heads]` (h is the FAST axis, t at
// stride n_heads). Production supplies these via ggml_permute on a tensor that
// is originally contiguous in a different order; the permute makes the
// "h-fast, t-second" interpretation match what the kernel's precomputed
// strides expect. The kernel uses precomputed strides (g_stride_batch =
// n_tokens * n_heads) rather than consulting nb[], so the BYTE LAYOUT in the
// supplied buffer is what matters — not the ggml-contiguous ne[] order.
//
// For K1' to compare n=5 vs n=k+1 runs consistently, we slice in the kernel's
// coordinate system: the first (k+1) tokens × n_heads floats are simply the
// first (k+1)*n_heads contiguous floats of the seq=0 slice in the n=5 buffer.
// (Per-seq slabs are size n_tokens*n_heads; n_seqs=1 here, so trivially the
// first (k+1)*n_heads floats of the whole buffer.)
static void slice_first_n_tokens_h_fast(const std::vector<float> & src,
                                        std::vector<float> & dst,
                                        int H, int n_seqs, int n_tokens) {
    const size_t per_seq_src = (size_t)N_TOKENS_5 * H;
    const size_t per_seq_dst = (size_t)n_tokens   * H;
    dst.assign(per_seq_dst * n_seqs, 0.0f);
    for (int s = 0; s < n_seqs; ++s) {
        // Kernel layout: for token t and head h, value at offset
        //   s*n_tokens*H + t*H + h
        // Slicing first n_tokens tokens means copying the leading
        // n_tokens*H floats of this seq's slab.
        std::memcpy(&dst[per_seq_dst * s],
                    &src[per_seq_src * s],
                    per_seq_dst * sizeof(float));
    }
}

static bool run_one(ggml_backend_t backend, int n_tokens, bool save_all_steps,
                    const std::vector<float> & Q,
                    const std::vector<float> & K,
                    const std::vector<float> & V,
                    const std::vector<float> & G,
                    const std::vector<float> & B,
                    const std::vector<float> & S,
                    std::vector<float> & out_dst) {
    static const size_t mem_size = 16 * 1024 * 1024;
    ggml_init_params params = { mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, n_tokens, H_K, N_SEQS);
    ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, n_tokens, H_K, N_SEQS);
    ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, n_tokens, H_V, N_SEQS);
    ggml_tensor * g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_tokens, 1, H_V, N_SEQS);
    ggml_tensor * b     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, n_tokens, H_V, N_SEQS);
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, HEAD_DIM * H_V, 1, N_SEQS);

    ggml_tensor * out_t = ggml_delta_net(ctx, q, k, v, g, b, state, save_all_steps);
    ggml_set_name(out_t,
        save_all_steps ? "delta_net_save_all_n5" : "delta_net_final_only_nk");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_t);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }

    ggml_backend_tensor_set(q,     Q.data(), 0, Q.size()*sizeof(float));
    ggml_backend_tensor_set(k,     K.data(), 0, K.size()*sizeof(float));
    ggml_backend_tensor_set(v,     V.data(), 0, V.size()*sizeof(float));
    ggml_backend_tensor_set(g,     G.data(), 0, G.size()*sizeof(float));
    ggml_backend_tensor_set(b,     B.data(), 0, B.size()*sizeof(float));
    ggml_backend_tensor_set(state, S.data(), 0, S.size()*sizeof(float));

    const auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph_compute failed status=%d n_tokens=%d save_all=%d\n",
                (int)status, n_tokens, save_all_steps ? 1 : 0);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    const size_t n_floats = (size_t) ggml_nelements(out_t);
    out_dst.assign(n_floats, 0.0f);
    ggml_backend_tensor_get(out_t, out_dst.data(), 0, n_floats * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static int count_bit_diffs(const float * a, const float * b, size_t n, float & max_abs,
                           int & first_diff_idx) {
    int n_diff = 0;
    max_abs = 0.0f;
    first_diff_idx = -1;
    for (size_t j = 0; j < n; ++j) {
        uint32_t ai, bi;
        std::memcpy(&ai, &a[j], 4);
        std::memcpy(&bi, &b[j], 4);
        if (ai != bi) {
            if (first_diff_idx < 0) first_diff_idx = (int) j;
            ++n_diff;
            const float d = std::abs(a[j] - b[j]);
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

    // Generate full-size n=5 inputs once.
    const size_t qk_n5 = (size_t)HEAD_DIM * N_TOKENS_5 * H_K * N_SEQS;
    const size_t v_n5  = (size_t)HEAD_DIM * N_TOKENS_5 * H_V * N_SEQS;
    const size_t g_n5  = (size_t)N_TOKENS_5 * 1 * H_V * N_SEQS;
    const size_t b_n5  = (size_t)1 * N_TOKENS_5 * H_V * N_SEQS;
    const size_t s_n5  = (size_t)HEAD_DIM * HEAD_DIM * H_V * N_SEQS;

    std::vector<float> Q5(qk_n5), K5(qk_n5), V5(v_n5);
    std::vector<float> G5(g_n5),  B5(b_n5),  S5(s_n5);
    fill_seeded(Q5, 0xE1ULL, 0.5f);
    fill_seeded(K5, 0xE2ULL, 0.5f);
    fill_seeded(V5, 0xE3ULL, 0.5f);
    fill_seeded(G5, 0xE4ULL, 0.1f);
    fill_seeded(B5, 0xE5ULL, 0.5f);
    fill_seeded(S5, 0xE6ULL, 0.05f);

    // Run A: n_tokens=5, save_all_steps=true.
    std::vector<float> dst_save_all;
    if (!run_one(backend, N_TOKENS_5, /*save_all_steps=*/true,
                 Q5, K5, V5, G5, B5, S5, dst_save_all)) {
        ggml_backend_free(backend);
        return 1;
    }

    const size_t output_floats_n5  = (size_t)HEAD_DIM * N_TOKENS_5 * H_V * N_SEQS;
    const size_t state_step_floats = (size_t)HEAD_DIM * HEAD_DIM * H_V * N_SEQS;

    // Sanity on dst size.
    const size_t expect_save_all = output_floats_n5 + (size_t)N_TOKENS_5 * state_step_floats;
    if (dst_save_all.size() != expect_save_all) {
        fprintf(stderr, "[FAIL] save_all dst size = %zu, expected %zu\n",
                dst_save_all.size(), expect_save_all);
        ggml_backend_free(backend);
        return 1;
    }

    // Run B_k for k ∈ {0, 1, 2, 3, 4}.
    int total_failures = 0;
    for (int k = 0; k < N_TOKENS_5; ++k) {
        const int n_tokens_k = k + 1;

        std::vector<float> Qk, Kk, Vk, Gk, Bk;
        // Q/K/V: HEAD_DIM is the kernel's fast axis, then t at stride HEAD_DIM,
        // then h at stride HEAD_DIM*n_tokens. Both the ggml-contiguous
        // interpretation AND the kernel's precomputed strides agree on this
        // layout, so the standard "slice first n tokens per head" copy works.
        slice_first_n_tokens       (Q5, Qk, HEAD_DIM, H_K, N_SEQS, n_tokens_k);
        slice_first_n_tokens       (K5, Kk, HEAD_DIM, H_K, N_SEQS, n_tokens_k);
        slice_first_n_tokens       (V5, Vk, HEAD_DIM, H_V, N_SEQS, n_tokens_k);
        // G and beta: h-fast layout per kernel's stride computation. Slicing
        // by memcpy of the first (n_tokens_k * H) floats matches the kernel's
        // reads at t ∈ [0, n_tokens_k).
        slice_first_n_tokens_h_fast(G5, Gk, H_V, N_SEQS, n_tokens_k);
        slice_first_n_tokens_h_fast(B5, Bk, H_V, N_SEQS, n_tokens_k);
        // State input is the same — represents the initial state at t=0.

        std::vector<float> dst_final_only;
        if (!run_one(backend, n_tokens_k, /*save_all_steps=*/false,
                     Qk, Kk, Vk, Gk, Bk, S5, dst_final_only)) {
            fprintf(stderr, "[FAIL] run_one failed at n_tokens=%d\n", n_tokens_k);
            total_failures++;
            continue;
        }

        const size_t output_floats_k = (size_t)HEAD_DIM * n_tokens_k * H_V * N_SEQS;
        const size_t expect_final_k  = output_floats_k + state_step_floats;
        if (dst_final_only.size() != expect_final_k) {
            fprintf(stderr,
                    "[FAIL] n_tokens=%d final_only dst size = %zu, expected %zu\n",
                    n_tokens_k, dst_final_only.size(), expect_final_k);
            total_failures++;
            continue;
        }

        // Compare: per_step_ssm[k] from Run A vs final_state from Run B_k.
        const float * per_step_k_from_A =
            dst_save_all.data() + output_floats_n5 + (size_t)k * state_step_floats;
        const float * final_from_Bk =
            dst_final_only.data() + output_floats_k;

        float max_abs = 0.0f;
        int first_diff = -1;
        const int n_diff = count_bit_diffs(per_step_k_from_A, final_from_Bk,
                                           state_step_floats, max_abs, first_diff);
        if (n_diff == 0) {
            fprintf(stderr,
                    "[k=%d PASS] per_step[%d] (from n=5 save_all) == final_state "
                    "(from n=%d save=false): %zu fp32 floats byte-identical\n",
                    k, k, n_tokens_k, state_step_floats);
        } else {
            fprintf(stderr,
                    "[k=%d FAIL] per_step[%d] (from n=5 save_all) DIFFERS from "
                    "final_state (from n=%d save=false): %d/%zu floats differ "
                    "(max |Δ|=%.3e, first diff at idx %d: %+.6f vs %+.6f)\n",
                    k, k, n_tokens_k, n_diff, state_step_floats, max_abs, first_diff,
                    per_step_k_from_A[first_diff], final_from_Bk[first_diff]);
            total_failures++;
        }
    }

    ggml_backend_free(backend);

    if (total_failures == 0) {
        printf("[PASS] kernel per_step_ssm[k] == fresh-decode final_state for "
               "all k ∈ {0..4} at production geometry; kernel is exonerated as "
               "a Suspect 4 source (per_step_restore source data is correct).\n");
        return 0;
    }
    printf("[FAIL] %d/%d per_step intermediate states diverge from fresh-decode "
           "reference — kernel writes incorrect per-step state at one or more "
           "intermediate iterations; this localises Suspect 4 to the kernel save "
           "side rather than the restore reconstruction.\n",
           total_failures, N_TOKENS_5);
    return 1;
}
