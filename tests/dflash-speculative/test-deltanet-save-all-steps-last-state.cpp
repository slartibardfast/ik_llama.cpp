// test-deltanet-save-all-steps-last-state.cpp
//
// K1 binding test for the save_per_step_ssm / save_all_steps divergence
// narrowed by the P0.A.3 bisect ladder (see PHASE_NSTREAM_KV_PERF.md
// "P0.A.3 test ladder for save_per_step_ssm / save_all_steps binding").
//
// Self-consistency invariant under test:
//
//   For the same Q/K/V/G/Beta/state inputs and the same n_tokens,
//   ggml_delta_net with save_all_steps=true must produce a LAST-step
//   per-step state buffer that is byte-identical (fp32 bit-pattern) to
//   the FINAL state buffer produced with save_all_steps=false.
//
// Why this must hold: the recurrent body inside `delta_net_recurrent_f32`
// at delta-net.cu:117-184 is the SAME sequential update in both modes —
// the only difference is whether per-step state is written to dst after
// every iteration, or only the final state is written after the loop.
// The math is identical. The state at the end of iteration n_tokens-1
// in save_all_steps=true MUST be the same bit-pattern as the state at
// the end of the loop in save_all_steps=false. Any mismatch is a kernel
// bug.
//
// Production geometry (Qwen 3.6 27B DeltaNet decoder verify-batch shape):
//   HEAD_DIM = 128
//   H_V      = 16
//   H_K      = 2   (gqa_ratio = H_V / H_K = 8)
//   n_tokens = 5   (DFlash verify batch: anchor + 4 drafts)
//   n_seqs   = 1
//
// Layout of dst (per delta-net.cu kernel + ggml.c:9989-9993):
//   output region:   [0,                                 output_offset)
//     output_offset = HEAD_DIM * n_tokens * H_V * n_seqs
//   state region:
//     save_all_steps = false:
//       final state at [output_offset, output_offset + state_size)
//       state_size = HEAD_DIM * HEAD_DIM * H_V * n_seqs
//     save_all_steps = true:
//       per-step states at [output_offset + t * state_step_stride, ...)
//       state_step_stride = HEAD_DIM * HEAD_DIM * H_V * n_seqs
//
// Returns: 0 = PASS, 1 = FAIL (binds: last-step state != final state),
//          77 = SKIP (no CUDA).

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

static constexpr int HEAD_DIM = 128;
static constexpr int H_V      = 16;
static constexpr int H_K      = 2;     // gqa_ratio = 8
static constexpr int N_TOKENS = 5;
static constexpr int N_SEQS   = 1;

static void fill_seeded(std::vector<float> & buf, uint64_t seed, float scale) {
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i < buf.size(); ++i) {
        const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
        buf[i] = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * scale;
    }
}

// Layout for n_seqs=1: the n_tokens per-step states live in successive
// state_step_stride floats after output_offset. Returns the full raw
// dst buffer; caller slices via the offsets we expose.
static bool run_one(ggml_backend_t backend, bool save_all_steps,
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

    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, N_TOKENS, H_K, N_SEQS);
    ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, N_TOKENS, H_K, N_SEQS);
    ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, N_TOKENS, H_V, N_SEQS);
    ggml_tensor * g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, N_TOKENS, 1, H_V, N_SEQS);
    ggml_tensor * b     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, N_TOKENS, H_V, N_SEQS);
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, HEAD_DIM * H_V, 1, N_SEQS);

    ggml_tensor * out_t = ggml_delta_net(ctx, q, k, v, g, b, state, save_all_steps);
    ggml_set_name(out_t, save_all_steps ? "delta_net_save_all" : "delta_net_final_only");

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
        fprintf(stderr, "graph_compute failed (status=%d) save_all_steps=%d\n",
                (int)status, save_all_steps ? 1 : 0);
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

static int count_bit_diffs(const float * a, const float * b, size_t n, float & max_abs) {
    int n_diff = 0;
    max_abs = 0.0f;
    for (size_t j = 0; j < n; ++j) {
        uint32_t ai, bi;
        std::memcpy(&ai, &a[j], 4);
        std::memcpy(&bi, &b[j], 4);
        if (ai != bi) {
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

    const size_t qk_total = (size_t)HEAD_DIM * N_TOKENS * H_K * N_SEQS;
    const size_t v_total  = (size_t)HEAD_DIM * N_TOKENS * H_V * N_SEQS;
    const size_t g_total  = (size_t)N_TOKENS * 1 * H_V * N_SEQS;
    const size_t b_total  = (size_t)1 * N_TOKENS * H_V * N_SEQS;
    const size_t s_total  = (size_t)HEAD_DIM * HEAD_DIM * H_V * N_SEQS;

    std::vector<float> Q(qk_total), K(qk_total), V(v_total);
    std::vector<float> G(g_total),  B(b_total),  S(s_total);
    fill_seeded(Q, 0xD1ULL, 0.5f);
    fill_seeded(K, 0xD2ULL, 0.5f);
    fill_seeded(V, 0xD3ULL, 0.5f);
    fill_seeded(G, 0xD4ULL, 0.1f);   // gate values — small (decay near 1).
    fill_seeded(B, 0xD5ULL, 0.5f);
    fill_seeded(S, 0xD6ULL, 0.05f);  // recurrent state — small magnitude.

    std::vector<float> dst_save_all, dst_final_only;
    if (!run_one(backend, /*save_all_steps=*/true,  Q,K,V,G,B,S, dst_save_all))   { ggml_backend_free(backend); return 1; }
    if (!run_one(backend, /*save_all_steps=*/false, Q,K,V,G,B,S, dst_final_only)) { ggml_backend_free(backend); return 1; }
    ggml_backend_free(backend);

    // Output region size and offset to state region.
    const size_t output_floats     = (size_t)HEAD_DIM * N_TOKENS * H_V * N_SEQS;
    const size_t state_step_floats = (size_t)HEAD_DIM * HEAD_DIM * H_V * N_SEQS;

    // Sanity: dst sizes match the documented layout.
    const size_t expect_save_all   = output_floats + (size_t)N_TOKENS * state_step_floats;
    const size_t expect_final_only = output_floats + state_step_floats;
    if (dst_save_all.size() != expect_save_all || dst_final_only.size() != expect_final_only) {
        fprintf(stderr,
                "[FAIL] dst size mismatch: save_all=%zu (expected %zu), "
                "final_only=%zu (expected %zu)\n",
                dst_save_all.size(), expect_save_all,
                dst_final_only.size(), expect_final_only);
        return 1;
    }

    // Gate A: the OUTPUT region (logits-equivalent path) must match between
    // modes. If it doesn't, the kernel's per-token output is also being
    // perturbed, not just the state writes. Useful to know separately.
    {
        float ma = 0.0f;
        const int nd = count_bit_diffs(dst_save_all.data(), dst_final_only.data(),
                                       output_floats, ma);
        if (nd == 0) {
            fprintf(stderr, "[gate-A output] byte-identical: %zu fp32 floats\n",
                    output_floats);
        } else {
            fprintf(stderr,
                    "[gate-A output] DIFFERS: %d/%zu floats (max |Δ|=%.3e). "
                    "Per-token output writes diverge between save_all_steps modes — "
                    "the recurrence is being reordered, not just the state write.\n",
                    nd, output_floats, ma);
        }
    }

    // Gate B (THE BINDING GATE): last per-step state from save_all=true must
    // byte-equal final state from save_all=false.
    const float * last_step_state = dst_save_all.data() + output_floats
                                    + (size_t)(N_TOKENS - 1) * state_step_floats;
    const float * final_state     = dst_final_only.data() + output_floats;

    fprintf(stderr, "last_step_state first 8: ");
    for (int k = 0; k < 8; ++k) fprintf(stderr, " %+10.6f", last_step_state[k]);
    fprintf(stderr, "\nfinal_state     first 8: ");
    for (int k = 0; k < 8; ++k) fprintf(stderr, " %+10.6f", final_state[k]);
    fprintf(stderr, "\n");

    float ma_b = 0.0f;
    const int nd_b = count_bit_diffs(last_step_state, final_state, state_step_floats, ma_b);
    if (nd_b == 0) {
        fprintf(stderr,
                "[gate-B state] PASS — last per-step state (save_all_steps=true) "
                "byte-identical to final state (save_all_steps=false): %zu fp32 floats\n",
                state_step_floats);
        printf("[PASS] save_all_steps self-consistency: last per-step state == "
               "final-only state, byte-identical at production geometry\n");
        return 0;
    }
    fprintf(stderr,
            "[gate-B state] FAIL — last per-step state (save_all_steps=true) "
            "does NOT match final state (save_all_steps=false): %d/%zu floats differ "
            "(max |Δ|=%.3e)\n",
            nd_b, state_step_floats, ma_b);
    printf("[FAIL] save_all_steps self-consistency: kernel produces different "
           "fp32 bit-pattern for the same algebraic state — bug binding\n");
    return 1;
}
