// test-rope-batch-shape-invariance.cpp
//
// Probes whether the CUDA RoPE op produces byte-identical token-0 output
// across batch shapes (ne[2] = n_tok ∈ {1, 2, 4, 8}). Each (token, head,
// dim-pair) is algebraically independent of other tokens; this test
// binds that fact at the bit level for the production shape
// (head_dim=128, n_heads_q=24, NEOX mode, Qwen 3.6 freq_base).
//
// Shape contract: input ne = [head_dim, n_heads, n_tok, n_seqs]; ggml.c
// line 9163 GGML_ASSERTs `a->ne[2] == b->ne[0]` for non-mrope, so the
// position vector is indexed by ne[2]. Output is the same shape as input.
// Token 0 occupies the first `head_dim * n_heads` floats of the buffer.
//
// Returns: 0 = PASS (byte-identical), 1 = FAIL, 77 = SKIP.
//
// Phase CX.C per PHASE_MMQ_Q4_0_AR16.md §6b.

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
static constexpr int N_HEADS_Q  = 24;
static constexpr int N_SEQS     = 1;
static constexpr int N_TOK_MAX  = 8;
static constexpr float FREQ_BASE   = 1000000.0f;  // Qwen 3.6 theta
static constexpr float FREQ_SCALE  = 1.0f;
static constexpr float EXT_FACTOR  = 0.0f;
static constexpr float ATTN_FACTOR = 1.0f;
static constexpr float BETA_FAST   = 32.0f;
static constexpr float BETA_SLOW   = 1.0f;
static constexpr int   N_CTX_ORIG  = 32768;

static bool run_one(ggml_backend_t backend,
                    int n_tok,
                    const std::vector<float>   & X_tok0,    // [HEAD_DIM * N_HEADS_Q]
                    const std::vector<int32_t> & POS_buf,   // [N_TOK_MAX] (pos[0]=0 always)
                    uint64_t extra_seed,
                    std::vector<float> & tok0_out) {
    static const size_t mem_size = 4 * 1024 * 1024;
    ggml_init_params params = { mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * x   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, N_HEADS_Q, n_tok, N_SEQS);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tok);
    ggml_tensor * out = ggml_rope_ext(ctx, x, pos, /*freq_factors=*/nullptr,
                                      HEAD_DIM, GGML_ROPE_TYPE_NEOX, N_CTX_ORIG,
                                      FREQ_BASE, FREQ_SCALE, EXT_FACTOR, ATTN_FACTOR,
                                      BETA_FAST, BETA_SLOW);
    ggml_set_name(out, "out");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }

    // Fill x: token 0 from X_tok0; tokens 1..n_tok-1 deterministic-random per extra_seed.
    {
        std::vector<float> X_buf((size_t)HEAD_DIM * N_HEADS_Q * n_tok * N_SEQS);
        std::mt19937_64 rng(extra_seed ^ 0xfeed1234ULL);
        for (int s = 0; s < N_SEQS; ++s) {
            for (int t = 0; t < n_tok; ++t) {
                for (int h = 0; h < N_HEADS_Q; ++h) {
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        const size_t idx = (size_t)d
                                         + (size_t)h * HEAD_DIM
                                         + (size_t)t * HEAD_DIM * N_HEADS_Q
                                         + (size_t)s * HEAD_DIM * N_HEADS_Q * n_tok;
                        if (t == 0) {
                            const size_t r0_idx = (size_t)d + (size_t)h * HEAD_DIM;
                            X_buf[idx] = X_tok0[r0_idx];
                        } else {
                            const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
                            X_buf[idx] = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 0.5f;
                        }
                    }
                }
            }
        }
        ggml_backend_tensor_set(x, X_buf.data(), 0, X_buf.size() * sizeof(float));
    }

    // pos: fixed [0, 1, 2, ..., n_tok-1] (pos[0]=0 across all runs).
    {
        ggml_backend_tensor_set(pos, POS_buf.data(), 0, (size_t)n_tok * sizeof(int32_t));
    }

    const auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph_compute failed (status=%d) at n_tok=%d\n", (int)status, n_tok);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    // Extract token 0 output: ne = [HEAD_DIM, N_HEADS_Q, n_tok, N_SEQS].
    // Token 0 is at offsets 0..HEAD_DIM*N_HEADS_Q-1 (then jumps to next seq).
    std::vector<float> dst_full((size_t)HEAD_DIM * N_HEADS_Q * n_tok * N_SEQS);
    ggml_backend_tensor_get(out, dst_full.data(), 0, dst_full.size() * sizeof(float));

    tok0_out.assign((size_t)HEAD_DIM * N_HEADS_Q * N_SEQS, 0.0f);
    for (int s = 0; s < N_SEQS; ++s) {
        for (int h = 0; h < N_HEADS_Q; ++h) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                const size_t full_idx = (size_t)d
                                      + (size_t)h * HEAD_DIM
                                      + 0 * (size_t)HEAD_DIM * N_HEADS_Q
                                      + (size_t)s * HEAD_DIM * N_HEADS_Q * n_tok;
                const size_t tok0_idx = (size_t)d + (size_t)h * HEAD_DIM + (size_t)s * HEAD_DIM * N_HEADS_Q;
                tok0_out[tok0_idx] = dst_full[full_idx];
            }
        }
    }

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

    std::vector<float> X_tok0((size_t)HEAD_DIM * N_HEADS_Q);
    {
        std::mt19937_64 rng(0xC0FFEEULL);
        for (auto & v : X_tok0) {
            const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
            v = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 0.5f;
        }
    }

    std::vector<int32_t> POS_buf(N_TOK_MAX);
    for (int t = 0; t < N_TOK_MAX; ++t) POS_buf[t] = t;

    const std::vector<int> n_toks = {1, 2, 4, 8};
    std::vector<std::vector<float>> outs;
    for (size_t i = 0; i < n_toks.size(); ++i) {
        std::vector<float> out;
        const uint64_t extra_seed = 0x55AA00ULL + (uint64_t)(i + 1) * 0x1000ULL;
        if (!run_one(backend, n_toks[i], X_tok0, POS_buf, extra_seed, out)) {
            ggml_backend_free(backend);
            return 1;
        }
        outs.push_back(std::move(out));
    }

    ggml_backend_free(backend);

    fprintf(stderr, "token-0 output first 8 floats per n_tok:\n");
    for (size_t i = 0; i < n_toks.size(); ++i) {
        fprintf(stderr, "  n_tok=%d:", n_toks[i]);
        for (int k = 0; k < 8; ++k) fprintf(stderr, " %+10.6f", outs[i][k]);
        fprintf(stderr, "\n");
    }

    int total_diffs = 0;
    const auto & ref = outs[0];
    for (size_t i = 1; i < outs.size(); ++i) {
        const auto & cur = outs[i];
        int diffs_here = 0;
        float max_abs = 0.0f;
        for (size_t j = 0; j < ref.size(); ++j) {
            uint32_t a, b;
            std::memcpy(&a, &ref[j], sizeof(a));
            std::memcpy(&b, &cur[j], sizeof(b));
            if (a != b) {
                ++diffs_here;
                const float d = std::abs(ref[j] - cur[j]);
                if (d > max_abs) max_abs = d;
            }
        }
        if (diffs_here > 0) {
            fprintf(stderr, "  n_tok=%d: %d/%zu values differ from n_tok=1 baseline (max |Δ| = %.3e)\n",
                    n_toks[i], diffs_here, ref.size(), max_abs);
            total_diffs += diffs_here;
        } else {
            fprintf(stderr, "  n_tok=%d: byte-identical to n_tok=1 baseline\n", n_toks[i]);
        }
    }

    if (total_diffs == 0) {
        fprintf(stderr, "OVERALL: PASS — token 0 byte-identical across n_tok ∈ {1, 2, 4, 8}\n");
        return 0;
    } else {
        fprintf(stderr, "OVERALL: FAIL — %d byte mismatches in RoPE token 0 across batch shape\n", total_diffs);
        return 1;
    }
}
