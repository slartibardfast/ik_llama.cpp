// test-rmsnorm-batch-shape-invariance.cpp
//
// Probes whether the CUDA RMSNorm op produces byte-identical row-0 output
// across batch shapes (ne[1] ∈ {1, 2, 4, 8}). Each row's normalization is
// algebraically independent of other rows; this test binds that fact at
// the bit level for the production ncols (5120 = Qwen 3.6 27B n_embd).
//
// Per ggml_rms_norm shape contract: input/output are same shape; reduction
// is along ne[0]. So row 0 is the first ne[0] elements (the t=0 ubatch
// slot in a [n_embd, n_tokens] hidden-state tensor).
//
// Returns: 0 = PASS (byte-identical), 1 = FAIL (divergence), 77 = SKIP (no CUDA).
//
// Phase CX.B per PHASE_MMQ_Q4_0_AR16.md §6b.

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

static constexpr int N_EMBD    = 5120;   // Qwen 3.6 27B hidden dim
static constexpr int N_TOK_MAX = 8;
static constexpr float EPS     = 1e-6f;

static bool run_one(ggml_backend_t backend,
                    int n_tok,
                    const std::vector<float> & X_row0,
                    uint64_t extra_seed,
                    std::vector<float> & row0_out) {
    static const size_t mem_size = 4 * 1024 * 1024;
    ggml_init_params params = { mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N_EMBD, n_tok);
    ggml_tensor * out = ggml_rms_norm(ctx, x, EPS);
    ggml_set_name(out, "out");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }

    // Fill x: row 0 from X_row0; rows 1..n_tok-1 deterministic-random per extra_seed.
    {
        std::vector<float> X_buf((size_t)N_EMBD * n_tok);
        std::mt19937_64 rng(extra_seed ^ 0xfeed1234ULL);
        for (int t = 0; t < n_tok; ++t) {
            for (int d = 0; d < N_EMBD; ++d) {
                const size_t idx = (size_t)d + (size_t)t * N_EMBD;
                if (t == 0) {
                    X_buf[idx] = X_row0[d];
                } else {
                    const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
                    X_buf[idx] = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 0.5f;
                }
            }
        }
        ggml_backend_tensor_set(x, X_buf.data(), 0, X_buf.size() * sizeof(float));
    }

    const auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph_compute failed (status=%d) at n_tok=%d\n", (int)status, n_tok);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    // Extract row 0: ne[0]=N_EMBD, row 0 is the first N_EMBD floats.
    std::vector<float> dst_full((size_t)N_EMBD * n_tok);
    ggml_backend_tensor_get(out, dst_full.data(), 0, dst_full.size() * sizeof(float));

    row0_out.assign((size_t)N_EMBD, 0.0f);
    for (int d = 0; d < N_EMBD; ++d) {
        row0_out[d] = dst_full[d];
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

    // Row 0 fixed input.
    std::vector<float> X_row0(N_EMBD);
    {
        std::mt19937_64 rng(0xC0FFEEULL);
        for (auto & v : X_row0) {
            const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
            v = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 0.5f;
        }
    }

    const std::vector<int> n_toks = {1, 2, 4, 8};
    std::vector<std::vector<float>> outs;
    for (size_t i = 0; i < n_toks.size(); ++i) {
        std::vector<float> out;
        const uint64_t extra_seed = 0x55AA00ULL + (uint64_t)(i + 1) * 0x1000ULL;
        if (!run_one(backend, n_toks[i], X_row0, extra_seed, out)) {
            ggml_backend_free(backend);
            return 1;
        }
        outs.push_back(std::move(out));
    }

    ggml_backend_free(backend);

    fprintf(stderr, "row-0 output first 8 floats per n_tok:\n");
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
        fprintf(stderr, "OVERALL: PASS — row 0 byte-identical across n_tok ∈ {1, 2, 4, 8}\n");
        return 0;
    } else {
        fprintf(stderr, "OVERALL: FAIL — %d byte mismatches in RMSNorm row 0 across batch shape\n", total_diffs);
        return 1;
    }
}
