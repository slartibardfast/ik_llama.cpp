// test-fattn-per-slot-kv-ncols-invariance.cpp
//
// Binds (or DISPROVES) the spec's claim at fattn-per-slot-kv-sm75.md §15.13
// that the production FA route, wmma_f16_case_pb1<256, 256, 8, float>, has
// inter-row fp16 rounding inside the WMMA frag_c_VKQ that breaks byte-
// identity at ne[1] > 1.
//
// The companion test test-fattn-per-slot-kv-dispatch-np-invariance.cpp
// only varies the K-cache stride at fixed ne[1]=1; it doesn't probe the
// ne[1]>1 case at all. This test does.
//
// Strategy:
//   1. Build the production-shape Q/K/V/mask/bound at ne[1]_MAX = 8.
//   2. Run with `n_tok` ∈ {1, 2, 4, 8}. In every run, query row 0's Q and
//      mask row 0 are identical; query rows 1..n_tok-1 contain different
//      random values per `n_tok` (so any inter-row interaction in the
//      kernel would surface as bit-divergence of row 0's output).
//   3. Bit-compare dst row 0 across all `n_tok` values.
//
// Expected outcome (per spec §15.13): FAIL — the wmma kernel's fp16 VKQ
// fragment accumulates with implicit inter-row dependence. This test
// becomes the binding unit test for whichever fix lands (fp32 VKQ
// fragment, alternative kernel path, etc.).
//
// Returns: 0 = byte-identical PASS, 1 = divergence FAIL, 77 = SKIP.

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

static constexpr int Dq        = 256;
static constexpr int Dv        = 256;
static constexpr int N_HEADS_Q = 24;
static constexpr int N_KV_HEADS = 4;
static constexpr int N_SEQS    = 1;
static constexpr int N_KV      = 256;   // FATTN_KQ_STRIDE-aligned (one block)
static constexpr int N_TOK_MAX = 8;     // wmma_f16-pb1 cols_per_block=8 ceiling
static constexpr int MASK_PAD  = 16;

static uint16_t f32_to_f16(float v) {
    uint32_t x;
    std::memcpy(&x, &v, sizeof(x));
    const uint32_t sign  = (x >> 16) & 0x8000u;
    int32_t        e     = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t       m     = x & 0x7fffffu;
    if (e <= 0) {
        if (e < -10) return (uint16_t)sign;
        m |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - e);
        return (uint16_t)(sign | (m >> shift));
    } else if (e >= 0x1f) {
        return (uint16_t)(sign | 0x7c00u | (m ? 1u : 0u));
    }
    return (uint16_t)(sign | ((uint32_t)e << 10) | (m >> 13));
}

static const uint16_t F16_ZERO = 0x0000u;

// Run one invocation at chosen n_tok (n_tok ∈ {1, 2, 4, 8}).
// Row 0 inputs are identical across runs; rows 1..n_tok-1 use `extra_seed`.
// Returns: dst slice for row 0 (Dv * N_HEADS_Q floats).
static bool run_one(ggml_backend_t backend,
                    int n_tok,
                    const std::vector<float>    & Q_row0,    // [Dq * N_HEADS_Q] (slot-0's Q for token 0)
                    const std::vector<uint16_t> & K_buf,     // [Dq * N_KV * N_KV_HEADS] - fp16, same all runs
                    const std::vector<uint16_t> & V_buf,     // [Dv * N_KV * N_KV_HEADS] - fp16, same all runs
                    const std::vector<uint16_t> & M_row0,    // [N_KV] - row 0 mask values
                    uint64_t extra_seed,
                    std::vector<float> & row0_out) {

    const int n_kv_padded_mask = ((N_KV + MASK_PAD - 1) / MASK_PAD) * MASK_PAD;
    const int n_tok_padded     = ((n_tok + MASK_PAD - 1) / MASK_PAD) * MASK_PAD;

    static const size_t mem_size = 8 * 1024 * 1024;
    ggml_init_params params = { mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Dq, n_tok, N_HEADS_Q, N_SEQS);
    ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, Dq, N_KV,   N_KV_HEADS, N_SEQS);
    ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, Dv, N_KV,   N_KV_HEADS, N_SEQS);
    ggml_tensor * mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_padded_mask, n_tok_padded);
    ggml_tensor * bound = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tok);

    const float scale    = 1.0f / std::sqrt((float)Dq);
    ggml_tensor * fa = ggml_flash_attn_ext_per_slot_kv(ctx, q, k, v, mask, bound,
                                                      scale, /*max_bias=*/0.0f, /*softcap=*/0.0f);
    ggml_set_name(fa, "fa");
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, fa);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }

    // Q: row 0 fixed from Q_row0; rows 1..n_tok-1 random per extra_seed.
    {
        const size_t n_Q_per_tok = (size_t)Dq * N_HEADS_Q * N_SEQS;  // Q is [Dq, n_tok, N_HEADS_Q, N_SEQS]
        // Layout: data[d + tok*Dq + head*Dq*n_tok + seq*Dq*n_tok*N_HEADS_Q]
        std::vector<float> Q_buf((size_t)Dq * n_tok * N_HEADS_Q * N_SEQS);
        std::mt19937_64 rng(extra_seed ^ 0xfeed1234ULL);
        for (int s = 0; s < N_SEQS; ++s) {
            for (int h = 0; h < N_HEADS_Q; ++h) {
                for (int t = 0; t < n_tok; ++t) {
                    for (int d = 0; d < Dq; ++d) {
                        const size_t idx = (size_t)d + (size_t)t * Dq
                                         + (size_t)h * Dq * n_tok
                                         + (size_t)s * Dq * n_tok * N_HEADS_Q;
                        if (t == 0) {
                            const size_t r0_idx = (size_t)d + (size_t)h * Dq;
                            Q_buf[idx] = Q_row0[r0_idx];
                        } else {
                            const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
                            Q_buf[idx] = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 0.5f;
                        }
                    }
                }
            }
        }
        ggml_backend_tensor_set(q, Q_buf.data(), 0, Q_buf.size() * sizeof(float));
    }

    ggml_backend_tensor_set(k, K_buf.data(), 0, K_buf.size() * sizeof(uint16_t));
    ggml_backend_tensor_set(v, V_buf.data(), 0, V_buf.size() * sizeof(uint16_t));

    // Mask: [n_kv_padded_mask, n_tok_padded] row-major on ne[0]=n_kv_padded_mask.
    // Row 0 = M_row0; rows 1..n_tok-1 random fp16 per extra_seed.
    {
        std::vector<uint16_t> M_buf((size_t)n_kv_padded_mask * n_tok_padded, F16_ZERO);
        // Row 0 from M_row0.
        for (int k_pos = 0; k_pos < N_KV; ++k_pos) {
            M_buf[k_pos] = M_row0[k_pos];
        }
        // Other rows: deterministic-per-seed random fp16 to provoke any inter-row leak.
        std::mt19937_64 rng(extra_seed ^ 0xC0DEBEEFULL);
        for (int t = 1; t < n_tok; ++t) {
            for (int k_pos = 0; k_pos < N_KV; ++k_pos) {
                const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
                const float fv = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 0.1f;
                M_buf[(size_t)t * n_kv_padded_mask + k_pos] = f32_to_f16(fv);
            }
        }
        // Padding columns past N_KV stay 0 in all rows (no -inf needed since K cells exist).
        ggml_backend_tensor_set(mask, M_buf.data(), 0, M_buf.size() * sizeof(uint16_t));
    }

    // per-row K bound: every row sees positions [0, N_KV).
    {
        std::vector<int32_t> bound_buf(n_tok, N_KV);
        ggml_backend_tensor_set(bound, bound_buf.data(), 0, bound_buf.size() * sizeof(int32_t));
    }

    const auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph_compute failed (status=%d) at n_tok=%d\n", (int)status, n_tok);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    // Extract row 0's output. fa shape is [Dv, N_HEADS_Q, n_tok, N_SEQS] F32
    // per ggml_flash_attn_ext_per_slot_kv (ne = {v->ne[0], q->ne[2], q->ne[1], q->ne[3]}).
    // Memory: idx = d + h*Dv + t*Dv*N_HEADS_Q + s*Dv*N_HEADS_Q*n_tok. Row 0 (t=0) per head: Dv * N_HEADS_Q values.
    std::vector<float> dst_full((size_t)Dv * N_HEADS_Q * n_tok * N_SEQS);
    ggml_backend_tensor_get(fa, dst_full.data(), 0, dst_full.size() * sizeof(float));

    row0_out.assign((size_t)Dv * N_HEADS_Q * N_SEQS, 0.0f);
    for (int s = 0; s < N_SEQS; ++s) {
        for (int h = 0; h < N_HEADS_Q; ++h) {
            for (int d = 0; d < Dv; ++d) {
                const size_t full_idx = (size_t)d + (size_t)h * Dv
                                      + 0 * (size_t)Dv * N_HEADS_Q
                                      + (size_t)s * Dv * N_HEADS_Q * n_tok;
                const size_t row0_idx = (size_t)d + (size_t)h * Dv + (size_t)s * Dv * N_HEADS_Q;
                row0_out[row0_idx] = dst_full[full_idx];
            }
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main() {
    setenv("LLAMA_FATTN_PER_SLOT_KV_ENABLE", "1", 1);
    setenv("LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH", "1", 1);

    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed; SKIP\n");
        return 77;
    }

    // Row 0 Q.
    std::vector<float> Q_row0((size_t)Dq * N_HEADS_Q);
    {
        std::mt19937_64 rng(0xC0FFEEULL);
        for (auto & v : Q_row0) {
            const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
            v = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 0.5f;
        }
    }

    // K, V buffers fixed across all runs.
    std::vector<uint16_t> K_buf((size_t)Dq * N_KV * N_KV_HEADS * N_SEQS);
    std::vector<uint16_t> V_buf((size_t)Dv * N_KV * N_KV_HEADS * N_SEQS);
    {
        std::mt19937_64 rng(0xC0FFEE2ULL);
        for (auto & u : K_buf) {
            const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
            const float f = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 0.5f;
            u = f32_to_f16(f);
        }
        for (auto & u : V_buf) {
            const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
            const float f = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 0.5f;
            u = f32_to_f16(f);
        }
    }

    // Mask row 0 — all zeros (all K positions valid).
    std::vector<uint16_t> M_row0(N_KV, F16_ZERO);

    const std::vector<int> n_toks = {1, 2, 4, 8};
    std::vector<std::vector<float>> outs;
    for (size_t i = 0; i < n_toks.size(); ++i) {
        std::vector<float> out;
        const uint64_t extra_seed = 0x55AA00ULL + (uint64_t)(i + 1) * 0x1000ULL;
        if (!run_one(backend, n_toks[i], Q_row0, K_buf, V_buf, M_row0, extra_seed, out)) {
            ggml_backend_free(backend);
            return 1;
        }
        outs.push_back(std::move(out));
    }

    ggml_backend_free(backend);

    // Bit-compare row 0 across all n_tok values.
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
        fprintf(stderr, "OVERALL: FAIL — %d byte mismatches (wmma fp16 frag_c_VKQ inter-row rounding suspected)\n", total_diffs);
        return 1;
    }
}
