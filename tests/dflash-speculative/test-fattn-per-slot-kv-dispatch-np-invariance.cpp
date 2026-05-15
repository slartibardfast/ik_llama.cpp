// test-fattn-per-slot-kv-dispatch-np-invariance.cpp
//
// Binds the NP-invariance contract of the production dispatch path for
// GGML_OP_FLASH_ATTN_EXT_PER_SLOT_KV at the ggml-op level.
//
// Per specs/deltanet/fattn-per-slot-kv-sm75.md §15.7, the dispatcher
// routes the new op to ggml_cuda_flash_attn_ext_wmma_f16_case_pb1
// <256, 256, 8, half> (parallel_blocks=1, cols_per_block=8, KQ_acc_t=half
// all fixed at compile time). This is the determinism primitive: the
// kernel's K-loop partitioning and template instantiation no longer
// depend on Q->ne[1] * ne[2] * ne[3].
//
// The test drives the op through the CUDA backend twice with K-cache
// occupancies that differ by an integer multiple of FATTN_KQ_STRIDE.
// Slot-0's actual content (Q row 0, K/V[0..N1), mask row [0..N1)) is
// kept byte-identical between runs. The "extra" K/V cells in run B at
// positions [N1..N2) are filled with random fp16 data, and the mask
// at those positions is set to -INFINITY (simulating cells that belong
// to other slots / unallocated entries that the per-row mask correctly
// excludes from slot 0's softmax).
//
// Binding: slot-0's output bytes must be identical across run A and B.
// PASS = the dispatch path is NP-invariant.
//
// Returns: 0 = PASS, 1 = FAIL, 77 = SKIP (no CUDA device).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Production target shape. The wmma_f16 dispatch path requires
// K->ne[1] % FATTN_KQ_STRIDE == 0 (FATTN_KQ_STRIDE = 256).
static constexpr int Dq        = 256;
static constexpr int Dv        = 256;
static constexpr int N_HEADS_Q = 24;
static constexpr int N_KV_HEADS = 4;
static constexpr int N_TOK     = 1;   // decode shape (cols_per_block=8 covers ne[1]<=8)
static constexpr int N_SEQS    = 1;
static constexpr int KQ_STRIDE = 256;
static constexpr int MASK_PAD  = 16;  // GGML_KQ_MASK_PAD

// fp32 -> fp16 (IEEE half) bit-cast helper. We avoid pulling in CUDA's
// __half here to keep this a pure host file.
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

static const uint16_t F16_NEG_INF = 0xFC00u;

struct Run {
    std::vector<float> out;   // [Dv * n_heads_q * n_tok * n_seqs] = 256 * 24 * 1 * 1
};

// Run the ggml_flash_attn_ext_per_slot_kv op once at the chosen n_kv_max.
// Slot-0 content (Q, K/V positions [0, N1), mask row [0, N1)) is filled
// from the supplied byte buffers — identical across runs. Positions
// [N1, n_kv_max) are filled with random fp16 (different per run) and
// masked to -INF.
static bool run_one(ggml_backend_t backend,
                    int n_kv_max,
                    int N1,
                    const std::vector<float>    & Q_f,            // [Dq * n_heads_q * n_tok * n_seqs] - F32 per FA convention
                    const std::vector<uint16_t> & K_slot0,        // [Dq * N1 * n_kv_heads * n_seqs]
                    const std::vector<uint16_t> & V_slot0,        // [Dv * N1 * n_kv_heads * n_seqs]
                    const std::vector<uint16_t> & M_slot0_valid,  // [N1] - mask values for valid range (typically all 0)
                    uint64_t extra_seed,
                    Run & out) {
    const int n_kv_max_padded_mask = ((n_kv_max + MASK_PAD - 1) / MASK_PAD) * MASK_PAD;
    const int n_tok_padded         = ((N_TOK     + MASK_PAD - 1) / MASK_PAD) * MASK_PAD;

    static const size_t mem_size = 8 * 1024 * 1024;
    ggml_init_params params = { mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "ggml_init failed\n");
        return false;
    }

    ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, Dq, N_TOK, N_HEADS_Q, N_SEQS);
    ggml_tensor * k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, Dq, n_kv_max, N_KV_HEADS, N_SEQS);
    ggml_tensor * v    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, Dv, n_kv_max, N_KV_HEADS, N_SEQS);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_max_padded_mask, n_tok_padded);
    ggml_tensor * bound = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_TOK);

    const float scale    = 1.0f / std::sqrt((float)Dq);
    const float max_bias = 0.0f;
    const float softcap  = 0.0f;
    ggml_tensor * fa = ggml_flash_attn_ext_per_slot_kv(ctx, q, k, v, mask, bound,
                                                      scale, max_bias, softcap);
    ggml_set_name(fa, "fa");

    // Force the kernel-precision op_param to GGML_PREC_F32 to bypass the
    // KQ_acc_t=float case; we want the cols_per_block=8, KQ_acc_t=half
    // path which is what production hits. The default (op_params[3] = 0
    // = GGML_PREC_DEFAULT) selects that branch.
    // op_param[3] is already GGML_PREC_DEFAULT from builder; leave it.

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, fa);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        fprintf(stderr, "alloc_ctx_tensors failed at n_kv_max=%d\n", n_kv_max);
        ggml_free(ctx);
        return false;
    }

    // Fill Q (same across runs). Q is F32 per ggml FA convention.
    ggml_backend_tensor_set(q, Q_f.data(), 0, Q_f.size() * sizeof(float));

    // Fill K. Slot-0's first N1 positions are byte-identical between runs.
    // For positions [N1, n_kv_max) we write deterministic-per-seed random
    // fp16 bytes that vary between runs (these must be masked out for
    // slot-0 not to see them in the softmax).
    {
        const size_t n_K = (size_t)Dq * n_kv_max * N_KV_HEADS * N_SEQS;
        std::vector<uint16_t> K_buf(n_K, 0);
        // Slot-0 valid range (n_kv_heads × Dq × N1 elements at positions [0, N1)).
        // K is laid out [Dq, n_kv_max, n_kv_heads, n_seqs] contiguous row-major
        // on ne[0]=Dq. K[d, k, h, s] -> index s*n_kv_heads*n_kv_max*Dq + h*n_kv_max*Dq + k*Dq + d.
        for (int h = 0; h < N_KV_HEADS; ++h) {
            for (int k_pos = 0; k_pos < N1; ++k_pos) {
                for (int d = 0; d < Dq; ++d) {
                    const size_t dst_idx = (size_t)h * n_kv_max * Dq + (size_t)k_pos * Dq + d;
                    const size_t src_idx = (size_t)h * N1 * Dq       + (size_t)k_pos * Dq + d;
                    K_buf[dst_idx] = K_slot0[src_idx];
                }
            }
        }
        // Fill [N1, n_kv_max) with random fp16 bits.
        std::mt19937_64 rng(extra_seed ^ 0xa11c0deULL);
        for (int h = 0; h < N_KV_HEADS; ++h) {
            for (int k_pos = N1; k_pos < n_kv_max; ++k_pos) {
                for (int d = 0; d < Dq; ++d) {
                    const size_t dst_idx = (size_t)h * n_kv_max * Dq + (size_t)k_pos * Dq + d;
                    // Keep magnitudes modest (avoid NaN propagation); sample from N(0,1)
                    // expressed as fp16.
                    const uint32_t r  = (uint32_t)(rng() & 0xffffffffULL);
                    const float    fv = ((int32_t)(r & 0xffff) - 32768) / 32768.0f;
                    K_buf[dst_idx] = f32_to_f16(fv);
                }
            }
        }
        ggml_backend_tensor_set(k, K_buf.data(), 0, K_buf.size() * sizeof(uint16_t));
    }

    {
        const size_t n_V = (size_t)Dv * n_kv_max * N_KV_HEADS * N_SEQS;
        std::vector<uint16_t> V_buf(n_V, 0);
        for (int h = 0; h < N_KV_HEADS; ++h) {
            for (int k_pos = 0; k_pos < N1; ++k_pos) {
                for (int d = 0; d < Dv; ++d) {
                    const size_t dst_idx = (size_t)h * n_kv_max * Dv + (size_t)k_pos * Dv + d;
                    const size_t src_idx = (size_t)h * N1 * Dv       + (size_t)k_pos * Dv + d;
                    V_buf[dst_idx] = V_slot0[src_idx];
                }
            }
        }
        std::mt19937_64 rng(extra_seed ^ 0xbeef0001ULL);
        for (int h = 0; h < N_KV_HEADS; ++h) {
            for (int k_pos = N1; k_pos < n_kv_max; ++k_pos) {
                for (int d = 0; d < Dv; ++d) {
                    const size_t dst_idx = (size_t)h * n_kv_max * Dv + (size_t)k_pos * Dv + d;
                    const uint32_t r  = (uint32_t)(rng() & 0xffffffffULL);
                    const float    fv = ((int32_t)(r & 0xffff) - 32768) / 32768.0f;
                    V_buf[dst_idx] = f32_to_f16(fv);
                }
            }
        }
        ggml_backend_tensor_set(v, V_buf.data(), 0, V_buf.size() * sizeof(uint16_t));
    }

    // Mask: row 0 (only row, since N_TOK=1). [0, N1) = valid (0.0); [N1, n_kv_max) = -inf.
    // mask layout is [n_kv_max_padded, n_tok_padded] row-major on ne[0]=n_kv_max_padded.
    {
        std::vector<uint16_t> M_buf((size_t)n_kv_max_padded_mask * n_tok_padded, F16_NEG_INF);
        for (int k_pos = 0; k_pos < N1; ++k_pos) {
            M_buf[k_pos] = M_slot0_valid[k_pos];
        }
        // M_buf[N1 .. n_kv_max_padded_mask) and across all other rows stays -inf.
        ggml_backend_tensor_set(mask, M_buf.data(), 0, M_buf.size() * sizeof(uint16_t));
    }

    // per-row K bound: row 0 sees positions [0, N1). Set bound[0] = N1.
    {
        std::vector<int32_t> bound_buf(N_TOK, N1);
        ggml_backend_tensor_set(bound, bound_buf.data(), 0, bound_buf.size() * sizeof(int32_t));
    }

    const auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph_compute failed (status=%d) at n_kv_max=%d\n", (int)status, n_kv_max);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    out.out.assign((size_t)Dv * N_HEADS_Q * N_TOK * N_SEQS, 0.0f);
    ggml_backend_tensor_get(fa, out.out.data(), 0, out.out.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed; skipping (no sm_75 CUDA device)\n");
        return 77;
    }

    // Slot-0's actual content. N1 must satisfy N1 + extra > N1 (test would be trivial otherwise).
    constexpr int N1 = KQ_STRIDE;       // 256 (one FATTN_KQ_STRIDE block)
    constexpr int N2 = 2 * KQ_STRIDE;   // 512 (two blocks; simulates ~2x cache from NP>1)

    std::mt19937_64 rng(0xC0FFEEULL);
    auto sample_f16 = [&](size_t n) {
        std::vector<uint16_t> out(n);
        for (size_t i = 0; i < n; ++i) {
            const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
            const float    f = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 0.5f;
            out[i] = f32_to_f16(f);
        }
        return out;
    };
    auto sample_f32 = [&](size_t n) {
        std::vector<float> out(n);
        for (size_t i = 0; i < n; ++i) {
            const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
            out[i] = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 0.5f;
        }
        return out;
    };

    auto Q_f        = sample_f32((size_t)Dq * N_HEADS_Q * N_TOK * N_SEQS);
    auto K_slot0    = sample_f16((size_t)Dq * N1 * N_KV_HEADS * N_SEQS);
    auto V_slot0    = sample_f16((size_t)Dv * N1 * N_KV_HEADS * N_SEQS);
    std::vector<uint16_t> M_slot0_valid(N1, f32_to_f16(0.0f));

    Run run_A, run_B;
    if (!run_one(backend, N1, N1, Q_f, K_slot0, V_slot0, M_slot0_valid, 0x11111111ULL, run_A)) {
        ggml_backend_free(backend);
        return 1;
    }
    if (!run_one(backend, N2, N1, Q_f, K_slot0, V_slot0, M_slot0_valid, 0x99999999ULL, run_B)) {
        ggml_backend_free(backend);
        return 1;
    }

    ggml_backend_free(backend);

    // Byte-compare. fp32 outputs; expect bitwise equality.
    if (run_A.out.size() != run_B.out.size()) {
        fprintf(stderr, "FAIL: output size mismatch (%zu vs %zu)\n",
                run_A.out.size(), run_B.out.size());
        return 1;
    }

    size_t n_diff = 0;
    float  max_abs_delta = 0.0f;
    for (size_t i = 0; i < run_A.out.size(); ++i) {
        uint32_t ba, bb;
        std::memcpy(&ba, &run_A.out[i], sizeof(ba));
        std::memcpy(&bb, &run_B.out[i], sizeof(bb));
        if (ba != bb) {
            ++n_diff;
            const float d = std::abs(run_A.out[i] - run_B.out[i]);
            if (d > max_abs_delta) max_abs_delta = d;
        }
    }

    if (n_diff != 0) {
        fprintf(stderr, "FAIL: %zu / %zu output floats differ bytewise across n_kv_max ∈ {%d, %d}; max |Δ| = %.3e\n",
                n_diff, run_A.out.size(), N1, N2, max_abs_delta);
        return 1;
    }

    printf("PASS: %zu output floats byte-identical across n_kv_max ∈ {%d, %d}\n",
           run_A.out.size(), N1, N2);
    return 0;
}
