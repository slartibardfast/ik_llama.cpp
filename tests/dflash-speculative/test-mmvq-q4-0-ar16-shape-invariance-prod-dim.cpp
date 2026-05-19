// test-mmvq-q4-0-ar16-shape-invariance
//
// Phase B (PHASE_MMQ_Q4_0_AR16.md §5): MMVQ kernel for Q4_0_AR16.
//
// Two closure bindings in one driver:
//
//   B.2 — correctness vs scalar fp32 reference at M=1 (MMVQ path active).
//         Required: cosine ≥ 0.9999, NMSE ≤ 1e-4.
//
//   B.4 — shape-invariance: row 0 of (W @ src1) byte-identical across
//         M ∈ {1, 2, 4, 8} when MMVQ is the chosen dispatch path
//         (LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH unset, batch ≤
//         MMVQ_MAX_BATCH_SIZE).
//
// Strategy:
//   1. Build random Q4_0_AR16 weight W [N, K] once, quantize CPU-side.
//   2. Generate fixed row-0 activation a0 [K] once.
//   3. For each M ∈ {1, 2, 4, 8}: construct src1 [K, M] with
//      src1[:, 0] = a0 and src1[:, j>0] random per-M (DIFFERENT seeds), run
//      ggml_mul_mat through the CUDA backend with the env UNSET so MMVQ
//      is the chosen path (batch ≤ MMVQ_MAX_BATCH_SIZE).
//   4. Bind:
//      a) row-0 of dst (= W @ a0) byte-identical across all M.
//      b) row-0 of dst at M=1 matches a CPU fp32 reference within cos +
//         NMSE thresholds.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// ----- Configuration -----
// N = output dim (rows of W). K = input dim. M sweep stays within
// MMVQ_MAX_BATCH_SIZE = 8 so MMVQ is chosen rather than MMQ.
constexpr int64_t N_ROWS = 5120;
constexpr int64_t K_COLS = 5120;  // production hidden dim

// Closure-binding thresholds for B.2.
constexpr float COSINE_MIN = 0.9999f;
constexpr float NMSE_MAX   = 1e-4f;

// ----- Q4_0_AR16 quantizer (CPU) -----
static void quantize_row_q4_0_ar16_cpu(const float * x, block_q4_0_ar16 * y, int k) {
    const int nb = k / QK_AR16;
    for (int b = 0; b < nb; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < QK_AR16; ++j) {
            const float v = std::fabs(x[b*QK_AR16 + j]);
            if (v > amax) amax = v;
        }
        const float d  = amax / -8.0f;
        const float id = d ? 1.0f/d : 0.0f;
        y[b].d = ggml_fp32_to_fp16(d);
        for (int j = 0; j < QK_AR16/2; ++j) {
            const float v0 = x[b*QK_AR16 + 2*j + 0] * id;
            const float v1 = x[b*QK_AR16 + 2*j + 1] * id;
            int q0 = (int)std::roundf(v0) + 8;
            int q1 = (int)std::roundf(v1) + 8;
            if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
            if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
            y[b].qs[j] = (uint8_t)((q1 << 4) | q0);
        }
    }
}

static void dequantize_row_q4_0_ar16_cpu(const block_q4_0_ar16 * x, float * y, int k) {
    const int nb = k / QK_AR16;
    for (int b = 0; b < nb; ++b) {
        const float d = ggml_fp16_to_fp32(x[b].d);
        for (int j = 0; j < QK_AR16/2; ++j) {
            const int q0 = (int)(x[b].qs[j] & 0x0F) - 8;
            const int q1 = (int)((x[b].qs[j] >> 4) & 0x0F) - 8;
            y[b*QK_AR16 + 2*j + 0] = d * (float)q0;
            y[b*QK_AR16 + 2*j + 1] = d * (float)q1;
        }
    }
}

static std::vector<float> run_one_M(ggml_backend_t backend, const std::vector<uint8_t> & W_q_bytes,
                                    const std::vector<float> & a0, int64_t M, uint64_t seed_for_other_rows) {
    static const size_t mem_size = 64 * 1024 * 1024;
    struct ggml_init_params params = { mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * w   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0_AR16, K_COLS, N_ROWS);
    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,       K_COLS, M);
    ggml_tensor * dst = ggml_mul_mat(ctx, w, src);
    ggml_set_name(dst, "dst");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    ggml_backend_tensor_set(w, W_q_bytes.data(), 0, W_q_bytes.size());

    std::vector<float> src_data((size_t)K_COLS * M);
    std::mt19937 rng(seed_for_other_rows);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    for (int64_t j = 0; j < M; ++j) {
        for (int64_t k = 0; k < K_COLS; ++k) {
            src_data[j*K_COLS + k] = (j == 0) ? a0[k] : dist(rng);
        }
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size()*sizeof(float));

    auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_backend_graph_compute failed (M=%lld)\n", (long long)M);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return {};
    }

    std::vector<float> dst_col0(N_ROWS);
    ggml_backend_tensor_get(dst, dst_col0.data(), 0, N_ROWS*sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return dst_col0;
}

int main() {
    // Belt-and-suspenders: explicitly unset the shape-invariant env so MMVQ is
    // chosen at small batch. (A typical CTest invocation won't have it set,
    // but if a developer exports it for local testing this guards us.)
    unsetenv("LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH");

    // Populate ggml_table_f32_f16 lookup table (used by ggml_fp16_to_fp32 when
    // F16C is OFF — this build has GGML_F16C=OFF). Without this prologue the
    // CPU dequantize step below returns all zeros.
    {
        struct ggml_init_params init_params = { 0, nullptr, true };
        ggml_context * init_ctx = ggml_init(init_params);
        if (init_ctx) ggml_free(init_ctx);
    }

    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed; SKIP\n");
        return 77;
    }

    // ---------- Generate F32 weights, quantize, dequantize for the reference ----------
    std::vector<float> W_f32(N_ROWS * K_COLS);
    {
        std::mt19937 rng(0xBEEFFACEULL);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        for (auto & v : W_f32) v = dist(rng);
    }

    const int64_t blocks_per_row = K_COLS / QK_AR16;
    const size_t  row_bytes      = blocks_per_row * sizeof(block_q4_0_ar16);
    std::vector<uint8_t> W_q_bytes((size_t)N_ROWS * row_bytes);
    std::vector<float>   W_dequant((size_t)N_ROWS * K_COLS);
    for (int64_t i = 0; i < N_ROWS; ++i) {
        block_q4_0_ar16 * row = (block_q4_0_ar16 *)(W_q_bytes.data() + (size_t)i * row_bytes);
        quantize_row_q4_0_ar16_cpu(&W_f32[i*K_COLS], row, K_COLS);
        dequantize_row_q4_0_ar16_cpu(row, &W_dequant[i*K_COLS], K_COLS);
    }

    // ---------- Fixed row-0 activation ----------
    std::vector<float> a0(K_COLS);
    {
        std::mt19937 rng(0xDEADBEEFULL);
        std::normal_distribution<float> dist(0.0f, 0.3f);
        for (auto & v : a0) v = dist(rng);
    }

    // ---------- CPU fp32 reference: dequant(W) @ a0 ----------
    std::vector<float> ref_dst(N_ROWS);
    for (int64_t i = 0; i < N_ROWS; ++i) {
        double acc = 0.0;
        for (int64_t k = 0; k < K_COLS; ++k) {
            acc += (double)W_dequant[i*K_COLS + k] * (double)a0[k];
        }
        ref_dst[i] = (float)acc;
    }

    // ---------- Sweep M values, compare row-0 output ----------
    const std::vector<int64_t> Ms = {1, 2, 4, 8};
    std::vector<std::vector<float>> outs;
    outs.reserve(Ms.size());

    bool ok = true;
    for (size_t mi = 0; mi < Ms.size(); ++mi) {
        const uint64_t seed = 0x1234ULL + mi*0x100ULL;
        std::vector<float> out = run_one_M(backend, W_q_bytes, a0, Ms[mi], seed);
        if (out.empty()) {
            ok = false;
            break;
        }
        outs.push_back(std::move(out));
    }

    ggml_backend_free(backend);

    if (!ok) {
        fprintf(stderr, "OVERALL: FAIL — backend compute error\n");
        return 1;
    }

    // ---------- B.2 — correctness binding at M=1 ----------
    fprintf(stderr, "B.2 correctness: GPU vs CPU fp32 reference at M=1\n");
    double dot_ab = 0.0, sqA = 0.0, sqB = 0.0, sq_err = 0.0, sq_ref = 0.0;
    for (int64_t i = 0; i < N_ROWS; ++i) {
        const double a = outs[0][i];
        const double b = ref_dst[i];
        dot_ab += a*b;
        sqA    += a*a;
        sqB    += b*b;
        sq_err += (a - b) * (a - b);
        sq_ref += b * b;
    }
    const double cosine = dot_ab / (std::sqrt(sqA * sqB) + 1e-30);
    const double nmse   = sq_err / (sq_ref + 1e-30);
    fprintf(stderr, "  cosine = %.6f (min %.4f)  NMSE = %.3e (max %.1e)\n",
            cosine, COSINE_MIN, nmse, NMSE_MAX);
    fprintf(stderr, "  GPU[0..7]: ");
    for (int k = 0; k < 8; ++k) fprintf(stderr, " %+10.6f", outs[0][k]);
    fprintf(stderr, "\n  REF[0..7]: ");
    for (int k = 0; k < 8; ++k) fprintf(stderr, " %+10.6f", ref_dst[k]);
    fprintf(stderr, "\n");

    bool b2_pass = (cosine >= COSINE_MIN) && (nmse <= NMSE_MAX);
    fprintf(stderr, "  B.2: %s\n\n", b2_pass ? "PASS" : "FAIL");

    // ---------- B.4 — shape-invariance binding ----------
    fprintf(stderr, "B.4 shape-invariance: row-0 across M ∈ {1, 2, 4, 8}\n");
    for (size_t mi = 0; mi < Ms.size(); ++mi) {
        fprintf(stderr, "  M=%2lld:", (long long)Ms[mi]);
        for (int k = 0; k < 8; ++k) fprintf(stderr, " %+10.6f", outs[mi][k]);
        fprintf(stderr, "\n");
    }

    int total_diffs = 0;
    const auto & ref = outs[0];
    for (size_t mi = 1; mi < Ms.size(); ++mi) {
        const auto & cur = outs[mi];
        int diffs_here = 0;
        for (int64_t i = 0; i < N_ROWS; ++i) {
            uint32_t a, b;
            std::memcpy(&a, &ref[i], sizeof(uint32_t));
            std::memcpy(&b, &cur[i], sizeof(uint32_t));
            if (a != b) {
                if (diffs_here < 3) {
                    fprintf(stderr, "  M=%2lld differs at row %2lld: ref=%+10.6f (0x%08x)  vs %+10.6f (0x%08x)  delta=%+.3e\n",
                            (long long)Ms[mi], (long long)i,
                            ref[i], a, cur[i], b, (double)(cur[i] - ref[i]));
                }
                ++diffs_here;
            }
        }
        if (diffs_here > 0) {
            fprintf(stderr, "  M=%2lld: %d/%lld values differ from M=1 baseline\n",
                    (long long)Ms[mi], diffs_here, (long long)N_ROWS);
            total_diffs += diffs_here;
        } else {
            fprintf(stderr, "  M=%2lld: byte-identical to M=1 baseline\n", (long long)Ms[mi]);
        }
    }
    bool b4_pass = (total_diffs == 0);
    fprintf(stderr, "  B.4: %s\n\n", b4_pass ? "PASS" : "FAIL");

    if (b2_pass && b4_pass) {
        fprintf(stderr, "OVERALL: PASS — Phase B closure binding\n");
        return 0;
    } else {
        fprintf(stderr, "OVERALL: FAIL — B.2:%s B.4:%s\n",
                b2_pass ? "ok" : "no",
                b4_pass ? "ok" : "no");
        return 1;
    }
}
