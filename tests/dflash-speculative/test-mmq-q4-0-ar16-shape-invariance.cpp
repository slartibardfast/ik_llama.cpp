// test-mmq-q4-0-ar16-shape-invariance
//
// Phase A.10 closure binding (PHASE_MMQ_Q4_0_AR16.md §11): with the
// LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH env var set, computing Q4_0_AR16 ×
// F32 activation at M ∈ {1, 4, 8, 16, 32} must produce a byte-identical
// row-0 output regardless of M. This is what "shape-invariant dispatch"
// promises — and what NP-cross byte-identity at decode time depends on.
//
// Strategy:
//   1. Allocate a Q4_0_AR16 weight tensor [K, N] once.
//   2. Generate a fixed F32 activation row a0 [K].
//   3. For each M in the sweep: build src1 [K, M] with src1[:, 0] = a0
//      and src1[:, 1..M-1] = different randoms per M.
//   4. Run ggml_mul_mat (weights × src1 → dst) under env var.
//   5. Read back dst[:, 0] (= W @ a0 only) and bit-compare across M's.

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
// N = output dim (rows of W). K = input dim (cols of W = rows of A).
// Must be multiples of QK_AR16=16 for the AR16 layout to apply cleanly.
constexpr int64_t N_ROWS = 64;   // output rows
constexpr int64_t K_COLS = 512;  // K dimension (must be % 16 == 0, and ideally % 256 == 0 for full kb0 iters)

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

    // Weight: copy bytes in (Q4_0_AR16-quantized).
    ggml_backend_tensor_set(w, W_q_bytes.data(), 0, W_q_bytes.size());

    // src1: row 0 = a0 (fixed across all M). Other rows = differing randoms.
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

    // dst is [N_ROWS, M] in ggml's nb convention (ne[0]=N_ROWS, ne[1]=M).
    // dst row 0 (column index j=0) covers N_ROWS values.
    std::vector<float> dst_col0(N_ROWS);
    ggml_backend_tensor_get(dst, dst_col0.data(), 0, N_ROWS*sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return dst_col0;
}

int main() {
    setenv("LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH", "1", 1);

    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed; SKIP\n");
        return 77;
    }

    // ---------- Generate F32 weights and quantize to Q4_0_AR16 (once) ----------
    std::vector<float> W_f32(N_ROWS * K_COLS);
    {
        std::mt19937 rng(0xBEEFFACEULL);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        for (auto & v : W_f32) v = dist(rng);
    }

    const int64_t blocks_per_row = K_COLS / QK_AR16;
    const size_t  row_bytes      = blocks_per_row * sizeof(block_q4_0_ar16);
    std::vector<uint8_t> W_q_bytes((size_t)N_ROWS * row_bytes);
    for (int64_t i = 0; i < N_ROWS; ++i) {
        block_q4_0_ar16 * row = (block_q4_0_ar16 *)(W_q_bytes.data() + (size_t)i * row_bytes);
        quantize_row_q4_0_ar16_cpu(&W_f32[i*K_COLS], row, K_COLS);
    }

    // ---------- Fixed row-0 activation ----------
    std::vector<float> a0(K_COLS);
    {
        std::mt19937 rng(0xDEADBEEFULL);
        std::normal_distribution<float> dist(0.0f, 0.3f);
        for (auto & v : a0) v = dist(rng);
    }

    // ---------- Sweep M values, compare row-0 output ----------
    const std::vector<int64_t> Ms = {1, 4, 8, 16, 32};
    std::vector<std::vector<float>> outs;
    outs.reserve(Ms.size());

    bool ok = true;
    for (size_t mi = 0; mi < Ms.size(); ++mi) {
        // Use a DIFFERENT seed for "other rows" per M, so we'd see divergence
        // if the kernel were summing across rows incorrectly.
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

    // ---------- Bit-compare row 0 across all M ----------
    fprintf(stderr, "row-0 first 8 values (column j=0) per M:\n");
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
            // Bit-compare via memcmp.
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

    if (total_diffs == 0) {
        fprintf(stderr, "OVERALL: PASS — shape-invariant row-0 output across M ∈ {1,4,8,16,32}\n");
        return 0;
    } else {
        fprintf(stderr, "OVERALL: FAIL — %d byte mismatches\n", total_diffs);
        return 1;
    }
}
