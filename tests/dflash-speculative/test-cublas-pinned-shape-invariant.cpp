// test-cublas-pinned-shape-invariant
//
// Phase C closure binding (PHASE_MMQ_Q4_0_AR16.md §6 / §11): under the
// production env stack LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1, cuBLAS
// matmuls (F16 / BF16 / F32 weights) must produce row-0 byte-identical
// outputs across batch sizes. Currently cuBLAS picks gemm algorithms
// heuristically per (M, N, K, dtype) → different M can pick different
// algo → different fp accumulator order → bit-different outputs.
//
// This driver exercises three weight dtypes — F16, BF16, F32 — at
// M ∈ {1, 4, 8, 16, 32} and bit-compares dst column 0. PASS when all
// three dtypes are stable.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Shape: token_embd / lm_head class. K and N moderate enough to keep
// memory footprint small while still exercising cuBLAS heuristics.
constexpr int64_t N_ROWS = 64;
constexpr int64_t K_COLS = 512;

static std::vector<float> run_one_M(ggml_backend_t backend,
                                    ggml_type wtype,
                                    const void * W_bytes, size_t W_nbytes,
                                    const std::vector<float> & a0,
                                    int64_t M, uint64_t seed_for_other_rows) {
    static const size_t mem_size = 64 * 1024 * 1024;
    struct ggml_init_params params = { mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * w   = ggml_new_tensor_2d(ctx, wtype,          K_COLS, N_ROWS);
    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K_COLS, M);
    ggml_tensor * dst = ggml_mul_mat(ctx, w, src);
    ggml_set_name(dst, "dst");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    ggml_backend_tensor_set(w, W_bytes, 0, W_nbytes);

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
        fprintf(stderr, "  ggml_backend_graph_compute failed (M=%lld, %s)\n",
                (long long)M, ggml_type_name(wtype));
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

static bool sweep_one_type(ggml_backend_t backend, ggml_type wtype,
                           const std::vector<float> & W_f32,
                           const std::vector<float> & a0) {
    const char * type_name = ggml_type_name(wtype);
    fprintf(stderr, "---- %s weight ----\n", type_name);

    // Convert F32 weight to wtype on CPU.
    std::vector<uint8_t> W_bytes;
    size_t W_n_elems = (size_t)N_ROWS * K_COLS;
    if (wtype == GGML_TYPE_F32) {
        W_bytes.assign((const uint8_t *)W_f32.data(),
                       (const uint8_t *)W_f32.data() + W_n_elems*sizeof(float));
    } else if (wtype == GGML_TYPE_F16) {
        W_bytes.resize(W_n_elems * sizeof(ggml_fp16_t));
        ggml_fp32_to_fp16_row(W_f32.data(), (ggml_fp16_t *)W_bytes.data(), W_n_elems);
    } else if (wtype == GGML_TYPE_BF16) {
        W_bytes.resize(W_n_elems * sizeof(ggml_bf16_t));
        ggml_fp32_to_bf16_row(W_f32.data(), (ggml_bf16_t *)W_bytes.data(), W_n_elems);
    } else {
        fprintf(stderr, "  unsupported wtype\n");
        return false;
    }

    const std::vector<int64_t> Ms = {1, 4, 8, 16, 32};
    std::vector<std::vector<float>> outs;
    for (size_t mi = 0; mi < Ms.size(); ++mi) {
        const uint64_t seed = 0x55AA0000ULL + (uint64_t)wtype*0x1000ULL + mi*0x100ULL;
        std::vector<float> out = run_one_M(backend, wtype, W_bytes.data(), W_bytes.size(),
                                           a0, Ms[mi], seed);
        if (out.empty()) return false;
        outs.push_back(std::move(out));
    }

    fprintf(stderr, "  row-0 first 8:\n");
    for (size_t mi = 0; mi < Ms.size(); ++mi) {
        fprintf(stderr, "    M=%2lld:", (long long)Ms[mi]);
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
                    fprintf(stderr, "    M=%2lld differs at row %2lld: ref=%+10.6f (0x%08x)  vs %+10.6f (0x%08x)  delta=%+.3e\n",
                            (long long)Ms[mi], (long long)i,
                            ref[i], a, cur[i], b, (double)(cur[i] - ref[i]));
                }
                ++diffs_here;
            }
        }
        if (diffs_here > 0) {
            fprintf(stderr, "    M=%2lld: %d/%lld values differ from M=1 baseline\n",
                    (long long)Ms[mi], diffs_here, (long long)N_ROWS);
            total_diffs += diffs_here;
        } else {
            fprintf(stderr, "    M=%2lld: byte-identical to M=1 baseline\n", (long long)Ms[mi]);
        }
    }
    fprintf(stderr, "  %s: %s\n\n", type_name, total_diffs == 0 ? "PASS" : "FAIL");
    return total_diffs == 0;
}

int main() {
    // Phase C is the shape-invariant cuBLAS pinning gated on the env var.
    setenv("LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH", "1", 1);

    // Populate ggml_table_f32_f16 (F16C is OFF in this build).
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

    // ---------- Random F32 weight (reused across types) ----------
    std::vector<float> W_f32(N_ROWS * K_COLS);
    {
        std::mt19937 rng(0xC0FFEEABULL);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        for (auto & v : W_f32) v = dist(rng);
    }

    // ---------- Fixed row-0 activation ----------
    std::vector<float> a0(K_COLS);
    {
        std::mt19937 rng(0xFEEDDEADULL);
        std::normal_distribution<float> dist(0.0f, 0.3f);
        for (auto & v : a0) v = dist(rng);
    }

    bool all_pass = true;
    all_pass &= sweep_one_type(backend, GGML_TYPE_F16,  W_f32, a0);
    all_pass &= sweep_one_type(backend, GGML_TYPE_BF16, W_f32, a0);
    all_pass &= sweep_one_type(backend, GGML_TYPE_F32,  W_f32, a0);

    ggml_backend_free(backend);

    fprintf(stderr, "OVERALL: %s\n", all_pass ? "PASS — Phase C closure binding"
                                              : "FAIL — cuBLAS algo / math mode still shape-dependent");
    return all_pass ? 0 : 1;
}
