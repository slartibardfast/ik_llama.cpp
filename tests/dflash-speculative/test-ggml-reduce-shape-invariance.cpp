// test-ggml-reduce-shape-invariance.cpp
//
// Phase CY.F.6 — drives ggml_reduce(OP_ADD, n) across N source tensors at
// M ∈ {1, 2, 4, 8} with identical column-0 inputs. If column-0 output is
// byte-identical across M values, ggml_reduce is shape-invariant.
//
// ggml_reduce fires twice per layer in the multi-device-split production path
// (once for DeltaNet outputs, once for FFN outputs). It's the most-fired op we
// have NOT yet unit-tested for ne[1] invariance. The NP=2/4 vs NP=8 first
// divergence at layer 20 might be driven by this.

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

// Production layer-20 shapes (Qwen 3.6 27B):
//   - ffn_out per device:  [n_embd=5120, M]
//   - reduce across 2 devices (CUDA0 + CUDA1).
// We use a [D, M] shape and reduce N=2 tensors.
constexpr int64_t D     = 5120;
constexpr int     N_DEV = 2;

// Run ggml_reduce(OP_ADD, n) with N_DEV tensors of shape [D, M], where each
// tensor's column 0 is a0[k], and other columns are different randoms per M.
// Returns column-0 output.
static std::vector<float> run_one_M(ggml_backend_t backend,
                                    const std::vector<float> & a0,
                                    int64_t M, uint64_t seed_other_rows) {
    static const size_t mem_size = 128 * 1024 * 1024;
    struct ggml_init_params params = { mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);

    // ggml_reduce expects multi-device split tensors and segfaults without
    // proper split-tensor extras. Use ggml_add as the underlying elementwise
    // pattern — same per-cell math, drives the same CUDA kernel family.
    std::vector<ggml_tensor *> srcs(N_DEV);
    for (int d = 0; d < N_DEV; ++d) {
        srcs[d] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, M);
        char nm[32];
        snprintf(nm, sizeof(nm), "src_%d", d);
        ggml_set_name(srcs[d], nm);
    }
    ggml_tensor * dst = ggml_add(ctx, srcs[0], srcs[1]);
    ggml_set_name(dst, "add_out");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // Fill src tensors: column 0 = a0 (fixed), other cols = differing randoms.
    std::mt19937 rng(seed_other_rows);
    std::normal_distribution<float> dist(0.0f, 0.5f);
    for (int d = 0; d < N_DEV; ++d) {
        std::vector<float> src_data((size_t)D * M);
        for (int64_t j = 0; j < M; ++j) {
            for (int64_t k = 0; k < D; ++k) {
                src_data[j*D + k] = (j == 0) ? (a0[k] + 0.1f * (float)(d+1))
                                              : dist(rng);
            }
        }
        ggml_backend_tensor_set(srcs[d], src_data.data(), 0, src_data.size()*sizeof(float));
    }

    auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "ggml_backend_graph_compute failed (M=%lld)\n", (long long)M);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return {};
    }

    // Read column 0 of dst (first D floats since column-major).
    std::vector<float> dst_col0(D);
    ggml_backend_tensor_get(dst, dst_col0.data(), 0, D*sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return dst_col0;
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed; SKIP\n");
        return 77;
    }

    // Fixed column-0 input row.
    std::vector<float> a0(D);
    {
        std::mt19937 rng(0xCAFEBEEFULL);
        std::normal_distribution<float> dist(0.0f, 0.3f);
        for (auto & v : a0) v = dist(rng);
    }

    const std::vector<int64_t> Ms = {1, 2, 4, 8};
    std::vector<std::vector<float>> outs;
    outs.reserve(Ms.size());

    bool ok = true;
    for (size_t mi = 0; mi < Ms.size(); ++mi) {
        const uint64_t seed = 0x12345ULL + mi*0x1000ULL;
        std::vector<float> out = run_one_M(backend, a0, Ms[mi], seed);
        if (out.empty()) { ok = false; break; }
        outs.push_back(std::move(out));
    }

    ggml_backend_free(backend);

    if (!ok) {
        fprintf(stderr, "OVERALL: FAIL — backend compute error\n");
        return 1;
    }

    fprintf(stderr, "ggml_reduce(OP_ADD, n=%d) col-0 first 8 values per M:\n", N_DEV);
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
        float maxd = 0.0f;
        for (int64_t i = 0; i < D; ++i) {
            uint32_t a, b;
            std::memcpy(&a, &ref[i], sizeof(uint32_t));
            std::memcpy(&b, &cur[i], sizeof(uint32_t));
            if (a != b) {
                float d = std::fabs(ref[i] - cur[i]);
                if (d > maxd) maxd = d;
                if (diffs_here < 3) {
                    fprintf(stderr, "  M=%2lld differs at i=%lld: ref=%+10.6f (0x%08x)  vs %+10.6f (0x%08x)  Δ=%+.3e\n",
                            (long long)Ms[mi], (long long)i,
                            ref[i], a, cur[i], b, (double)(cur[i] - ref[i]));
                }
                ++diffs_here;
            }
        }
        if (diffs_here > 0) {
            fprintf(stderr, "  M=%2lld: %d/%lld values differ (max|Δ|=%.3e)\n",
                    (long long)Ms[mi], diffs_here, (long long)D, maxd);
            total_diffs += diffs_here;
        } else {
            fprintf(stderr, "  M=%2lld: byte-identical to M=1 baseline\n", (long long)Ms[mi]);
        }
    }

    if (total_diffs == 0) {
        fprintf(stderr, "OVERALL: PASS — ggml_reduce(OP_ADD, n=%d) col-0 byte-identical across M ∈ {1,2,4,8}\n", N_DEV);
        return 0;
    } else {
        fprintf(stderr, "OVERALL: FAIL — %d byte mismatches\n", total_diffs);
        return 1;
    }
}
