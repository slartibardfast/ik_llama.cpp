// CUDA graph cache: state-mutation across alternating-shape Update sequence.
//
// Drives a single topology (ggml_add of two F32 tensors) through five
// shapes A→B→A→C→A in sequence, capturing each output. Compares each
// to an eager-path reference computed via ggml_backend_cpu_compute on
// the same inputs. Asserts elementwise exact match across all five.
//
// Today (pre-B): the shape-keyed cache assigns each shape its own
// entry; outputs are correct trivially.
// Post-B: the topology-keyed cache shares a single entry across all
// five submissions, with cudaGraphExecUpdate patching ne between them.
// If Update leaks state from a prior shape's pointers/extents, this
// test catches it.
//
// Build: gated on GGML_CUDA via tests/CMakeLists.txt.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

struct shape_run {
    int64_t n;        // ne[1]
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> out_cuda;
    std::vector<float> out_cpu;
};

static bool run_shape_on_backend(ggml_backend_t backend, shape_run & r) {
    static const size_t mem_size = 4 * 1024 * 1024;
    struct ggml_init_params params = { mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, r.n);
    ggml_tensor * b   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, r.n);
    ggml_tensor * out = ggml_add(ctx, a, b);
    ggml_set_name(out, "out");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }

    ggml_backend_tensor_set(a, r.a.data(), 0, r.a.size() * sizeof(float));
    ggml_backend_tensor_set(b, r.b.data(), 0, r.b.size() * sizeof(float));

    const auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    r.out_cuda.assign(r.a.size(), 0.0f);
    ggml_backend_tensor_get(out, r.out_cuda.data(), 0, r.out_cuda.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static bool run_shape_eager_reference(shape_run & r) {
    r.out_cpu.resize(r.a.size());
    for (size_t i = 0; i < r.a.size(); ++i) {
        r.out_cpu[i] = r.a[i] + r.b[i];
    }
    return true;
}

int main() {
    setenv("GGML_CUDA_GRAPH_MAX", "256", 1);

    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed; skipping\n");
        return 0;
    }

    // Five distinct shapes (n=8, 16, 8, 24, 8) — alternating returns
    // to the same shape stress the post-B Update→re-Update sequence.
    std::vector<shape_run> seq;
    for (int64_t n : {8, 16, 8, 24, 8}) {
        shape_run r;
        r.n = n;
        r.a.resize((size_t) 32 * n);
        r.b.resize((size_t) 32 * n);
        for (size_t i = 0; i < r.a.size(); ++i) {
            r.a[i] = 1.0f + 0.001f * (float) (i % 13);
            r.b[i] = -0.5f + 0.002f * (float) ((i + 7) % 17);
        }
        seq.push_back(std::move(r));
    }

    // Run every shape on the CUDA backend in sequence (this exercises
    // the cache state across the alternating sequence) and compute the
    // eager reference independently.
    for (auto & r : seq) {
        if (!run_shape_on_backend(backend, r)) {
            fprintf(stderr, "cuda compute failed at n=%lld\n", (long long) r.n);
            ggml_backend_free(backend);
            return 1;
        }
        run_shape_eager_reference(r);
    }
    ggml_backend_free(backend);

    printf("=== test-cuda-graph-update-state-mutation ===\n");
    printf("  drove sequence n=8,16,8,24,8 (5 runs across 1 topology, 3 distinct shapes)\n");

    // Element-wise compare each cuda output to the eager reference.
    int n_mismatch = 0;
    for (size_t step = 0; step < seq.size(); ++step) {
        const auto & r = seq[step];
        size_t mism_this_step = 0;
        for (size_t i = 0; i < r.out_cuda.size(); ++i) {
            if (r.out_cuda[i] != r.out_cpu[i]) {
                if (mism_this_step < 4) {
                    fprintf(stderr,
                            "  step=%zu n=%lld idx=%zu cuda=%.6f cpu=%.6f\n",
                            step, (long long) r.n, i, r.out_cuda[i], r.out_cpu[i]);
                }
                mism_this_step++;
            }
        }
        if (mism_this_step) {
            n_mismatch++;
            printf("  step=%zu n=%lld: %zu mismatched elements\n", step, (long long) r.n, mism_this_step);
        }
    }
    if (n_mismatch) {
        printf("RESULT: FAIL — Update path corrupted output across alternating-shape sequence\n");
        return 1;
    }
    printf("RESULT: PASS — all 5 outputs match eager reference\n");
    return 0;
}
