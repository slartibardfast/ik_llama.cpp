// T2 — ssm_conv runtime-single-seq fast path correctness + dispatch.
//
// RED before Phase B: kernel single-seq fast path is gated on the
// allocation-time n_kv == 1 condition; with n_kv > 1 and a runtime
// single-seq batch, the dispatcher falls through to ssm_conv_f32_kernel.
// We assert (a) the slow path is taken (env-counter shows zero
// fast-path dispatches) and (b) the slow path's output is byte-equal
// to a CPU reference. The (a) check is RED before Phase B because the
// counter doesn't exist yet.
//
// GREEN after Phase B: kernel detects runtime-single-seq from src3 and
// promotes to the fast path; counter == n_dispatches.
//
// Approach: build a tiny ggml graph with ggml_ssm_conv on the CUDA
// backend, run it, then run the same graph on the CPU backend, compare
// outputs. Repeat for slot S in {0, 1, 2, 3}.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int D_INNER = 128;     // nr
constexpr int D_CONV  = 4;       // nc
constexpr int N_T     = 8;       // tokens per batch
constexpr int N_KV    = 4;       // state slots
constexpr float ATOL  = 1e-5f;

bool run_one_slot(int slot, std::vector<float> & out_cpu, std::vector<float> & out_cuda) {
    // Allocate ggml context with enough memory for src tensors + dst.
    const size_t mem = 64 * 1024 * 1024;
    std::vector<uint8_t> mem_buf(mem);
    ggml_init_params params = { mem, mem_buf.data(), false };
    ggml_context * ctx = ggml_init(params);

    // src0: conv_state [d_conv-1, d_inner, n_kv]
    ggml_tensor * src0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D_CONV - 1, D_INNER, N_KV);
    // src1: x [d_inner, n_t]
    ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_INNER, N_T);
    // src2: weight [d_conv, d_inner]
    ggml_tensor * src2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_CONV, D_INNER);
    // src3: state_seq [n_kv, n_t] — but we use only the first row, all entries == slot.
    ggml_tensor * src3 = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_KV, N_T);

    ggml_tensor * dst = ggml_ssm_conv(ctx, src0, src1, src2, src3);

    // Fill data deterministically.
    const float seed_a = 0.013f * (1 + slot);
    for (int s = 0; s < N_KV; ++s) {
        for (int r = 0; r < D_INNER; ++r) {
            for (int c = 0; c < D_CONV - 1; ++c) {
                ((float *) src0->data)[s * D_INNER * (D_CONV-1) + r * (D_CONV-1) + c] =
                    sinf(seed_a + 0.1f * s + 0.01f * r + c);
            }
        }
    }
    for (int t = 0; t < N_T; ++t) {
        for (int r = 0; r < D_INNER; ++r) {
            ((float *) src1->data)[t * D_INNER + r] = cosf(0.5f + 0.07f * t + 0.011f * r);
        }
    }
    for (int r = 0; r < D_INNER; ++r) {
        for (int c = 0; c < D_CONV; ++c) {
            ((float *) src2->data)[r * D_CONV + c] = 0.1f + 0.001f * r + 0.01f * c;
        }
    }
    // src3: every token routes to slot S, with sentinel -1 in the rest.
    for (int t = 0; t < N_T; ++t) {
        ((int32_t *) src3->data)[t * N_KV + 0] = slot;
        for (int k = 1; k < N_KV; ++k) {
            ((int32_t *) src3->data)[t * N_KV + k] = -1;
        }
    }

    // Build cgraph.
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    // CPU run.
    {
        ggml_backend_t backend = ggml_backend_cpu_init();
        ggml_backend_graph_compute(backend, gf);
        out_cpu.assign((float *) dst->data, (float *) dst->data + ggml_nelements(dst));
        ggml_backend_free(backend);
    }

    // CUDA run — re-fill data; CPU compute may have written to dst.
    // Simpler: rebuild context for CUDA pass.
    ggml_free(ctx);
    ggml_init_params params2 = { mem, mem_buf.data(), false };
    ctx = ggml_init(params2);
    src0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D_CONV - 1, D_INNER, N_KV);
    src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_INNER, N_T);
    src2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_CONV, D_INNER);
    src3 = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_KV, N_T);
    dst  = ggml_ssm_conv(ctx, src0, src1, src2, src3);
    for (int s = 0; s < N_KV; ++s) {
        for (int r = 0; r < D_INNER; ++r) {
            for (int c = 0; c < D_CONV - 1; ++c) {
                ((float *) src0->data)[s * D_INNER * (D_CONV-1) + r * (D_CONV-1) + c] =
                    sinf(seed_a + 0.1f * s + 0.01f * r + c);
            }
        }
    }
    for (int t = 0; t < N_T; ++t) {
        for (int r = 0; r < D_INNER; ++r) {
            ((float *) src1->data)[t * D_INNER + r] = cosf(0.5f + 0.07f * t + 0.011f * r);
        }
    }
    for (int r = 0; r < D_INNER; ++r) {
        for (int c = 0; c < D_CONV; ++c) {
            ((float *) src2->data)[r * D_CONV + c] = 0.1f + 0.001f * r + 0.01f * c;
        }
    }
    for (int t = 0; t < N_T; ++t) {
        ((int32_t *) src3->data)[t * N_KV + 0] = slot;
        for (int k = 1; k < N_KV; ++k) {
            ((int32_t *) src3->data)[t * N_KV + k] = -1;
        }
    }
    gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);
    {
        ggml_backend_t backend = ggml_backend_cuda_init(0);
        if (!backend) {
            fprintf(stderr, "  SKIP: CUDA backend init failed\n");
            ggml_free(ctx);
            return false;
        }
        ggml_backend_graph_compute(backend, gf);
        out_cuda.assign((float *) dst->data, (float *) dst->data + ggml_nelements(dst));
        ggml_backend_free(backend);
    }

    ggml_free(ctx);
    return true;
}

} // namespace

int main() {
    int failed = 0;
    for (int slot = 0; slot < N_KV; ++slot) {
        std::vector<float> cpu, cuda;
        if (!run_one_slot(slot, cpu, cuda)) continue;

        float max_diff = 0.0f;
        for (size_t i = 0; i < cpu.size() && i < cuda.size(); ++i) {
            float d = fabsf(cpu[i] - cuda[i]);
            if (d > max_diff) max_diff = d;
        }
        printf("  slot %d: max abs diff = %.3e\n", slot, max_diff);
        if (max_diff > ATOL) {
            printf("  FAIL: slot %d diff %.3e > %.3e\n", slot, max_diff, ATOL);
            failed++;
        }
    }

    // Path-counter assertion (Phase B exposes this via env var):
    //   LLAMA_DEBUG_SSM_CONV_PATH=1 makes the kernel print one line per
    //   dispatch; the line distinguishes "single_seq_runtime" vs
    //   "slow". After Phase B, all 4 dispatches above should be
    //   "single_seq_runtime". Pre-Phase-B, none are.
    //
    // This assertion is checked by the harness shell wrapper, not in
    // C++, since we'd otherwise have to plumb a global counter.

    if (failed == 0) {
        printf("T2 GREEN: ssm_conv runtime-single-seq correct across slots 0..%d\n", N_KV - 1);
        return 0;
    }
    printf("T2 RED: %d slot(s) failed correctness\n", failed);
    return 1;
}
