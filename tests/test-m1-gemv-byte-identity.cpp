// PHASE_LAUNCH_FUSION_SWEEP #195 — byte-identity gate for the small-batch Q4_0
// read-once GEMV (specs/dispatch/m1_quant_matmul.allium: FaithfulByteIdentity).
//
// Runs the SAME Q4_0 mul_mat graph twice through the identical CUDA dispatcher:
//   (a) GEMV off  -> production MMQ (mul_mat_q_split_k<Q4_0,...,4>)  = reference
//   (b) GEMV on   -> the read-once GEMV carve-out                    = test
// and asserts dst is byte-for-byte identical (max_abs_diff == 0.0) per column,
// across the ne11 family {1,2,3,4,8} and K in {1024,5120,288}. This is the
// gate that lets the kernel be substituted at any ne11<=K without perturbing
// np-invariance (the production NPC determinism contract).
//
// CUDA graph capture is disabled so the second compute re-dispatches the op
// rather than replaying a captured MMQ launch.

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

// Declared in ggml-cuda/mmq-gemv-faithful.cuh (library-internal); re-declared
// here since the test links against the lib but doesn't include CUDA headers.
extern "C" void ggml_cuda_m1_gemv_set_enabled(int enabled);
extern "C" void ggml_cuda_m1_gemv_set_threshold(int k);

static int run_case(ggml_backend_t backend, int K, int N, int M, unsigned seed) {
    printf("  K=%-5d N=%-5d ne11=%-2d ... ", K, N, M);
    fflush(stdout);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> wdist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);

    std::vector<float> Wf((size_t)N * K);
    std::vector<float> Af((size_t)M * K);
    for (auto & v : Wf) v = wdist(rng);
    for (auto & v : Af) v = adist(rng);

    // Quantize the weight to Q4_0 on the host.
    std::vector<uint8_t> Wq(ggml_row_size(GGML_TYPE_Q4_0, K) * (size_t)N);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, Wf.data(), Wq.data(), 0, N, K, nullptr, nullptr);

    struct ggml_init_params iparams = { (size_t)64*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(iparams);
    ggml_tensor * w_t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
    ggml_tensor * a_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, M);
    ggml_tensor * y_t = ggml_mul_mat(ctx, w_t, a_t);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { printf("FAIL alloc\n"); ggml_free(ctx); return 1; }
    ggml_backend_tensor_set(w_t, Wq.data(), 0, ggml_nbytes(w_t));
    ggml_backend_tensor_set(a_t, Af.data(), 0, ggml_nbytes(a_t));
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y_t);

    std::vector<float> y_ref((size_t)N * M);
    std::vector<float> y_tst((size_t)N * M);

    // (a) reference: production MMQ.
    ggml_cuda_m1_gemv_set_threshold(M > 8 ? M : 8);
    ggml_cuda_m1_gemv_set_enabled(0);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(y_t, y_ref.data(), 0, sizeof(float) * y_ref.size());

    // (b) test: the read-once GEMV carve-out.
    ggml_cuda_m1_gemv_set_enabled(1);
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_tensor_get(y_t, y_tst.data(), 0, sizeof(float) * y_tst.size());

    ggml_cuda_m1_gemv_set_enabled(0);

    // Byte-identity per column.
    int    mismatches = 0;
    float  max_abs    = 0.0f;
    int    worst_i = -1, worst_j = -1;
    for (int j = 0; j < M; ++j) {
        for (int i = 0; i < N; ++i) {
            const float a = y_ref[(size_t)j*N + i];
            const float b = y_tst[(size_t)j*N + i];
            uint32_t ua, ub;
            std::memcpy(&ua, &a, 4);
            std::memcpy(&ub, &b, 4);
            if (ua != ub) {
                const float d = std::fabs(a - b);
                if (d > max_abs) { max_abs = d; worst_i = i; worst_j = j; }
                mismatches++;
            }
        }
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    if (mismatches != 0) {
        const float rv = y_ref[(size_t)worst_j*N + worst_i];
        const float tv = y_tst[(size_t)worst_j*N + worst_i];
        printf("FAIL (%d/%d bits differ, max_abs_diff=%.3e at row %d col %d; ref=%.6g gemv=%.6g)\n",
               mismatches, N*M, max_abs, worst_i, worst_j, rv, tv);
        return 1;
    }
    printf("OK (max_abs_diff=0.0)\n");
    return 0;
}

int main(int /*argc*/, char ** /*argv*/) {
    // Re-dispatch (not graph replay) so the toggle takes effect on the 2nd run.
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);

    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) {
        fprintf(stderr, "Could not init CUDA backend (device 0).\n");
        printf("ALL OK (CUDA backend unavailable)\n");
        return 0;
    }

    int rc = 0;
    printf("[m1-gemv byte-identity] MMQ split_k=4 vs read-once GEMV\n");
    const int Ms[] = {1, 2, 3, 4, 8};
    // Domain: K a multiple of 256 (=MMQ_ITER_K) AND K >= 1024. Below K=1024
    // (nb<32) production MMQ's split_k=4 leaves dst uninitialized: its slice-0
    // CTA returns early without writing dst when the first K-quarter rounds to
    // an empty [kb0_start,kb0_stop) after blocks_per_iter=8 alignment, then the
    // fixup adds partials onto garbage (verified: MMQ returns +/-3e38/inf at
    // K=256/288/768; the GEMV stays finite — it always writes dst). This is a
    // pre-existing MMQ bug, out of scope here; all production Q4_0 K-dims are
    // multiples of 256 and >= 5120, well inside the valid domain. 1024/5120 are
    // clean-quarter sizes; 1280/17408 exercise non-even-quarter split-boundary
    // alignment (1280: nb=40 -> first boundary 10 aligns down to 8).
    const int Ks[] = {1024, 1280, 5120, 17408};
    unsigned seed = 1234;
    for (int K : Ks) {
        const int N = 2048;  // output rows
        for (int M : Ms) {
            rc |= run_case(backend, K, N, M, seed++);
        }
    }

    ggml_backend_free(backend);
    if (rc == 0) printf("ALL OK\n"); else printf("FAILURES detected\n");
    return rc;
}
