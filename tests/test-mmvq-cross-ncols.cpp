// PHASE_LAUNCH_FUSION_SWEEP path B diagnostic — is MMVQ byte-identical ACROSS
// ncols? np-invariance under the force-MMVQ decode carve-out requires
// mmvq(col 0, ncols=1) == mmvq(col 0, ncols=K) byte-for-byte for K in [1,8].
// The full-server NPC gate FAILED at np=8 (np<=4 passed); this isolates the
// matmul from server timing to pin exactly which ncols diverge and by how much.
//
// For each ncols in {1,2,3,4,8}: build a Q4_0 mul_mat with a FIXED column-0
// activation (other columns random), force MMVQ (ggml_cuda_decode_mmvq_set_enabled),
// and compare dst column 0 against the ncols=1 result.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

extern "C" void ggml_cuda_decode_mmvq_set_enabled(int enabled);

static std::vector<float> run_col0(ggml_backend_t backend, const std::vector<uint8_t>& Wq,
                                   int K, int N, int ncols, const std::vector<float>& col0, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> adist(-1.0f, 1.0f);
    std::vector<float> Af((size_t)ncols * K);
    for (int k = 0; k < K; ++k) Af[k] = col0[k];                 // column 0 fixed
    for (int j = 1; j < ncols; ++j)
        for (int k = 0; k < K; ++k) Af[(size_t)j*K + k] = adist(rng);

    struct ggml_init_params ip = { (size_t)64*1024*1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, K, N);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  K, ncols);
    ggml_tensor * y = ggml_mul_mat(ctx, w, a);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(w, Wq.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(a, Af.data(), 0, ggml_nbytes(a));
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> col0_out(N);
    ggml_backend_tensor_get(y, col0_out.data(), 0, sizeof(float)*N);  // dst col 0 = first N floats
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return col0_out;
}

int main() {
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) { printf("ALL OK (CUDA unavailable)\n"); return 0; }
    ggml_cuda_decode_mmvq_set_enabled(1);   // force MMVQ for ne11<=8

    const int K = 5120, N = 2048;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> wdist(-0.5f, 0.5f), adist(-1.0f, 1.0f);
    std::vector<float> Wf((size_t)N*K);
    for (auto& v : Wf) v = wdist(rng);
    std::vector<uint8_t> Wq(ggml_row_size(GGML_TYPE_Q4_0, K)*(size_t)N);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, Wf.data(), Wq.data(), 0, N, K, nullptr, nullptr);
    std::vector<float> col0(K);
    for (auto& v : col0) v = adist(rng);

    printf("[mmvq cross-ncols] K=%d N=%d — dst col0 vs ncols=1 (MMVQ forced)\n", K, N);
    std::vector<float> ref = run_col0(backend, Wq, K, N, 1, col0, 1000);
    int rc = 0;
    for (int ncols : {1, 2, 3, 4, 8}) {
        std::vector<float> got = run_col0(backend, Wq, K, N, ncols, col0, 1000 + ncols);
        int diff = 0; float maxabs = 0; int wi = -1;
        for (int i = 0; i < N; ++i) {
            uint32_t a, b; std::memcpy(&a,&ref[i],4); std::memcpy(&b,&got[i],4);
            if (a != b) { float d = std::fabs(ref[i]-got[i]); if (d>maxabs){maxabs=d;wi=i;} diff++; }
        }
        printf("  ncols=%-2d : %s (%d/%d differ, max_abs=%.3e%s)\n", ncols,
               diff==0?"BYTE-IDENTICAL":"DIFFERS", diff, N, maxabs,
               wi>=0?"":"");
        if (ncols>1 && diff!=0) rc = 1;
    }
    ggml_backend_free(backend);
    printf(rc==0 ? "ALL OK\n" : "MMVQ NOT byte-identical across ncols\n");
    return rc;
}
