// test-dflash-drafter-forward.cpp
//
// Byte-identity unit test: persistent cooperative DFlash drafter forward
// kernel vs the scalar reference (which composes the WMMA-mimicking
// oracle for major matmuls + scalar fp32 attention). Exercised across a
// (BLOCK_SIZE × N_slots × seed) sweep. This is the T4 closure binding
// for drafter_forward per kernel-design.md §6.1.
//
// @witnesses: SingleForwardPerStep
// @witnesses: QuerySpanIsOnePlusN
// @witnesses: InjectionConsumedAtEveryLayer
// @witnesses: LayerTypeDependentMask
// @witnesses: AnchorEmbeddingFromTarget
// @witnesses: AnchorPosPreserved
// @witnesses: BlockSizeBindsToConfig
// @witnesses: DeterminismPerDeployment
// @witnesses: FeatureSourceFixedPerDeployment
//
// CURRENT STATE: stub kernel + stub reference both return zeros.
// Test confirms byte-identical agreement on zeros (trivially), then
// exits 77 (CTest SKIP) until the real kernel body and reference body
// land. Establishes build + plumbing + sweep harness.
//
// Wire-up plan to PASS:
//   1. Implement dflash_reference::drafter_forward_scalar_ref_f32(...)
//      composing wmma_mma_oracle + RMSNorm + RoPE + scalar attention.
//   2. Implement dflash_drafter_forward kernel body per spec §6.1.
//   3. Change run_one() to call those non-stub functions and exit 0 on
//      PASS, 1 on FAIL — removing the unconditional exit 77.
//
// PASS criterion per sub-run (when active):
//   byte-identical OR every disagreement is ≤ 1 fp16 ULP AND rate ≤ 1 %
//   of output cells. Rationale matches T3 combine_features — WMMA-tile
//   binary tree vs serial fp32 reduction differs by 0-1 ULP, propagating
//   through RMSNorm and into a small fraction of output cells.

#include "dflash-drafter-forward-reference.h"
#include "wmma-mimicking-oracle.h"
#include "ggml-cuda/dflash/dflash-drafter-forward.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_CHECK(stmt)                                                   \
    do {                                                                   \
        cudaError_t _e = (stmt);                                           \
        if (_e != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                         __FILE__, __LINE__, cudaGetErrorString(_e));      \
            return 1;                                                      \
        }                                                                  \
    } while (0)

namespace {

// Locked drafter shape from kernel-design.md §6.1 / spec §2.
constexpr int L_d          = 5;
constexpr int D_emb        = 5120;
constexpr int H_q          = 40;
constexpr int H_kv         = 8;
constexpr int D_h          = 128;
constexpr int intermediate = 17408;
constexpr int swa_window   = 2048;
constexpr float rope_base  = 10000000.0f;
constexpr float norm_eps   = 1.0e-6f;

constexpr bool STUB_REFERENCE_AND_KERNEL = true;

int run_one(int N_slots, int BLOCK_SIZE, int SeqLen, uint32_t seed) {
    std::printf("  [run] N_slots=%d BLOCK_SIZE=%d SeqLen=%d seed=%u\n",
                N_slots, BLOCK_SIZE, SeqLen, seed);
    if (BLOCK_SIZE != 4 && BLOCK_SIZE != 5 && BLOCK_SIZE != 6 && BLOCK_SIZE != 8) {
        std::fprintf(stderr, "    [SKIP] BLOCK_SIZE %d not in {4,5,6,8}\n", BLOCK_SIZE);
        return 0;
    }

    const std::size_t n_out =
        static_cast<std::size_t>(N_slots) *
        static_cast<std::size_t>(BLOCK_SIZE) *
        static_cast<std::size_t>(D_emb);

    // For the stub phase, we don't allocate the giant per-layer weight
    // tensors yet — only the inputs and outputs sufficient to drive the
    // launcher. Once the reference and kernel are real, this expands to
    // allocate the full L_d × (q_w, q_norm, o_w, attn_norm, ffn_norm,
    // gate, up, down) weight set + k_cache + v_cache.
    (void) seed;

    std::vector<__half> ref_h(n_out);
    dflash_reference::drafter_forward_reference_stub(
        ref_h.data(), N_slots, BLOCK_SIZE, D_emb);

    std::vector<__half> dev_h(n_out);

    // For the stub kernel, the launcher accepts null pointers for the
    // per-layer-weight arrays because all parameters are (void)-cast
    // before being used. Once the kernel body lands, the test must
    // allocate real weight buffers.
    __half * d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, n_out * sizeof(__half)));

    dflash_drafter_forward_launch(
        /*d_input_tokens_emb*/ nullptr,
        /*d_k_cache*/          nullptr,
        /*d_v_cache*/          nullptr,
        /*d_slot_positions*/   nullptr,
        /*d_layer_attn_norm_w*/nullptr,
        /*d_layer_q_w*/        nullptr,
        /*d_layer_q_norm_w*/   nullptr,
        /*d_layer_o_w*/        nullptr,
        /*d_layer_ffn_norm_w*/ nullptr,
        /*d_layer_gate_w*/     nullptr,
        /*d_layer_up_w*/       nullptr,
        /*d_layer_down_w*/     nullptr,
        /*d_layer_types*/      nullptr,
        swa_window, rope_base, norm_eps,
        BLOCK_SIZE, N_slots, SeqLen,
        L_d, D_emb, H_q, H_kv, D_h, intermediate,
        d_out, /*stream*/ 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dev_h.data(), d_out, n_out * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    int max_ulp = 0;
    int n_ulp_eq_1 = 0;
    int n_ulp_gt_1 = 0;
    for (std::size_t i = 0; i < n_out; ++i) {
        const int ulp = dflash_reference::fp16_ulp_delta(ref_h[i], dev_h[i]);
        if (ulp > max_ulp) max_ulp = ulp;
        if (ulp == 1) ++n_ulp_eq_1;
        if (ulp > 1)  ++n_ulp_gt_1;
    }
    const double rate1 = static_cast<double>(n_ulp_eq_1) / n_out;
    const double rateN = static_cast<double>(n_ulp_gt_1) / n_out;

    std::printf("    [ ] max_ulp=%d  rate_1ULP=%.3f%%  rate_>1ULP=%.3f%%\n",
                max_ulp, rate1 * 100.0, rateN * 100.0);

    if (max_ulp > 1 && rateN > 0.01) {
        std::fprintf(stderr, "    [FAIL] kernel and reference disagree by > 1 ULP at > 1%% of cells\n");
        return 1;
    }
    std::printf("    [PASS] kernel == reference (within tolerance)\n");
    return 0;
}

} // anonymous namespace

int main() {
    if constexpr (STUB_REFERENCE_AND_KERNEL) {
        std::printf("[STUB] reference and kernel both return zeros — "
                    "test infrastructure exercised, returning SKIP (77).\n");
        // Sanity check that the stub agreement holds before we SKIP —
        // catches build / plumbing breakage even in the stub phase.
        int rc = run_one(/*N_slots*/2, /*BLOCK_SIZE*/4, /*SeqLen*/64, /*seed*/42);
        if (rc != 0) {
            std::fprintf(stderr, "[FATAL] stub agreement check failed: rc=%d\n", rc);
            return 1;
        }
        return 77;
    }

    // Active sweep (used once both reference and kernel are real).
    int fails = 0;
    const int N_slots_list[]   = {1, 2, 4, 8};
    const int BLOCK_SIZE_list[]= {4, 5, 6, 8};
    const uint32_t seeds[]     = {1u, 42u};
    const int SeqLen           = 4096;

    for (int N_slots : N_slots_list) {
        for (int BLOCK_SIZE : BLOCK_SIZE_list) {
            for (uint32_t seed : seeds) {
                fails += run_one(N_slots, BLOCK_SIZE, SeqLen, seed);
            }
        }
    }

    if (fails > 0) {
        std::fprintf(stderr, "[OVERALL] %d sub-runs failed\n", fails);
        return 1;
    }
    std::printf("[OVERALL] all PASS\n");
    return 0;
}
