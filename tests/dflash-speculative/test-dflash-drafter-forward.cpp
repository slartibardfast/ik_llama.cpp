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
// CURRENT STATE: stub kernel returns zeros; reference is REAL (composes
// the WMMA oracle through 5 layers of {Q proj, q_norm, RoPE, attn, O
// proj, residual, ffn_norm, gate/up/silu/down, residual}).
//
// Run paths:
//   1. Reference smoke test (DEFAULT WHEN STUB_KERNEL=true) — runs the
//      reference at a tiny shape with random fp16 weights and checks
//      output is non-zero. Validates reference is at least callable.
//      Test then exits 77 (CTest SKIP) until the kernel body lands.
//   2. Kernel-vs-reference byte-identity sweep — runs once STUB_KERNEL
//      is flipped to false (kernel body in place). Compares kernel
//      output against reference output across a (BLOCK_SIZE × N_slots)
//      grid of sub-runs. Exit 0 PASS / 1 FAIL.
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

// Tiny-shape config for the reference smoke test. Real T4 shape is the
// production drafter (D_emb=5120, etc.) — but we don't allocate that
// scale of random fp16 weights inside a unit test; the kernel-vs-reference
// sweep uses the production shape and reads weights from the drafter GGUF
// once #60 plumbing lands.
struct TinyShape {
    int L_d         = 2;
    int N_slots     = 2;
    int BLOCK_SIZE  = 4;
    int SeqLen      = 32;
    int D_emb       = 64;
    int H_q         = 4;
    int H_kv        = 2;
    int D_h         = 16;
    int intermediate= 96;
    int swa_window  = 16;
    float rope_base = 10000.0f;
    float norm_eps  = 1.0e-6f;
};

// Allocate + fill random fp16 weights for one drafter layer.
struct LayerWeights {
    std::vector<__half> attn_norm;
    std::vector<__half> q_w;
    std::vector<__half> q_norm;
    std::vector<__half> o_w;
    std::vector<__half> ffn_norm;
    std::vector<__half> gate_w;
    std::vector<__half> up_w;
    std::vector<__half> down_w;
};

LayerWeights gen_layer_weights(const TinyShape & s, std::mt19937 & rng) {
    std::uniform_real_distribution<float> d_w(-0.1f, 0.1f);
    std::uniform_real_distribution<float> d_n(0.5f, 1.5f);
    LayerWeights L;
    L.attn_norm.resize(s.D_emb);
    for (auto & h : L.attn_norm) h = __float2half(d_n(rng));
    L.q_w.resize(static_cast<std::size_t>(s.H_q * s.D_h) * s.D_emb);
    for (auto & h : L.q_w) h = __float2half(d_w(rng));
    L.q_norm.resize(s.D_h);
    for (auto & h : L.q_norm) h = __float2half(d_n(rng));
    L.o_w.resize(static_cast<std::size_t>(s.D_emb) * s.H_q * s.D_h);
    for (auto & h : L.o_w) h = __float2half(d_w(rng));
    L.ffn_norm.resize(s.D_emb);
    for (auto & h : L.ffn_norm) h = __float2half(d_n(rng));
    L.gate_w.resize(static_cast<std::size_t>(s.intermediate) * s.D_emb);
    for (auto & h : L.gate_w) h = __float2half(d_w(rng));
    L.up_w.resize(static_cast<std::size_t>(s.intermediate) * s.D_emb);
    for (auto & h : L.up_w) h = __float2half(d_w(rng));
    L.down_w.resize(static_cast<std::size_t>(s.D_emb) * s.intermediate);
    for (auto & h : L.down_w) h = __float2half(d_w(rng));
    return L;
}

int reference_smoke_test() {
    TinyShape s;
    std::printf("[smoke] tiny shape: L_d=%d N_slots=%d BLOCK_SIZE=%d D_emb=%d H_q=%d D_h=%d intermediate=%d SeqLen=%d\n",
                s.L_d, s.N_slots, s.BLOCK_SIZE, s.D_emb, s.H_q, s.D_h, s.intermediate, s.SeqLen);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d_w(-0.2f, 0.2f);

    const int Q = 1 + s.BLOCK_SIZE;
    std::vector<__half> input_tokens_emb(
        static_cast<std::size_t>(s.N_slots) * Q * s.D_emb);
    for (auto & h : input_tokens_emb) h = __float2half(d_w(rng));

    std::vector<__half> k_cache(
        static_cast<std::size_t>(s.L_d) * s.N_slots * s.SeqLen * s.H_kv * s.D_h);
    std::vector<__half> v_cache(
        static_cast<std::size_t>(s.L_d) * s.N_slots * s.SeqLen * s.H_kv * s.D_h);
    for (auto & h : k_cache) h = __float2half(d_w(rng));
    for (auto & h : v_cache) h = __float2half(d_w(rng));

    std::vector<int> slot_positions(s.N_slots);
    for (int i = 0; i < s.N_slots; ++i) slot_positions[i] = 8 + i;

    std::vector<LayerWeights> layers;
    layers.reserve(s.L_d);
    for (int l = 0; l < s.L_d; ++l) layers.push_back(gen_layer_weights(s, rng));

    std::vector<const __half *> p_attn_norm(s.L_d);
    std::vector<const __half *> p_q_w(s.L_d);
    std::vector<const __half *> p_q_norm(s.L_d);
    std::vector<const __half *> p_o_w(s.L_d);
    std::vector<const __half *> p_ffn_norm(s.L_d);
    std::vector<const __half *> p_gate(s.L_d);
    std::vector<const __half *> p_up(s.L_d);
    std::vector<const __half *> p_down(s.L_d);
    for (int l = 0; l < s.L_d; ++l) {
        p_attn_norm[l] = layers[l].attn_norm.data();
        p_q_w[l]       = layers[l].q_w.data();
        p_q_norm[l]    = layers[l].q_norm.data();
        p_o_w[l]       = layers[l].o_w.data();
        p_ffn_norm[l]  = layers[l].ffn_norm.data();
        p_gate[l]      = layers[l].gate_w.data();
        p_up[l]        = layers[l].up_w.data();
        p_down[l]      = layers[l].down_w.data();
    }
    std::vector<int> layer_types(s.L_d, 0);
    layer_types[s.L_d - 1] = 1;  // last layer full attention

    std::vector<__half> ref_out(
        static_cast<std::size_t>(s.N_slots) * s.BLOCK_SIZE * s.D_emb);
    dflash_reference::drafter_forward_reference(
        input_tokens_emb.data(), k_cache.data(), v_cache.data(),
        slot_positions.data(),
        p_attn_norm.data(), p_q_w.data(), p_q_norm.data(), p_o_w.data(),
        p_ffn_norm.data(), p_gate.data(), p_up.data(), p_down.data(),
        layer_types.data(),
        s.swa_window, s.rope_base, s.norm_eps,
        s.BLOCK_SIZE, s.N_slots, s.SeqLen, s.L_d,
        s.D_emb, s.H_q, s.H_kv, s.D_h, s.intermediate,
        ref_out.data());

    // Confirm reference produced non-zero output.
    int nz = 0;
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    for (auto & h : ref_out) {
        const float f = __half2float(h);
        if (f != 0.0f) ++nz;
        sum_abs += std::abs(static_cast<double>(f));
        if (std::abs(f) > max_abs) max_abs = std::abs(f);
    }
    const double mean_abs = sum_abs / static_cast<double>(ref_out.size());
    std::printf("[smoke] reference output: nz=%d/%zu  mean_abs=%.4f  max_abs=%.4f\n",
                nz, ref_out.size(), mean_abs, max_abs);
    if (nz < static_cast<int>(ref_out.size() / 2)) {
        std::fprintf(stderr, "[FAIL] reference output unexpectedly sparse (%d non-zero of %zu)\n",
                     nz, ref_out.size());
        return 1;
    }
    std::printf("[smoke] PASS — reference is callable and produces non-degenerate output\n");
    return 0;
}

constexpr bool STUB_KERNEL = true;

int run_stub_kernel_check(int N_slots, int BLOCK_SIZE, int SeqLen) {
    std::printf("  [stub-check] N_slots=%d BLOCK_SIZE=%d SeqLen=%d\n",
                N_slots, BLOCK_SIZE, SeqLen);
    constexpr int D_emb = 5120;
    constexpr int L_d   = 5;
    constexpr int H_q   = 40;
    constexpr int H_kv  = 8;
    constexpr int D_h   = 128;
    constexpr int intermediate = 17408;
    constexpr int swa_window = 2048;
    constexpr float rope_base = 10000000.0f;
    constexpr float norm_eps = 1.0e-6f;

    const std::size_t n_out =
        static_cast<std::size_t>(N_slots) *
        static_cast<std::size_t>(BLOCK_SIZE) *
        static_cast<std::size_t>(D_emb);

    std::vector<__half> ref_h(n_out);
    dflash_reference::drafter_forward_reference_stub(
        ref_h.data(), N_slots, BLOCK_SIZE, D_emb);

    std::vector<__half> dev_h(n_out);

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
    for (std::size_t i = 0; i < n_out; ++i) {
        const int ulp = dflash_reference::fp16_ulp_delta(ref_h[i], dev_h[i]);
        if (ulp > max_ulp) max_ulp = ulp;
    }
    if (max_ulp != 0) {
        std::fprintf(stderr, "    [FAIL] stub agreement check: max_ulp=%d, expected 0\n", max_ulp);
        return 1;
    }
    std::printf("    [PASS] stub kernel and stub reference agree byte-identically\n");
    return 0;
}

} // anonymous namespace

int main() {
    std::printf("=== test-dflash-drafter-forward ===\n");

    // Phase 1: reference smoke test — confirms reference is callable.
    int sm = reference_smoke_test();
    if (sm != 0) {
        std::fprintf(stderr, "[FATAL] reference smoke test failed\n");
        return 1;
    }
    std::printf("\n");

    if constexpr (STUB_KERNEL) {
        // Phase 2 (stub mode): confirm stub kernel agrees with stub reference
        // on zeros — validates plumbing without the kernel body.
        const int rc = run_stub_kernel_check(2, 4, 64);
        if (rc != 0) return 1;
        std::printf("\n[OVERALL] reference smoke PASS + stub agreement PASS — returning SKIP (77) until kernel body lands\n");
        return 77;
    }

    // Phase 2 (active): kernel-vs-real-reference byte-identity sweep.
    // Wired once the kernel body lands. See plan in plan file.
    std::fprintf(stderr, "[ERROR] active sweep path not yet implemented — flip STUB_KERNEL=false once kernel + plumbing land\n");
    return 1;
}
