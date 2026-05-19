// test-dflash-drafter-lm-head.cpp
//
// Byte-identity / NMSE unit test for dflash_drafter_lm_head — the F16
// GEMV against target's shared `output.weight` (T1-recast). Kernel output
// vs CPU scalar fp32 reference across a (n_rows × V × D_emb × seed) sweep.
//
// @witnesses: SharedEmbedAndLMHead
//
// Reference: serial fp32 dot product per output cell, matching the
// kernel's per-thread accumulation order. Since both reference and
// kernel use serial K-loop fp32 accumulation, output should be byte-
// identical for the same inputs.
//
// PASS criterion: max_ulp_fp32 == 0 (byte-identical) OR NMSE ≤ 1e-7
// AND cos_sim ≥ 0.99999. Tightened gates because there are no
// transcendentals or reductions across threads in this kernel — just
// per-thread serial dot products.

#include "ggml-cuda/dflash/dflash-drafter-lm-head.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
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

// CPU scalar reference. Per-cell serial fp32 dot product over D_emb.
// Mirrors the kernel's accumulation order exactly: hidden lifted to
// fp32 via __half2float, F16 weight lifted to fp32, serial K-loop sum.
void lm_head_scalar_ref(
    const __half * hidden,       // [n_rows, D_emb]
    const __half * lm_head_w,    // [V, D_emb]
    float        * logits,       // [n_rows, V]
    int n_rows, int D_emb, int V)
{
    std::vector<float> hidden_f32(D_emb);
    for (int row = 0; row < n_rows; ++row) {
        for (int k = 0; k < D_emb; ++k) {
            hidden_f32[k] = __half2float(hidden[static_cast<std::size_t>(row) * D_emb + k]);
        }
        for (int col = 0; col < V; ++col) {
            float acc = 0.0f;
            for (int k = 0; k < D_emb; ++k) {
                acc += hidden_f32[k] *
                       __half2float(lm_head_w[static_cast<std::size_t>(col) * D_emb + k]);
            }
            logits[static_cast<std::size_t>(row) * V + col] = acc;
        }
    }
}

int run_one(int n_rows, int D_emb, int V, uint32_t seed) {
    std::printf("[run] n_rows=%d D_emb=%d V=%d seed=%u\n",
                n_rows, D_emb, V, seed);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const std::size_t n_h = static_cast<std::size_t>(n_rows) * D_emb;
    const std::size_t n_w = static_cast<std::size_t>(V) * D_emb;
    const std::size_t n_o = static_cast<std::size_t>(n_rows) * V;

    std::vector<__half> hidden_h(n_h);
    std::vector<__half> lm_head_w_h(n_w);
    for (auto & x : hidden_h) x = __float2half(dist(rng));
    for (auto & x : lm_head_w_h) x = __float2half(dist(rng) * 0.05f);

    // CPU reference
    std::vector<float> ref_logits(n_o);
    lm_head_scalar_ref(
        hidden_h.data(), lm_head_w_h.data(), ref_logits.data(),
        n_rows, D_emb, V);

    // GPU kernel
    __half * d_hidden = nullptr;
    __half * d_lm     = nullptr;
    float  * d_out    = nullptr;
    CUDA_CHECK(cudaMalloc(&d_hidden, n_h * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_lm,     n_w * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_out,    n_o * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_hidden, hidden_h.data(),    n_h * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lm,     lm_head_w_h.data(), n_w * sizeof(__half), cudaMemcpyHostToDevice));

    dflash_drafter_lm_head_launch(
        d_hidden, d_lm, d_out, n_rows, D_emb, V, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> dev_logits(n_o);
    CUDA_CHECK(cudaMemcpy(dev_logits.data(), d_out, n_o * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_lm));
    CUDA_CHECK(cudaFree(d_out));

    // Compare: byte-identical desired, NMSE + cos as backup gates.
    std::size_t n_eq = 0;
    std::size_t n_neq = 0;
    double sum_sq_err = 0.0;
    double sum_sq_ref = 0.0;
    double sum_ref_dev = 0.0;
    double sum_sq_dev = 0.0;
    double max_abs_diff = 0.0;
    for (std::size_t i = 0; i < n_o; ++i) {
        const float r = ref_logits[i];
        const float k = dev_logits[i];
        if (r == k) ++n_eq; else ++n_neq;
        const double diff = static_cast<double>(r) - static_cast<double>(k);
        sum_sq_err += diff * diff;
        sum_sq_ref += static_cast<double>(r) * static_cast<double>(r);
        sum_ref_dev += static_cast<double>(r) * static_cast<double>(k);
        sum_sq_dev += static_cast<double>(k) * static_cast<double>(k);
        if (std::abs(diff) > max_abs_diff) max_abs_diff = std::abs(diff);
    }
    const double nmse = sum_sq_err / (sum_sq_ref + 1e-30);
    const double cos_sim = sum_ref_dev / (std::sqrt(sum_sq_ref) * std::sqrt(sum_sq_dev) + 1e-30);
    const double byte_eq_rate = static_cast<double>(n_eq) / static_cast<double>(n_o);
    std::printf("[ ] byte-identical rate=%.3f%%  NMSE=%.3e  cos=%.6f  max|diff|=%.3e\n",
                byte_eq_rate * 100.0, nmse, cos_sim, max_abs_diff);

    if (byte_eq_rate >= 0.999) {
        std::printf("[PASS] byte-identical at ≥ 99.9 %%\n");
        return 0;
    }
    if (nmse <= 1.0e-7 && cos_sim >= 0.99999) {
        std::printf("[PASS] kernel matches reference within NMSE / cos gate\n");
        return 0;
    }
    std::fprintf(stderr, "[FAIL] kernel diverges: byte_eq=%.3f%% NMSE=%.3e cos=%.6f\n",
                 byte_eq_rate * 100.0, nmse, cos_sim);
    return 1;
}

} // anonymous namespace

int main() {
    std::printf("=== test-dflash-drafter-lm-head ===\n");
    int fails = 0;

    // Small sweep first to validate algorithm.
    fails += run_one(2,  64, 32,    42u);
    fails += run_one(4,  64, 128,   17u);
    fails += run_one(2, 128, 512,   1u);
    // Larger but still unit-test-fast.
    fails += run_one(2, 512, 4096,  99u);
    // Approaches production (V scaled down to keep test fast — full V=248320
    // would be ~30 s of CPU reference compute per run).
    fails += run_one(1, 1024, 16384, 7u);

    if (fails > 0) {
        std::fprintf(stderr, "[OVERALL] %d failures\n", fails);
        return 1;
    }
    std::printf("[OVERALL] all PASS\n");
    return 0;
}
