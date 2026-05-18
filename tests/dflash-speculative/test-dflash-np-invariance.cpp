// test-dflash-np-invariance.cpp
//
// Empirical np-invariance probe for the DFlash drafter kernels.
//
// Closure binding: when drafter_forward is run at N_slots ∈ {1, 2, 4, 8}
// with IDENTICAL slot 0 content (replicated across all slots, same
// slot_position for each slot, same weights), slot 0's output bytes must
// be byte-identical across all four N values.
//
// This is a kernel-level invariance probe — it intentionally bypasses the
// target attention path (where the production multi-slot bug lives per
// `project_mtp_multislot_determinism_investigation_failed`) to focus on
// whether the drafter kernels honor the TML 3-kernel batch-invariance
// pattern (kernel-design.md §5.5) at the implementation level.
//
// Architectural prediction (per code read): drafter_forward.cu uses
// per-row CTA launches (`grid_rows = N_slots * Q`) with no cross-row
// reductions, no atomicAdd, fixed block sizes. By construction this is
// the TML pattern. This probe empirically witnesses that prediction.
//
// PASS criterion: byte-identical slot 0 output across N ∈ {1, 2, 4, 8}.
// FAIL criterion: ANY differing byte at slot 0 across any pair of N values.
//
// @witnesses: DeterminismPerDeployment
// @witnesses: SingleForwardPerStep
// @witnesses: QuerySpanIsOnePlusN

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

struct TinyShape {
    int L_d         = 2;
    int BLOCK_SIZE  = 4;
    int SeqLen      = 32;
    int D_emb       = 64;
    int H_q         = 4;
    int H_kv        = 2;
    int D_h         = 16;
    int intermediate = 96;
    int swa_window  = 16;
    float rope_base = 10000.0f;
    float norm_eps  = 1.0e-6f;
};

struct LayerWeights {
    std::vector<__half> attn_norm;
    std::vector<__half> q_w;
    std::vector<__half> q_norm;
    std::vector<__half> k_w;
    std::vector<__half> k_norm;
    std::vector<__half> v_w;
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
    L.k_w.resize(static_cast<std::size_t>(s.H_kv * s.D_h) * s.D_emb);
    for (auto & h : L.k_w) h = __float2half(d_w(rng));
    L.k_norm.resize(s.D_h);
    for (auto & h : L.k_norm) h = __float2half(d_n(rng));
    L.v_w.resize(static_cast<std::size_t>(s.H_kv * s.D_h) * s.D_emb);
    for (auto & h : L.v_w) h = __float2half(d_w(rng));
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

// Holders for device-side allocations for one N value.
struct DeviceArrays {
    __half * d_input_emb       = nullptr;
    __half * d_k_cache         = nullptr;
    __half * d_v_cache         = nullptr;
    __half * d_out             = nullptr;
    int    * d_slot_positions  = nullptr;
    int    * d_layer_types     = nullptr;
    std::vector<__half *> d_attn_norm, d_q_w, d_q_norm, d_k_w, d_k_norm,
                         d_v_w, d_o_w, d_ffn_norm, d_gate, d_up, d_down;
    __half * d_output_norm     = nullptr;
};

void free_device(DeviceArrays & d) {
    cudaFree(d.d_input_emb); cudaFree(d.d_k_cache); cudaFree(d.d_v_cache);
    cudaFree(d.d_out); cudaFree(d.d_slot_positions); cudaFree(d.d_layer_types);
    for (auto p : d.d_attn_norm) cudaFree(p);
    for (auto p : d.d_q_w)       cudaFree(p);
    for (auto p : d.d_q_norm)    cudaFree(p);
    for (auto p : d.d_k_w)       cudaFree(p);
    for (auto p : d.d_k_norm)    cudaFree(p);
    for (auto p : d.d_v_w)       cudaFree(p);
    for (auto p : d.d_o_w)       cudaFree(p);
    for (auto p : d.d_ffn_norm)  cudaFree(p);
    for (auto p : d.d_gate)      cudaFree(p);
    for (auto p : d.d_up)        cudaFree(p);
    for (auto p : d.d_down)      cudaFree(p);
    cudaFree(d.d_output_norm);
}

// Upload one fp16 tensor; track ptr in `out_vec`.
int upload_layer_w(const std::vector<__half> & h_w, std::vector<__half *> & out_vec) {
    __half * p = nullptr;
    cudaError_t err = cudaMalloc(&p, h_w.size() * sizeof(__half));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMemcpy(p, h_w.data(), h_w.size() * sizeof(__half), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(p);
        return 1;
    }
    out_vec.push_back(p);
    return 0;
}

// Run drafter_forward at given N_slots with identical content per slot.
// On return, host_out_slot0 contains the BLOCK_SIZE × D_emb fp16 bytes of
// slot 0's output. Returns 0 on success, nonzero on error.
int run_at_N(int N_slots, const TinyShape & s, const std::vector<LayerWeights> & layers,
             const std::vector<__half> & one_slot_input_emb,  // [Q * D_emb]
             const std::vector<__half> & one_slot_k_cache,    // [L_d * SeqLen * H_kv * D_h]
             const std::vector<__half> & one_slot_v_cache,    // same
             int anchor_pos,
             const std::vector<__half> & output_norm_w,
             const std::vector<int> & layer_types,
             std::vector<__half> & host_out_slot0)
{
    const int Q = 1 + s.BLOCK_SIZE;
    const std::size_t per_slot_input = (std::size_t) Q * s.D_emb;
    const std::size_t per_slot_cache = (std::size_t) s.L_d * s.SeqLen * s.H_kv * s.D_h;

    // Build N-slot replicated input
    std::vector<__half> input_emb_N((std::size_t) N_slots * per_slot_input);
    for (int slot = 0; slot < N_slots; ++slot) {
        std::memcpy(input_emb_N.data() + slot * per_slot_input,
                    one_slot_input_emb.data(),
                    per_slot_input * sizeof(__half));
    }

    // For caches the layout is [L_d, N_slots, SeqLen, H_kv, D_h] — slot index
    // is the 2nd dim. Repack: for each layer, the slot rows are contiguous.
    const std::size_t per_layer_slot = (std::size_t) s.SeqLen * s.H_kv * s.D_h;
    std::vector<__half> k_cache_N((std::size_t) s.L_d * N_slots * per_layer_slot);
    std::vector<__half> v_cache_N((std::size_t) s.L_d * N_slots * per_layer_slot);
    for (int l = 0; l < s.L_d; ++l) {
        const __half * src_k = one_slot_k_cache.data() + l * per_layer_slot;
        const __half * src_v = one_slot_v_cache.data() + l * per_layer_slot;
        for (int slot = 0; slot < N_slots; ++slot) {
            __half * dst_k = k_cache_N.data() + ((std::size_t) l * N_slots + slot) * per_layer_slot;
            __half * dst_v = v_cache_N.data() + ((std::size_t) l * N_slots + slot) * per_layer_slot;
            std::memcpy(dst_k, src_k, per_layer_slot * sizeof(__half));
            std::memcpy(dst_v, src_v, per_layer_slot * sizeof(__half));
        }
    }

    std::vector<int> slot_positions(N_slots, anchor_pos);

    DeviceArrays d;
    const std::size_t n_input = input_emb_N.size();
    const std::size_t n_cache = k_cache_N.size();
    const std::size_t n_out   = (std::size_t) N_slots * s.BLOCK_SIZE * s.D_emb;

    CUDA_CHECK(cudaMalloc(&d.d_input_emb, n_input * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d.d_k_cache,   n_cache * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d.d_v_cache,   n_cache * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d.d_out,       n_out * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d.d_slot_positions, N_slots * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d.d_layer_types,    s.L_d * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d.d_input_emb, input_emb_N.data(), n_input * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_k_cache,   k_cache_N.data(),   n_cache * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_v_cache,   v_cache_N.data(),   n_cache * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_slot_positions, slot_positions.data(), N_slots * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.d_layer_types,    layer_types.data(),    s.L_d * sizeof(int),   cudaMemcpyHostToDevice));

    // Initialize output to a known sentinel so a no-op kernel can't accidentally
    // produce identical "outputs" across N.
    CUDA_CHECK(cudaMemset(d.d_out, 0xFF, n_out * sizeof(__half)));

    // Upload per-layer weights
    for (int l = 0; l < s.L_d; ++l) {
        if (upload_layer_w(layers[l].attn_norm, d.d_attn_norm)) { free_device(d); return 1; }
        if (upload_layer_w(layers[l].q_w,       d.d_q_w))       { free_device(d); return 1; }
        if (upload_layer_w(layers[l].q_norm,    d.d_q_norm))    { free_device(d); return 1; }
        if (upload_layer_w(layers[l].k_w,       d.d_k_w))       { free_device(d); return 1; }
        if (upload_layer_w(layers[l].k_norm,    d.d_k_norm))    { free_device(d); return 1; }
        if (upload_layer_w(layers[l].v_w,       d.d_v_w))       { free_device(d); return 1; }
        if (upload_layer_w(layers[l].o_w,       d.d_o_w))       { free_device(d); return 1; }
        if (upload_layer_w(layers[l].ffn_norm,  d.d_ffn_norm))  { free_device(d); return 1; }
        if (upload_layer_w(layers[l].gate_w,    d.d_gate))      { free_device(d); return 1; }
        if (upload_layer_w(layers[l].up_w,      d.d_up))        { free_device(d); return 1; }
        if (upload_layer_w(layers[l].down_w,    d.d_down))      { free_device(d); return 1; }
    }
    CUDA_CHECK(cudaMalloc(&d.d_output_norm, s.D_emb * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d.d_output_norm, output_norm_w.data(),
                          s.D_emb * sizeof(__half), cudaMemcpyHostToDevice));

    dflash_drafter_forward_launch(
        d.d_input_emb, d.d_k_cache, d.d_v_cache, d.d_slot_positions,
        d.d_attn_norm.data(), d.d_q_w.data(), d.d_q_norm.data(),
        d.d_k_w.data(), d.d_k_norm.data(), d.d_v_w.data(),
        d.d_o_w.data(), d.d_ffn_norm.data(),
        d.d_gate.data(), d.d_up.data(), d.d_down.data(),
        d.d_output_norm, d.d_layer_types,
        s.swa_window, s.rope_base, s.norm_eps,
        s.BLOCK_SIZE, N_slots, /*n_slots_cap*/ N_slots, s.SeqLen, s.L_d,
        s.D_emb, s.H_q, s.H_kv, s.D_h, s.intermediate,
        d.d_out, /*stream=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Pull back slot 0's output (the first BLOCK_SIZE * D_emb rows).
    const std::size_t slot0_count = (std::size_t) s.BLOCK_SIZE * s.D_emb;
    host_out_slot0.resize(slot0_count);
    CUDA_CHECK(cudaMemcpy(host_out_slot0.data(), d.d_out,
                          slot0_count * sizeof(__half), cudaMemcpyDeviceToHost));

    free_device(d);
    return 0;
}

// FNV-1a 64-bit over a byte buffer (non-crypto, but deterministic + easy to log).
uint64_t fnv1a64(const void * data, std::size_t n) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    uint64_t h = 0xCBF29CE484222325ULL;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001B3ULL;
    }
    return h;
}

bool is_stub_output(const std::vector<__half> & buf) {
    // A buffer that's still all-0xFF (the sentinel) means the kernel didn't
    // overwrite the output — kernel is in stub state.
    bool all_ff = true;
    for (auto h : buf) {
        uint16_t b;
        std::memcpy(&b, &h, sizeof(b));
        if (b != 0xFFFF) { all_ff = false; break; }
    }
    return all_ff;
}

} // namespace

// Run one full (seed) configuration: generate weights + single-slot inputs
// for that seed, run drafter_forward at N ∈ {1,2,4,8}, and verify all four
// slot 0 outputs are byte-identical. Returns 0 on PASS, 1 on FAIL, 77 on SKIP.
int run_one_seed(uint32_t seed, const TinyShape & s) {
    std::printf("[seed=%u] ----\n", seed);
    std::mt19937 rng(seed);

    // Weights
    std::vector<LayerWeights> layers;
    layers.reserve(s.L_d);
    for (int l = 0; l < s.L_d; ++l) layers.push_back(gen_layer_weights(s, rng));

    std::uniform_real_distribution<float> d_n(0.5f, 1.5f);
    std::vector<__half> output_norm_w(s.D_emb);
    for (auto & h : output_norm_w) h = __float2half(d_n(rng));

    std::vector<int> layer_types(s.L_d, 0);
    layer_types[s.L_d - 1] = 1;  // last layer is full attention

    // Single-slot inputs
    std::uniform_real_distribution<float> d_in(-0.2f, 0.2f);
    const int Q = 1 + s.BLOCK_SIZE;
    std::vector<__half> one_slot_input_emb((std::size_t) Q * s.D_emb);
    for (auto & h : one_slot_input_emb) h = __float2half(d_in(rng));

    const std::size_t per_layer_slot = (std::size_t) s.SeqLen * s.H_kv * s.D_h;
    std::vector<__half> one_slot_k_cache((std::size_t) s.L_d * per_layer_slot);
    std::vector<__half> one_slot_v_cache((std::size_t) s.L_d * per_layer_slot);
    for (auto & h : one_slot_k_cache) h = __float2half(d_in(rng));
    for (auto & h : one_slot_v_cache) h = __float2half(d_in(rng));

    const int anchor_pos = 8;

    const int Ns[] = {1, 2, 4, 8};
    constexpr int N_COUNT = sizeof(Ns) / sizeof(Ns[0]);
    std::vector<__half> slot0_outputs[N_COUNT];
    uint64_t hashes[N_COUNT] = {0, 0, 0, 0};

    for (int i = 0; i < N_COUNT; ++i) {
        const int N = Ns[i];
        if (run_at_N(N, s, layers, one_slot_input_emb, one_slot_k_cache, one_slot_v_cache,
                     anchor_pos, output_norm_w, layer_types, slot0_outputs[i])) {
            std::fprintf(stderr, "  [N=%d] FAIL: kernel run errored\n", N);
            return 1;
        }
        if (is_stub_output(slot0_outputs[i])) {
            std::printf("  [N=%d] SKIP: kernel output is still sentinel 0xFF — stub state\n", N);
            return 77;
        }
        hashes[i] = fnv1a64(slot0_outputs[i].data(),
                            slot0_outputs[i].size() * sizeof(__half));
        std::printf("  [N=%d]  slot 0 FNV-1a64: 0x%016llx\n",
                    N, (unsigned long long) hashes[i]);
    }

    // Compare all N outputs to N=1's output
    bool all_equal = true;
    for (int i = 1; i < N_COUNT; ++i) {
        const std::vector<__half> & a = slot0_outputs[0];
        const std::vector<__half> & b = slot0_outputs[i];
        int n_diff = 0, max_ulp = 0, worst_idx = -1;
        for (std::size_t j = 0; j < a.size(); ++j) {
            uint16_t ua, ub;
            std::memcpy(&ua, &a[j], sizeof(ua));
            std::memcpy(&ub, &b[j], sizeof(ub));
            if (ua != ub) {
                ++n_diff;
                int d = std::abs(static_cast<int>(ua) - static_cast<int>(ub));
                if (d > max_ulp) { max_ulp = d; worst_idx = static_cast<int>(j); }
            }
        }
        if (n_diff == 0) {
            std::printf("  [N=1 vs N=%d]  BYTE-IDENTICAL (%zu elements)\n", Ns[i], a.size());
        } else {
            std::printf("  [N=1 vs N=%d]  DIFFER: %d/%zu bytes (max_ulp=%d, worst_idx=%d)\n",
                        Ns[i], n_diff, a.size(), max_ulp, worst_idx);
            if (worst_idx >= 0) {
                std::printf("    worst: N=1 = %.6e, N=%d = %.6e\n",
                            __half2float(a[worst_idx]),
                            Ns[i], __half2float(b[worst_idx]));
            }
            all_equal = false;
        }
    }

    return all_equal ? 0 : 1;
}

int main() {
    int dev_count = 0;
    cudaError_t derr = cudaGetDeviceCount(&dev_count);
    if (derr != cudaSuccess || dev_count == 0) {
        std::printf("SKIP: no CUDA device available\n");
        return 77;
    }

    TinyShape s;
    std::printf("test-dflash-np-invariance: L_d=%d BLOCK_SIZE=%d D_emb=%d H_q=%d H_kv=%d D_h=%d intermediate=%d SeqLen=%d\n",
                s.L_d, s.BLOCK_SIZE, s.D_emb, s.H_q, s.H_kv, s.D_h, s.intermediate, s.SeqLen);
    std::printf("Sweep: seeds × N ∈ {1,2,4,8} — slot 0 output must be byte-identical across N for each seed.\n");

    const uint32_t seeds[] = {31337, 42, 137, 4242};
    const int n_seeds = sizeof(seeds) / sizeof(seeds[0]);
    int passes = 0, skips = 0, fails = 0;
    for (int i = 0; i < n_seeds; ++i) {
        const int rc = run_one_seed(seeds[i], s);
        if      (rc == 0)  ++passes;
        else if (rc == 77) ++skips;
        else               ++fails;
    }
    std::printf("---\n");
    std::printf("sweep summary: %d/%d seeds PASSed (fails=%d skips=%d)\n",
                passes, n_seeds, fails, skips);
    if (fails == 0 && passes > 0) {
        std::printf("[PASS] drafter_forward is np-invariant across N ∈ {1,2,4,8} for %d seeds\n", passes);
        return 0;
    }
    std::printf("[FAIL] drafter_forward np-invariance violated\n");
    return 1;
}
