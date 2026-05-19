// dflash-inject-kv.cu
//
// Per-drafter-layer fused KV projection + per-head K_norm + RoPE on K +
// cache scatter. V is never normed and never RoPE'd
// (per @KAsymmetricallyNormedVNot).
//
// Spec: specs/dflash/kernel-design.md §6.2 + §6.2.A
//
// Dispatch (revised 2026-05-19, spec §6.2.A): batched pinned-HMMA GEMM
// via `dflash_gemm_npc` for K_proj and V_proj (one call each per layer,
// batched across all (slot, anchor) pairs), followed by a thin post-
// process sub-kernel that does per-head K_norm + RoPE + cache scatter.
// The scalar fp32 K-loop kernel is retired; cuBLAS HGEMM forbidden.
//
// Layout coincidence: context_states[N_slots, MAL_anchors, D_d=5120]
// row-major is byte-identical to [M=N_slots*MAL_anchors, K=D_d=5120]
// for pinned. k_weight, v_weight are [D_kv=1024, D_d=5120] row-major.
//
// Allium bindings (unchanged):
//   @PerLayerArity, @HeadShapeMatchesDraft, @KAsymmetricallyNormedVNot,
//   @InjectedAnchorAlignment, @InjectPerLayerLaunches

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>

#include "dflash-inject-kv.cuh"
#include "dflash-gemm.cuh"

namespace {

constexpr int H_KV_LOCKED = 8;
constexpr int D_LOCKED    = 128;
constexpr int D_d_LOCKED  = 5120;
constexpr int D_kv_LOCKED = H_KV_LOCKED * D_LOCKED;  // 1024
constexpr int N_THREADS   = 128;                     // one position per thread per head
constexpr int OUTS_PER_THREAD = D_kv_LOCKED / N_THREADS;  // 8
constexpr int D_HALF      = D_LOCKED / 2;            // 64

// Per-row post-process: read F32 K and V from pinned outputs (shape
// [M, D_kv]), apply per-head RMSNorm + NeoX RoPE on K, scatter F16 K
// and F16 V to the per-layer cache at anchor_positions[slot, anchor].
//
// One CTA per (slot, anchor) row. Each thread owns ONE (head, position)
// output per slot — specifically register slot o ↔ (head=o, position=tid).
// Head h's 128 positions distributed across the 128 threads enables
// block-wide K_norm reduction via warp-shuffle + SMEM tree. RoPE pair
// partners are 64 lanes apart, so staging goes through SMEM.
__global__ void inject_kv_postprocess_kernel(
    const float  * __restrict__ k_proj_f32,        // [M, D_kv]
    const float  * __restrict__ v_proj_f32,        // [M, D_kv]
    const __half * __restrict__ k_norm_weight,     // [D=128]
    float                       rope_base,
    float                       norm_eps,
    __half       * __restrict__ k_cache_layer,     // [N_slots, SeqLen, H_kv, D]
    __half       * __restrict__ v_cache_layer,
    const int    * __restrict__ anchor_positions,  // [N_slots, MAL]
    int                         MAL_anchors,
    int                         SeqLen)
{
    const int row    = blockIdx.x;
    const int slot   = row / MAL_anchors;
    const int anchor = row % MAL_anchors;
    const int tid    = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int position = anchor_positions[slot * MAL_anchors + anchor];

    __shared__ float reduce_smem[4];
    __shared__ float K_smem[H_KV_LOCKED * D_LOCKED];

    // Per-thread fp32 registers: K[h]/V[h] = K_proj/V_proj[row, head=h, pos=tid].
    const float * row_k = k_proj_f32 + static_cast<std::size_t>(row) * D_kv_LOCKED;
    const float * row_v = v_proj_f32 + static_cast<std::size_t>(row) * D_kv_LOCKED;

    float K[OUTS_PER_THREAD];
    float V[OUTS_PER_THREAD];
    #pragma unroll
    for (int o = 0; o < OUTS_PER_THREAD; ++o) {
        const int idx = o * N_THREADS + tid;  // (head=o, pos=tid) in [H_kv * D]
        K[o] = row_k[idx];
        V[o] = row_v[idx];
    }

    // Per-head K_norm — RMSNorm across the 128 positions of head h.
    const float inv_D = 1.0f / static_cast<float>(D_LOCKED);
    const float knorm_w = __half2float(k_norm_weight[tid]);

    for (int h = 0; h < H_KV_LOCKED; ++h) {
        const float kh = K[h];
        float sq = kh * kh;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sq += __shfl_xor_sync(0xFFFFFFFFu, sq, offset);
        }
        if (lane == 0) reduce_smem[warp_id] = sq;
        __syncthreads();
        float total_sq;
        if (warp_id == 0) {
            float v = (lane < 4) ? reduce_smem[lane] : 0.0f;
            v += __shfl_xor_sync(0xFFFFFFFFu, v, 2);
            v += __shfl_xor_sync(0xFFFFFFFFu, v, 1);
            if (lane == 0) reduce_smem[0] = v;
        }
        __syncthreads();
        total_sq = reduce_smem[0];

        const float rsqrt_val = rsqrtf(total_sq * inv_D + norm_eps);
        K[h] = kh * rsqrt_val * knorm_w;
        __syncthreads();
    }

    // RoPE on K only — stage to SMEM for pair-partner access.
    #pragma unroll
    for (int h = 0; h < H_KV_LOCKED; ++h) {
        K_smem[h * D_LOCKED + tid] = K[h];
    }
    __syncthreads();

    const int partner_pos = (tid < D_HALF) ? (tid + D_HALF) : (tid - D_HALF);
    const int dim_idx     = tid % D_HALF;
    const double exp_val_d  = static_cast<double>(2 * dim_idx) / static_cast<double>(D_LOCKED);
    const double inv_freq_d = pow(static_cast<double>(rope_base), -exp_val_d);
    const double theta_d    = static_cast<double>(position) * inv_freq_d;
    const float c           = static_cast<float>(cos(theta_d));
    const float s           = static_cast<float>(sin(theta_d));

    #pragma unroll
    for (int h = 0; h < H_KV_LOCKED; ++h) {
        const float k_self    = K_smem[h * D_LOCKED + tid];
        const float k_partner = K_smem[h * D_LOCKED + partner_pos];
        if (tid < D_HALF) {
            K[h] = k_self * c - k_partner * s;
        } else {
            K[h] = k_partner * s + k_self * c;
        }
    }

    // Cache scatter — F16 store of post-norm-RoPE K and raw V.
    const std::size_t cache_base =
        ((static_cast<std::size_t>(slot) * SeqLen + position) * H_KV_LOCKED) * D_LOCKED;
    #pragma unroll
    for (int o = 0; o < OUTS_PER_THREAD; ++o) {
        const int output_idx = o * N_THREADS + tid;
        k_cache_layer[cache_base + output_idx] = __float2half(K[o]);
        v_cache_layer[cache_base + output_idx] = __float2half(V[o]);
    }
}

} // anonymous namespace

extern "C" void dflash_inject_kv_fused_launch(
    const __half * d_context_states,
    const __half * d_k_weight,
    const __half * d_v_weight,
    const __half * d_k_norm_weight,
    float          rope_base,
    float          norm_eps,
    __half       * d_k_cache_layer,
    __half       * d_v_cache_layer,
    const int    * d_anchor_positions,
    int            N_slots,
    int            MAL_anchors,
    int            H_kv,
    int            D,
    int            D_d,
    int            SeqLen,
    cudaStream_t   stream)
{
    // Hard-validate the locked shape. Out-of-shape calls zero the cache
    // so the test driver SKIPs rather than producing wrong values silently.
    if (H_kv != H_KV_LOCKED || D != D_LOCKED || D_d != D_d_LOCKED) {
        const std::size_t n_cells_bytes =
            static_cast<std::size_t>(N_slots) *
            static_cast<std::size_t>(SeqLen) *
            static_cast<std::size_t>(H_kv) *
            static_cast<std::size_t>(D) * sizeof(__half);
        cudaMemsetAsync(d_k_cache_layer, 0, n_cells_bytes, stream);
        cudaMemsetAsync(d_v_cache_layer, 0, n_cells_bytes, stream);
        return;
    }

    const int M = N_slots * MAL_anchors;
    if (M <= 0) return;

    // K-divisibility precondition for pinned HMMA (TILE_K=16).
    if (D_d_LOCKED % 16) {
        std::fprintf(stderr,
            "[dflash_inject_kv_fused_launch] pinned-HMMA requires K%%16==0, "
            "got K=%d\n", D_d_LOCKED);
        return;
    }

    // F32 scratch for K_proj and V_proj outputs: [M, D_kv].
    float * k_proj_f32 = nullptr;
    float * v_proj_f32 = nullptr;
    const std::size_t proj_bytes =
        static_cast<std::size_t>(M) * D_kv_LOCKED * sizeof(float);
    cudaMallocAsync(&k_proj_f32, proj_bytes, stream);
    cudaMallocAsync(&v_proj_f32, proj_bytes, stream);

    // K_proj — pinned HMMA, F16 act × F16 k_weight → F32 K_proj.
    dflash_gemm_npc(
        /*weight =*/d_k_weight,
        /*act    =*/d_context_states,
        /*dst_f32=*/k_proj_f32,
        /*K      =*/D_d_LOCKED,
        /*N_cols =*/D_kv_LOCKED,
        /*n_rows =*/M,
        stream);

    // V_proj — pinned HMMA.
    dflash_gemm_npc(
        /*weight =*/d_v_weight,
        /*act    =*/d_context_states,
        /*dst_f32=*/v_proj_f32,
        /*K      =*/D_d_LOCKED,
        /*N_cols =*/D_kv_LOCKED,
        /*n_rows =*/M,
        stream);

    // Per-row post-process: K_norm + RoPE on K, cache scatter.
    const dim3 grid(M);
    const dim3 block(N_THREADS);
    inject_kv_postprocess_kernel<<<grid, block, 0, stream>>>(
        k_proj_f32, v_proj_f32, d_k_norm_weight,
        rope_base, norm_eps,
        d_k_cache_layer, d_v_cache_layer, d_anchor_positions,
        MAL_anchors, SeqLen);

    cudaFreeAsync(k_proj_f32, stream);
    cudaFreeAsync(v_proj_f32, stream);
}
