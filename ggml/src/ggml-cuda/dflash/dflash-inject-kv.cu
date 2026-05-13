// dflash-inject-kv.cu
//
// Per-(slot, anchor) per-drafter-layer fused KV projection:
//   K_proj + V_proj + per-head K_norm + RoPE(K) + cache write.
// V is never normed and never RoPE'd (per @KAsymmetricallyNormedVNot).
//
// Spec:
//   - specs/dflash/kernel-design.md §6.2 (kernel contract)
//   - specs/dflash/dflash.allium @PerLayerArity, @HeadShapeMatchesDraft,
//                                @KAsymmetricallyNormedVNot,
//                                @InjectedAnchorAlignment,
//                                @InjectPerLayerLaunches
//
// Design deviation from §6.2 (matches the combine_features deviation):
// Uses scalar fp32 accumulators rather than WMMA, for byte-identity with
// the fp32 scalar reference. WMMA fragment-internal reduction order does
// not match a serial K-order scalar oracle. Performance: bandwidth-bound
// either way at our shapes.
//
// Output layout per (slot, anchor) for THIS layer:
//   output_idx = o * N_THREADS + tid  for o in [0, OUTS_PER_THREAD)
// With OUTS_PER_THREAD = 8 and N_THREADS = 128, each thread owns ONE
// (head, position) output per slot — specifically:
//   thread tid, register slot o: K/V[head = o, position = tid]
// This places head h's 128 positions across all 128 threads of the CTA,
// so K_norm per-head reductions go via a block-wide warp-shuffle +
// SMEM-tree reduction. Pair-partner positions for RoPE differ in lane
// by 64, so RoPE staging goes through SMEM (cross-warp shuffle would be
// needed otherwise).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "dflash-inject-kv.cuh"

namespace {

constexpr int H_KV_LOCKED       = 8;
constexpr int D_LOCKED          = 128;
constexpr int D_d_LOCKED        = 5120;
constexpr int D_kv_LOCKED       = H_KV_LOCKED * D_LOCKED;       // 1024
constexpr int N_THREADS         = 128;
constexpr int OUTS_PER_THREAD   = D_kv_LOCKED / N_THREADS;       // 8
constexpr int K_TILE            = 128;
constexpr int D_HALF            = D_LOCKED / 2;                  // 64

__global__ void dflash_inject_kv_fused_kernel(
    const __half * __restrict__ context_states,    // [N_slots, MAL, D_d]
    const __half * __restrict__ k_weight,          // [H_kv*D, D_d] for this layer
    const __half * __restrict__ v_weight,          // [H_kv*D, D_d] for this layer
    const __half * __restrict__ k_norm_weight,     // [D]
    float                       rope_base,
    float                       norm_eps,
    __half       * __restrict__ k_cache_layer,     // [N_slots, SeqLen, H_kv, D]
    __half       * __restrict__ v_cache_layer,
    const int    * __restrict__ anchor_positions,  // [N_slots, MAL]
    int                         MAL_anchors,
    int                         SeqLen)
{
    const int slot     = blockIdx.x;
    const int anchor   = blockIdx.y;
    const int tid      = threadIdx.x;
    const int warp_id  = tid >> 5;
    const int lane     = tid & 31;
    const int position = anchor_positions[slot * MAL_anchors + anchor];

    // Per-(slot, anchor) input pointer — flatten of [slot, anchor, :, :].
    const __half * ctx = context_states
                       + (static_cast<std::size_t>(slot) * MAL_anchors + anchor)
                         * static_cast<std::size_t>(D_d_LOCKED);

    // SMEM scratch:
    //   ctx_smem        — source K-tile staging                (256 B)
    //   reduce_smem     — per-warp partial sums for K_norm     (16 B)
    //   K_smem          — K register-spill for RoPE pair access (4 KiB)
    __shared__ __half ctx_smem[K_TILE];
    __shared__ float  reduce_smem[4];
    __shared__ float  K_smem[H_KV_LOCKED * D_LOCKED];

    // Per-thread fp32 accumulators. Register slot o ↔ head o, position tid.
    float K[OUTS_PER_THREAD];
    float V[OUTS_PER_THREAD];
    #pragma unroll
    for (int o = 0; o < OUTS_PER_THREAD; ++o) { K[o] = 0.0f; V[o] = 0.0f; }

    // --- Steps 1 & 2: K_proj and V_proj GEMVs ---
    // Same pattern as combine_features: stream fc/k_weight from HBM (L2-
    // cacheable across (slot, anchor) CTAs sharing this layer's weights),
    // serial fp32 accumulation matches scalar reference's K-iteration order.
    for (int k_base = 0; k_base < D_d_LOCKED; k_base += K_TILE) {
        ctx_smem[tid] = ctx[k_base + tid];
        __syncthreads();

        for (int o = 0; o < OUTS_PER_THREAD; ++o) {
            const int my_row = o * N_THREADS + tid;
            const __half * wk = k_weight + static_cast<std::size_t>(my_row) * D_d_LOCKED + k_base;
            const __half * wv = v_weight + static_cast<std::size_t>(my_row) * D_d_LOCKED + k_base;
            float ka = K[o];
            float va = V[o];
            #pragma unroll 8
            for (int kk = 0; kk < K_TILE; ++kk) {
                const float src = __half2float(ctx_smem[kk]);
                ka += __half2float(wk[kk]) * src;
                va += __half2float(wv[kk]) * src;
            }
            K[o] = ka;
            V[o] = va;
        }
        __syncthreads();
    }

    // --- Step 3: K_norm — per-head RMSNorm with shared per-layer weight ---
    // Each thread owns ONE K value per head; head h's 128 positions are
    // distributed across the 128 threads. Reduction is block-wide:
    // per-thread sum_sq (trivial — one element) → warp-shuffle → SMEM tree.
    const float inv_D = 1.0f / static_cast<float>(D_LOCKED);
    const float knorm_w = __half2float(k_norm_weight[tid]);

    for (int h = 0; h < H_KV_LOCKED; ++h) {
        const float kh = K[h];
        float sq = kh * kh;

        // Warp-shuffle butterfly within warp
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sq += __shfl_xor_sync(0xFFFFFFFFu, sq, offset);
        }
        // SMEM tree across the 4 warps
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

        // Apply norm to thread's K value for this head:
        //   K[h, pos=tid] = K[h, pos=tid] * rsqrt * k_norm_weight[tid]
        K[h] = kh * rsqrt_val * knorm_w;

        __syncthreads();  // protect reduce_smem reuse for next head
    }

    // --- Step 4: RoPE on K only ---
    // Stage K to SMEM so each thread can read its pair partner's K value.
    // Layout: K_smem[head * D + position]. Thread tid writes its 8 K
    // values to K_smem[h*D + tid] for h in 0..7.
    #pragma unroll
    for (int h = 0; h < H_KV_LOCKED; ++h) {
        K_smem[h * D_LOCKED + tid] = K[h];
    }
    __syncthreads();

    // Pair partner position. tid in [0, D) since N_THREADS == D.
    const int partner_pos = (tid < D_HALF) ? (tid + D_HALF) : (tid - D_HALF);
    const int dim_idx     = tid % D_HALF;        // 0..63 (same for both pair members)
    // fp64 evaluation of pow/cos/sin keeps the kernel byte-aligned with
    // the CPU-side scalar reference: CUDA libdevice and glibc both
    // target ≤ 1 fp64 ULP for these functions, so the fp32 cast at use
    // absorbs the residual delta to identical fp32 bits in nearly all
    // cases. fp32 versions of these (powf, cosf, sinf) diverge enough
    // (≤ 6 ULP for powf, ≤ 2 ULP for cosf/sinf per CUDA docs) to push
    // the final fp16 output past the ≤ 2-ULP test gate at multi-anchor
    // configurations where larger positions stress the trig path.
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
            // tid is the "lo" position of its pair: out = lo*c - hi*s
            K[h] = k_self * c - k_partner * s;
        } else {
            // tid is the "hi" position: out = lo*s + hi*c
            // (partner holds true lo, self holds true hi)
            K[h] = k_partner * s + k_self * c;
        }
    }
    // No __syncthreads() needed — K_smem read is complete, K values now
    // in registers ready for cache write.

    // --- Step 5: Vectorized cache write ---
    // Cache layout: [N_slots, SeqLen, H_kv, D].
    // For thread tid, register o → (slot, position, head=o, pos=tid).
    const std::size_t cache_base =
        ((static_cast<std::size_t>(slot) * SeqLen + position)
          * H_KV_LOCKED) * D_LOCKED;
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
    // Hard-validate the locked shape. Out-of-shape calls zero the cache so
    // the test driver SKIPs rather than producing wrong values silently.
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

    const dim3 grid(N_slots, MAL_anchors);
    const dim3 block(N_THREADS);

    dflash_inject_kv_fused_kernel<<<grid, block, 0, stream>>>(
        d_context_states, d_k_weight, d_v_weight, d_k_norm_weight,
        rope_base, norm_eps,
        d_k_cache_layer, d_v_cache_layer, d_anchor_positions,
        MAL_anchors, SeqLen);
}
