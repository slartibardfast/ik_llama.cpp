// dflash-argmax-match.cu
//
// Greedy-only argmax + accept-prefix + bonus-token kernel.
//
// Spec: specs/dflash/kernel-design.md §6.5.
//
// Design:
//   - One CTA per slot, 256 threads.
//   - For each of (BLOCK_SIZE + BLOCK_SIZE+1) logit rows:
//       1. Each thread strides across V columns, finds local max
//          (value, id) with tie-break favouring lowest id.
//       2. Block-wide butterfly + SMEM-tree reduction over (value, id)
//          pairs, same tie-break rule. lane 0 has global argmax.
//       3. Lane 0 writes argmax to SMEM scratch.
//   - After all rows processed, lane 0 walks SMEM:
//       - Find n_accepted = max k where draft[i] == target_argmax[i]
//         for all i < k.
//       - bonus_token = target_argmax[n_accepted]
//       - bonus_pos   = anchor_pos[slot] + n_accepted + 1
//     Write outputs.

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>

#include "dflash-argmax-match.cuh"

namespace {

constexpr int ARGMAX_THREADS = 256;
constexpr int ARGMAX_WARPS   = ARGMAX_THREADS / 32;
constexpr int MAX_BLOCK_SIZE = 8;  // template-instantiated upper bound

// Pair (value, id) tie-break: select winner with HIGHER value; if values
// are equal, select LOWER id.
__device__ inline void pair_max_lt(float & a_val, int & a_id, float b_val, int b_id) {
    if (b_val > a_val || (b_val == a_val && b_id < a_id)) {
        a_val = b_val;
        a_id  = b_id;
    }
}

// Warp-wide butterfly that simultaneously reduces (value, id) by the
// max-with-low-id-tie-break operator. After the butterfly, lane 0 holds
// the warp's argmax.
__device__ inline void warp_argmax(float & val, int & id) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_val = __shfl_xor_sync(0xFFFFFFFFu, val, offset);
        const int   other_id  = __shfl_xor_sync(0xFFFFFFFFu, id,  offset);
        pair_max_lt(val, id, other_val, other_id);
    }
}

// Block-wide argmax. `vals_smem[ARGMAX_WARPS]` and `ids_smem[ARGMAX_WARPS]`
// are caller-provided per-warp reduction slots.
__device__ inline void block_argmax(
    float val, int id,
    float * vals_smem, int * ids_smem,
    int   * out_id, float * out_val)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    warp_argmax(val, id);
    if (lane == 0) {
        vals_smem[warp] = val;
        ids_smem[warp]  = id;
    }
    __syncthreads();

    if (warp == 0) {
        float v = (lane < ARGMAX_WARPS) ? vals_smem[lane] : -CUDART_INF_F;
        int   i = (lane < ARGMAX_WARPS) ? ids_smem[lane]  : INT_MAX;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float ov = __shfl_xor_sync(0xFFFFFFFFu, v, offset);
            const int   oi = __shfl_xor_sync(0xFFFFFFFFu, i, offset);
            pair_max_lt(v, i, ov, oi);
        }
        if (lane == 0) {
            *out_id  = i;
            *out_val = v;
        }
    }
    __syncthreads();
}

__global__ void dflash_argmax_match_kernel(
    const float * __restrict__ drafter_logits,  // [N_slots, BS, V]
    const float * __restrict__ target_logits,   // [N_slots, BS+1, V]
    const int   * __restrict__ anchor_pos,      // [N_slots]
    int         * __restrict__ n_accepted_out,  // [N_slots]
    int         * __restrict__ bonus_token_out, // [N_slots]
    int         * __restrict__ bonus_pos_out,   // [N_slots]
    int          BLOCK_SIZE,
    int          V)
{
    const int slot = blockIdx.x;
    const int tid  = threadIdx.x;

    __shared__ float warp_vals[ARGMAX_WARPS];
    __shared__ int   warp_ids [ARGMAX_WARPS];
    __shared__ int   draft_tokens [MAX_BLOCK_SIZE];
    __shared__ int   target_argmax[MAX_BLOCK_SIZE + 1];
    __shared__ int   final_argmax;
    __shared__ float final_val;

    // Drafter argmax: BLOCK_SIZE rows.
    for (int row = 0; row < BLOCK_SIZE; ++row) {
        const float * row_p = drafter_logits +
            (static_cast<std::size_t>(slot) * BLOCK_SIZE + row) * V;
        float my_val = -CUDART_INF_F;
        int   my_id  = INT_MAX;
        for (int c = tid; c < V; c += blockDim.x) {
            pair_max_lt(my_val, my_id, row_p[c], c);
        }
        block_argmax(my_val, my_id, warp_vals, warp_ids, &final_argmax, &final_val);
        if (tid == 0) draft_tokens[row] = final_argmax;
        __syncthreads();
    }

    // Target argmax: BLOCK_SIZE + 1 rows.
    for (int row = 0; row < BLOCK_SIZE + 1; ++row) {
        const float * row_p = target_logits +
            (static_cast<std::size_t>(slot) * (BLOCK_SIZE + 1) + row) * V;
        float my_val = -CUDART_INF_F;
        int   my_id  = INT_MAX;
        for (int c = tid; c < V; c += blockDim.x) {
            pair_max_lt(my_val, my_id, row_p[c], c);
        }
        block_argmax(my_val, my_id, warp_vals, warp_ids, &final_argmax, &final_val);
        if (tid == 0) target_argmax[row] = final_argmax;
        __syncthreads();
    }

    // Lane 0 computes n_accepted, bonus_token, bonus_pos.
    if (tid == 0) {
        int n_accepted = 0;
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            if (draft_tokens[i] == target_argmax[i]) {
                ++n_accepted;
            } else {
                break;
            }
        }
        n_accepted_out[slot]   = n_accepted;
        bonus_token_out[slot]  = target_argmax[n_accepted];
        bonus_pos_out[slot]    = anchor_pos[slot] + n_accepted + 1;
    }
}

} // anonymous namespace

extern "C" void dflash_argmax_match_launch(
    const float * d_drafter_logits,
    const float * d_target_logits,
    const int   * d_anchor_pos,
    int         * d_n_accepted,
    int         * d_bonus_token,
    int         * d_bonus_pos,
    int           N_slots,
    int           BLOCK_SIZE,
    int           V,
    cudaStream_t  stream)
{
    if (BLOCK_SIZE > MAX_BLOCK_SIZE) {
        cudaMemsetAsync(d_n_accepted, 0, N_slots * sizeof(int), stream);
        cudaMemsetAsync(d_bonus_token, 0, N_slots * sizeof(int), stream);
        cudaMemsetAsync(d_bonus_pos, 0, N_slots * sizeof(int), stream);
        return;
    }
    const dim3 grid(N_slots);
    const dim3 block(ARGMAX_THREADS);
    dflash_argmax_match_kernel<<<grid, block, 0, stream>>>(
        d_drafter_logits, d_target_logits, d_anchor_pos,
        d_n_accepted, d_bonus_token, d_bonus_pos,
        BLOCK_SIZE, V);
}
