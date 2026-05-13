// dflash-drafter-forward.cu
//
// Persistent cooperative kernel for the 5-layer DFlash drafter forward.
// Currently a stub: launcher zeros the output buffer and returns. Body
// to be implemented per kernel-design.md §6.1 once the test harness +
// scalar reference are in place.
//
// Spec: specs/dflash/kernel-design.md §6.1.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "dflash-drafter-forward.cuh"

extern "C" void dflash_drafter_forward_launch(
    const __half * d_input_tokens_emb,
    const __half * d_k_cache,
    const __half * d_v_cache,
    const int    * d_slot_positions,
    const __half * const * d_layer_attn_norm_w,
    const __half * const * d_layer_q_w,
    const __half * const * d_layer_q_norm_w,
    const __half * const * d_layer_o_w,
    const __half * const * d_layer_ffn_norm_w,
    const __half * const * d_layer_gate_w,
    const __half * const * d_layer_up_w,
    const __half * const * d_layer_down_w,
    const int    * d_layer_types,
    int            swa_window,
    float          rope_base,
    float          norm_eps,
    int            BLOCK_SIZE,
    int            N_slots,
    int            SeqLen,
    int            L_d,
    int            D_emb,
    int            H_q,
    int            H_kv,
    int            D_h,
    int            intermediate,
    __half       * d_out_hidden,
    cudaStream_t   stream)
{
    // Stub: zero the output buffer. Real body comes in subsequent commits.
    (void) d_input_tokens_emb;
    (void) d_k_cache;
    (void) d_v_cache;
    (void) d_slot_positions;
    (void) d_layer_attn_norm_w;
    (void) d_layer_q_w;
    (void) d_layer_q_norm_w;
    (void) d_layer_o_w;
    (void) d_layer_ffn_norm_w;
    (void) d_layer_gate_w;
    (void) d_layer_up_w;
    (void) d_layer_down_w;
    (void) d_layer_types;
    (void) swa_window;
    (void) rope_base;
    (void) norm_eps;
    (void) SeqLen;
    (void) L_d;
    (void) H_q;
    (void) H_kv;
    (void) D_h;
    (void) intermediate;

    const std::size_t n_out_bytes =
        static_cast<std::size_t>(N_slots) *
        static_cast<std::size_t>(BLOCK_SIZE) *
        static_cast<std::size_t>(D_emb) * sizeof(__half);
    cudaMemsetAsync(d_out_hidden, 0, n_out_bytes, stream);
}
