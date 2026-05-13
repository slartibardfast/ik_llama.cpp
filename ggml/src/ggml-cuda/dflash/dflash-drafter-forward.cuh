// dflash-drafter-forward.cuh
//
// Launcher declaration for the persistent cooperative kernel that runs
// the 5-layer DFlash drafter forward pass. Implementation:
// dflash-drafter-forward.cu.
//
// Spec: specs/dflash/kernel-design.md §6.1.
//
// The kernel produces the drafter's per-position hidden state at the
// (1 + BLOCK_SIZE) query positions of each slot's query span. The
// lm_head projection (hidden → logits) is a SEPARATE kernel
// (dflash-drafter-lm-head.cu) — this kernel outputs hidden states,
// not logits. See spec §6.1 "Kernel boundary — lm_head" for rationale.
//
// All pointer arguments are device pointers (cudaMalloc'd by caller).
// Caller is responsible for having populated k_cache and v_cache via
// `dflash_inject_kv_fused` prior to this kernel — drafter forward
// consumes the pre-populated cache, it does NOT compute K/V
// projections (per @KAsymmetricallyNormedVNot's symmetric K/V proj
// only at the InjectKV step, NOT inside the drafter forward).
//
// Allium witnesses (see specs/dflash/allium-tla-binding.json):
//   - SingleForwardPerStep              (one launch per cycle)
//   - QuerySpanIsOnePlusN               (input/output shape match)
//   - InjectionConsumedAtEveryLayer     (k_cache, v_cache read every layer)
//   - LayerTypeDependentMask            (K-loop bound per layer type)
//   - AnchorEmbeddingFromTarget         (input_tokens_emb sourced from target's token_embd)
//   - AnchorPosPreserved                (slot_positions flows through)
//   - BlockSizeBindsToConfig            (BLOCK_SIZE template param)
//   - DeterminismPerDeployment          (deterministic across runs)
//   - InjectionConsumedAtEveryLayer     (cache reads per layer)

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Locked drafter shape for z-lab/Qwen3.6-27B-DFlash (read at runtime
// against drafter GGUF metadata at server init; constexpr here for
// kernel compile-time dispatch).
//
//   - L_d = 5 drafter layers
//   - 4 SWA layers (0..3) with window = 2048
//   - 1 full-attention layer (layer 4)
//   - D_emb = 5120 hidden size
//   - H_q = 32 query heads, H_kv = 8 KV heads (GQA factor 4)
//   - D_h = 128 per-head dimension
//   - intermediate = 17408 MLP dim
//   - V = 248320 vocab
//   - rope_base = 10000000.0
//   - norm_eps = 1e-6

// Launch the drafter forward cooperative kernel.
//
// Shape parameters are passed at runtime (not all are template parameters
// because the kernel uses them in pointer arithmetic / loop bounds), but
// the locked-shape kernel only supports the values above. Out-of-shape
// calls zero the output buffer.
//
// BLOCK_SIZE must be one of {4, 5, 6, 8}.
void dflash_drafter_forward_launch(
    const __half * d_input_tokens_emb,      // [N_slots, 1+BLOCK_SIZE, D_emb]
    __half       * d_k_cache,               // [L_d, N_slots, SeqLen, H_kv, D_h] (drafter writes K/V at query positions)
    __half       * d_v_cache,               // [L_d, N_slots, SeqLen, H_kv, D_h]
    const int    * d_slot_positions,        // [N_slots] — anchor_pos for each slot
    const __half * const * d_layer_attn_norm_w,   // [L_d] pointers to [D_emb]
    const __half * const * d_layer_q_w,           // [L_d] pointers to [H_q*D_h, D_emb]
    const __half * const * d_layer_q_norm_w,      // [L_d] pointers to [D_h]
    const __half * const * d_layer_k_w,           // [L_d] pointers to [H_kv*D_h, D_emb]
    const __half * const * d_layer_k_norm_w,      // [L_d] pointers to [D_h]
    const __half * const * d_layer_v_w,           // [L_d] pointers to [H_kv*D_h, D_emb]
    const __half * const * d_layer_o_w,           // [L_d] pointers to [D_emb, H_q*D_h]
    const __half * const * d_layer_ffn_norm_w,    // [L_d] pointers to [D_emb]
    const __half * const * d_layer_gate_w,        // [L_d] pointers to [intermediate, D_emb]
    const __half * const * d_layer_up_w,          // [L_d] pointers to [intermediate, D_emb]
    const __half * const * d_layer_down_w,        // [L_d] pointers to [D_emb, intermediate]
    const __half * d_output_norm_w,                // [D_emb] — final RMSNorm before lm_head
    const int    * d_layer_types,                  // [L_d] — 0 = SWA, 1 = full
    int            swa_window,                     // = 2048 for Qwen3.6-27B-DFlash
    float          rope_base,                      // = 10000000.0
    float          norm_eps,                       // = 1e-6
    int            BLOCK_SIZE,                     // {4, 5, 6, 8}
    int            N_slots,
    int            SeqLen,
    int            L_d,
    int            D_emb,
    int            H_q,
    int            H_kv,
    int            D_h,
    int            intermediate,
    __half       * d_out_hidden,                  // [N_slots, BLOCK_SIZE, D_emb] — output
    cudaStream_t   stream
);

#ifdef __cplusplus
}
#endif
