// dflash-drafter-forward-reference.h
//
// CPU scalar reference for the 5-layer DFlash drafter forward. Mirrors
// the cooperative kernel in dflash-drafter-forward.cu step-for-step.
// Used as the test oracle in test-dflash-drafter-forward.cpp.
//
// Spec: specs/dflash/kernel-design.md §6.1.
//
// Algorithm — for each (slot, query_position) of the (N_slots, 1+BLOCK_SIZE)
// query span:
//   1. Embedding lookup: hidden = target_token_embd[input_token]  (D_emb)
//   2. For each layer ℓ in 0..L_d-1:
//        a. residual = hidden
//        b. hidden_n = RMSNorm(hidden, attn_norm_w[ℓ], norm_eps)
//        c. q = WMMA-oracle(hidden_n, q_w[ℓ])                 (H_q*D_h)
//           reshape q to (H_q, D_h)
//        d. q[h] = q_norm_w[ℓ] * RMSNorm(q[h])                (per head)
//        e. q = RoPE(q, query_position, rope_base, fp64-transcendentals)
//        f. K = k_cache[ℓ, slot, :, :, :]                     (already populated)
//           V = v_cache[ℓ, slot, :, :, :]
//        g. attention with mask:
//              - SWA layer (layer_types[ℓ] == 0): K-loop ∈
//                  [max(0, query_position - swa_window + 1), query_position]
//              - Full attention (layer_types[ℓ] == 1): K-loop ∈
//                  [0, query_position]
//              - Layer 4 (full) is bidirectional within the current block
//                (vLLM mask treats block positions symmetrically)
//           - scores = (Q @ K^T) / sqrt(D_h)  (scalar fp32 attention)
//           - softmax(scores)
//           - attn_out = softmax @ V
//           - reshape attn_out (H_q, D_h) → (H_q*D_h)
//        h. proj = WMMA-oracle(attn_out, o_w[ℓ])             (D_emb)
//        i. hidden = residual + proj
//        j. residual = hidden
//        k. hidden_n = RMSNorm(hidden, ffn_norm_w[ℓ], norm_eps)
//        l. gate = WMMA-oracle(hidden_n, gate_w[ℓ])          (intermediate)
//        m. up   = WMMA-oracle(hidden_n, up_w[ℓ])            (intermediate)
//        n. activated = silu(gate) * up                       (element-wise)
//        o. ffn_out = WMMA-oracle(activated, down_w[ℓ])      (D_emb)
//        p. hidden = residual + ffn_out
//   3. Output: out_hidden[slot, query_position, :] = hidden
//
// The cooperative kernel boundary (per spec §6.1) ends here. lm_head and
// argmax go in separate kernels with their own reference functions.
//
// Reference reads K and V from the pre-populated cache (caller's job:
// run dflash_inject_kv_fused first). The reference does NOT recompute
// K/V projections — that is the asymmetric K/V property
// (@KAsymmetricallyNormedVNot).
//
// Implementation note — query positions: the (1+BLOCK_SIZE) query
// positions of a slot's query span occupy contiguous KV cache positions
// starting at slot_positions[slot]. Position 0 of the query span =
// anchor_pos (== slot_positions[slot]). Position i of the query span
// (i in 1..BLOCK_SIZE) is the mask-token at anchor_pos + i.
//
// Output: out_hidden has shape [N_slots, BLOCK_SIZE, D_emb]. The anchor
// (query position 0) is dropped; only the BLOCK_SIZE mask-token output
// positions are written. This matches what the lm_head kernel consumes
// (it produces logits for the BLOCK_SIZE candidate tokens, not the
// anchor, which is a known input token).
//
// STUB IMPLEMENTATION: this reference currently returns zeros. Real body
// to be implemented in a follow-up commit before the kernel body lands;
// test compiles + runs but returns SKIP (exit 77) until both kernel and
// reference are non-stub.

#pragma once

#include <cuda_fp16.h>

#include <cstddef>
#include <cstring>

namespace dflash_reference {

inline void drafter_forward_reference_stub(
    __half * out_hidden,
    int      N_slots,
    int      BLOCK_SIZE,
    int      D_emb)
{
    std::memset(out_hidden, 0,
                static_cast<std::size_t>(N_slots) *
                static_cast<std::size_t>(BLOCK_SIZE) *
                static_cast<std::size_t>(D_emb) * sizeof(__half));
}

} // namespace dflash_reference
