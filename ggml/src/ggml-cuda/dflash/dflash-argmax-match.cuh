// dflash-argmax-match.cuh
//
// Launcher declaration for the DFlash argmax_match kernel — implements
// AcceptPrefixDecision contract from specs/dflash/dflash.allium.
//
// Spec: specs/dflash/kernel-design.md §6.5.
//
// Per (slot):
//   - For each of BLOCK_SIZE rows of drafter_logits: compute argmax →
//     draft_tokens[i]   (with tie-break: lowest token id)
//   - For each of BLOCK_SIZE+1 rows of target_logits: compute argmax →
//     target_argmax[i]
//   - n_accepted = longest prefix length where draft_tokens[i] ==
//     target_argmax[i] for i in [0, n_accepted)
//   - bonus_token = target_argmax[n_accepted]    (per @BonusIsArgmaxAtFirstUnacceptedRow)
//   - bonus_pos   = anchor_pos[slot] + n_accepted + 1  (per @BonusPosIsAnchorPlusNAcceptedPlusOne)
//
// Output arrays are int32 [N_slots] each.
//
// Allium witnesses (per specs/dflash/allium-tla-binding.json):
//   - LongestPrefixMatchUnderArgmax
//   - NAcceptedWithinBound
//   - BonusIsArgmaxAtFirstUnacceptedRow
//   - BonusPosIsAnchorPlusNAcceptedPlusOne
//   - DeterminismUnderFixedInputs
//   - ProbabilisticVerifyOutOfScope  (greedy-only kernel; no probabilistic path)

#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void dflash_argmax_match_launch(
    const float * d_drafter_logits,   // [N_slots, BLOCK_SIZE, V]
    const float * d_target_logits,    // [N_slots, BLOCK_SIZE + 1, V]
    const int   * d_anchor_pos,       // [N_slots]
    int         * d_n_accepted,       // [N_slots]
    int         * d_bonus_token,      // [N_slots]
    int         * d_bonus_pos,        // [N_slots]
    int           N_slots,
    int           BLOCK_SIZE,
    int           V,
    cudaStream_t  stream
);

#ifdef __cplusplus
}
#endif
