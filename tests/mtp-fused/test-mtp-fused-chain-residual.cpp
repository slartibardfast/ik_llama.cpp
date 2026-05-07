// test-mtp-fused-chain-residual.cpp
//
// Drives:
//   - mtp_fused_draft.allium ChainResidualParity
//
// Step k>0 of the fused chain consumes the SAME residual that the
// per-step path consumes from llama_get_embeddings_ith(ctx, 0) after
// step k-1's DRAFT_GEN decode. Per-step extracts "result_norm" (the
// shared_head_norm output of the MTP layer); fused must thread the
// same tensor between chain steps.
//
// This test fixes the chain-residual leak that produced the
// 2026-05-07 regression: fused acceptance dropped to 13% at d=5 vs
// per-step's 58% because step k+1 was being fed the post-FFN residual
// (pre-norm) instead of the post-shared-head-norm residual.
//
// Strategy:
//   1. Run llama_mtp_fused_draft_invoke with n_steps=N, capture
//      tokens[0..N).
//   2. Run the per-step path (LLAMA_MTP_FUSED unset) with the same
//      seed_token, seed_hidden, and KV state, capture per-step
//      argmax tokens [0..N).
//   3. Assert byte-identical token sequences.
//
// Greedy + identical inputs + identical KV ⇒ identical argmax. Any
// divergence indicates a chain-state mismatch (residual variant,
// causality bug, or stale KV).

#include "llama.h"

#include <cassert>
#include <cstdio>

int main() {
    // TODO(spec-driven, RED until per-step → fused parity harness
    // lands. The harness needs:
    //   - A reproducible seed_hidden (last row of h_pre_norm from a
    //     fixed verify forward).
    //   - Two ctx instances (or one with KV-snapshot/restore) so the
    //     per-step and fused runs see identical pre-state.
    //   - Per-step argmax capture (the per-step path doesn't surface
    //     an argmax-array API today; would need a thin observer hook
    //     in mtp_speculative_gen_draft).
    //
    // Until then this is RED-stub. The 2026-05-07 measurement (X02
    // 256K, LLAMA_MTP_FUSED=1, d=5) showed 13% accept vs per-step's
    // 58%, with step 0 matching per-step (~80% accept) and step >=1
    // diverging — bound here for regression coverage.

    fprintf(stderr,
            "TODO: implement ChainResidualParity test once the\n"
            "      per-step argmax observer hook + KV snapshot\n"
            "      primitive land.\n");

    return 77;
}
