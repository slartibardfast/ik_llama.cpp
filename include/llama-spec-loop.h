//
// Copyright (C) 2023-2026 The llama.cpp authors
// Copyright (C) 2024-2026 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
// PHASE 45 D5: Public API for `llama_spec_loop`.
//
// Orchestrates speculative decoding: holds one verify decoder and N
// draft decoders, runs the accept/reject algorithm, manages KV
// transactions for draft writes, and produces accepted tokens per
// step.
//
// Today's `common/speculative.cpp` is the body that becomes this type.
// Single-draft MTP (`-mtp --draft 3`) instantiates with N=1. Tree
// drafting (PHASE40) instantiates with N=K branches. Dual-model spec
// decoding instantiates with the draft decoder bound to a different
// session/model than verify.
//

#ifndef LLAMA_SPEC_LOOP_H
#define LLAMA_SPEC_LOOP_H

#include "llama.h"
#include "llama-spec.h"  // PHASE45 D10.b: llama_spec_mtp_slot_in/out

#ifdef __cplusplus
extern "C" {
#endif

    struct llama_spec_loop;
    struct llama_decoder;

    struct llama_spec_loop_params {
        // Acceptance threshold: drafts with chain probability below this
        // are not emitted (skip the speculation cost).
        float min_chain_prob;

        // Maximum draft depth per step. For MTP: 1..n_steps inclusive.
        int   max_draft_depth;

        // Stop criterion (see llama_sampler API for sampler chain;
        // spec_loop applies the chain to verify's logits).
        struct llama_sampler * sampler;
    };

    LLAMA_API struct llama_spec_loop_params llama_spec_loop_default_params(void);

    // Construct a spec loop with one verify decoder + N draft decoders.
    // The verify decoder's role MUST be LLAMA_DECODER_VERIFY (or
    // PRIMARY); each draft's role MUST be a DRAFT or TREE_BRANCH variant.
    // All decoders MUST share the same session as verify.
    LLAMA_API struct llama_spec_loop * llama_spec_loop_create(
            struct llama_decoder        * verify,
            struct llama_decoder       ** drafts,
            int                           n_drafts,
            struct llama_spec_loop_params params);

    LLAMA_API void llama_spec_loop_free(struct llama_spec_loop * loop);

    // Granular API (D8.2): mirrors server's existing draft + accept flow.
    // The caller drives the verify decode + accept-prefix computation
    // between these two calls.
    //
    // Generate up to `n_draft_max` MTP tokens off the loop's draft
    // decoder. Caller is responsible for the verify forward and the
    // accept-prefix logic that follows. Returns drafts in `drafts_out`.
    // Returns count, or negative on setup error.
    LLAMA_API int32_t llama_spec_loop_gen_drafts(
            struct llama_spec_loop * loop,
            llama_token              id_last,
            float                    p_min,
            int32_t                  n_draft_max,
            llama_seq_id             seq_id,
            llama_pos                n_past,
            llama_token            * drafts_out);

    // Inform the loop that `n_accepted` of the last drafted tokens were
    // accepted by the verifier. Updates internal stats.
    LLAMA_API void llama_spec_loop_accept_n(
            struct llama_spec_loop * loop,
            int32_t                  n_accepted);

    // PHASE45 D10.b: batched-draft wrapper. Each `loops[i]` contributes
    // one slot to the batch; all loops MUST share the same verify_decoder
    // and draft_decoder pair (post-D9.5 collapsed-context invariant: each
    // per-slot common_speculative_state_mtp wraps the same shared ctx_tgt).
    //
    // `slots[i]` carries the slot's seq_id / id_last / n_past / n_draft_max.
    // `drafts_out` is slot-major with stride `drafts_out_stride` (in
    // tokens). `outs[i]` returns the per-slot count + truncation flag.
    //
    // Returns total drafts emitted, or negative on setup error.
    LLAMA_API int32_t llama_spec_loop_gen_drafts_batched(
            struct llama_spec_loop      ** loops,
            int32_t                        n_loops,
            const llama_spec_mtp_slot_in * slots,
            float                          p_min,
            llama_token                  * drafts_out,
            int32_t                        drafts_out_stride,
            llama_spec_mtp_slot_out      * outs);

    // Step the loop with a prompt batch (for first call) or an empty
    // batch (subsequent calls). Returns number of accepted tokens, or
    // negative on error.
    //
    // Acceptance algorithm:
    //   1. Each draft decoder produces N speculative tokens.
    //   2. Verify decoder runs forward on the speculation.
    //   3. Accept the longest prefix where verify agrees with drafts.
    //   4. Commit accepted tokens to session (via kv_txn); roll back the
    //      tail.
    //   5. Sample one additional token from verify's logits at the first
    //      mismatch position.
    LLAMA_API int32_t llama_spec_loop_step(
            struct llama_spec_loop * loop,
            struct llama_batch       batch);

    // Read the most recent step's accepted token IDs.
    // Caller must not free the returned pointer.
    LLAMA_API const llama_token * llama_spec_loop_last_accepted(
            const struct llama_spec_loop * loop,
            int32_t                      * n_accepted);

    // Statistics (cumulative since loop creation)
    LLAMA_API float   llama_spec_loop_accept_rate    (const struct llama_spec_loop * loop);
    LLAMA_API int64_t llama_spec_loop_n_drafted      (const struct llama_spec_loop * loop);
    LLAMA_API int64_t llama_spec_loop_n_accepted     (const struct llama_spec_loop * loop);
    LLAMA_API int64_t llama_spec_loop_n_verify_steps (const struct llama_spec_loop * loop);

    // Per-decoder access (for instrumentation, perf collection, sampling
    // intervention).
    LLAMA_API struct llama_decoder * llama_spec_loop_verify(const struct llama_spec_loop * loop);
    LLAMA_API struct llama_decoder * llama_spec_loop_draft (const struct llama_spec_loop * loop, int idx);
    LLAMA_API int                    llama_spec_loop_n_draft_decoders(const struct llama_spec_loop * loop);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_SPEC_LOOP_H
