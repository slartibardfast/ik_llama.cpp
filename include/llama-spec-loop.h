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
