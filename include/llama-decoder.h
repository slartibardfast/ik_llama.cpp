//
// Copyright (C) 2023-2026 The llama.cpp authors
// Copyright (C) 2024-2026 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
// PHASE 45 D5: Public API for `llama_decoder`.
//
// A decoder is a parameterized executor running against a `llama_session`.
// Each decoder owns its own scheduler, recurrent state slots, output
// buffers, batch tracking, and graph builder. The role determines which
// graph the decoder builds:
//
//   PRIMARY      — standalone full forward (single-context users)
//   VERIFY       — full forward producing logits to validate drafts
//   DRAFT_MTP    — MTP-only forward producing speculative tokens
//   TREE_BRANCH  — MTP forward for a specific tree branch (PHASE40+)
//
// Multiple decoders can share a session. Recurrent state stays per-decoder
// because draft and verify run independent trajectories.
//

#ifndef LLAMA_DECODER_H
#define LLAMA_DECODER_H

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

    struct llama_decoder;
    struct llama_session;

    enum llama_decoder_role {
        LLAMA_DECODER_PRIMARY     = 0,  // standalone full forward
        LLAMA_DECODER_VERIFY      = 1,  // spec-decoding verifier
        LLAMA_DECODER_DRAFT_MTP   = 2,  // spec-decoding MTP draft
        LLAMA_DECODER_TREE_BRANCH = 3,  // spec-decoding tree branch
    };

    struct llama_decoder_params {
        enum llama_decoder_role role;

        // Compute
        uint32_t n_threads;
        uint32_t n_threads_batch;

        // Behavior flags
        bool causal_attn;
        bool embeddings;
        bool fused_moe_up_gate;
        bool grouped_expert_routing;
        bool fused_up_gate;
        bool fused_mmad;
        bool rope_cache;
        bool graph_reuse;
        bool scheduler_async;

        // MoE controls
        int   min_experts;
        float thresh_experts;

        // MTP-specific (only meaningful for VERIFY / DRAFT_MTP / TREE_BRANCH roles)
        int   mtp_fused_n_steps;     // 0 = not fused
        int   mtp_fused_n_extend;    // PHASE38 C extended chain
        bool  mtp_inline_kv_hook;    // PHASE36 Step 3 KV write hook

        // Scheduler eval callback
        ggml_backend_sched_eval_callback cb_eval;
        void                           * cb_eval_user_data;

        // Tree-branch only (ignored for other roles)
        int    tree_branch_id;       // [0..K) for TREE_BRANCH role
        float  tree_branch_min_prob;
    };

    LLAMA_API struct llama_decoder_params llama_decoder_default_params(enum llama_decoder_role role);

    LLAMA_API struct llama_decoder * llama_decoder_create(
            struct llama_session         * session,
            struct llama_decoder_params    params);

    LLAMA_API void llama_decoder_free(struct llama_decoder * decoder);

    // Accessors
    LLAMA_API enum llama_decoder_role llama_decoder_role_of    (const struct llama_decoder * decoder);
    LLAMA_API struct llama_session  * llama_decoder_session    (const struct llama_decoder * decoder);
    LLAMA_API const struct llama_model * llama_decoder_model   (const struct llama_decoder * decoder);

    // Behavior toggles
    LLAMA_API void llama_decoder_set_n_threads (struct llama_decoder * decoder, uint32_t n_threads, uint32_t n_threads_batch);
    LLAMA_API void llama_decoder_set_causal    (struct llama_decoder * decoder, bool causal_attn);
    LLAMA_API void llama_decoder_set_embeddings(struct llama_decoder * decoder, bool embeddings);
    LLAMA_API void llama_decoder_set_warmup    (struct llama_decoder * decoder, bool warmup);

    // Forward pass
    LLAMA_API int32_t llama_decoder_decode(struct llama_decoder * decoder, struct llama_batch batch);
    LLAMA_API int32_t llama_decoder_encode(struct llama_decoder * decoder, struct llama_batch batch);

    // Synchronize pending compute (for async-dispatched decoders)
    LLAMA_API void llama_decoder_synchronize(struct llama_decoder * decoder);

    // Output access
    LLAMA_API float * llama_decoder_get_logits        (struct llama_decoder * decoder);
    LLAMA_API float * llama_decoder_get_logits_ith    (struct llama_decoder * decoder, int32_t i);
    LLAMA_API float * llama_decoder_get_embeddings    (struct llama_decoder * decoder);
    LLAMA_API float * llama_decoder_get_embeddings_ith(struct llama_decoder * decoder, int32_t i);
    LLAMA_API float * llama_decoder_get_embeddings_seq(struct llama_decoder * decoder, llama_seq_id seq_id);

    // Performance counters (per-decoder; verify and draft are tracked separately)
    LLAMA_API struct llama_timings llama_decoder_timings    (const struct llama_decoder * decoder);
    LLAMA_API void                 llama_decoder_perf_reset(      struct llama_decoder * decoder);
    LLAMA_API void                 llama_decoder_perf_print(const struct llama_decoder * decoder);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_DECODER_H
