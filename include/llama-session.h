//
// Copyright (C) 2023-2026 The llama.cpp authors
// Copyright (C) 2024-2026 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
// PHASE 45 D5: Public API for `llama_session`.
//
// A session owns model-aligned sequence state — transformer K/V cache, cell
// metadata, position tracking, defragmentation. It is shareable across
// multiple `llama_decoder` instances (verify + draft for spec decoding,
// or N tree-draft branches), each of which carries its own per-execution
// state (recurrent layers, scheduler, output buffers).
//
// One session per tenant. Multiple decoders per session.
//

#ifndef LLAMA_SESSION_H
#define LLAMA_SESSION_H

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

    struct llama_session;

    struct llama_session_params {
        uint32_t n_ctx;          // context length
        uint32_t n_seq_max;      // max parallel sequences (slots)
        uint32_t n_batch;        // logical batch size
        uint32_t n_ubatch;       // physical sub-batch size

        enum ggml_type type_k;   // K cache dtype
        enum ggml_type type_v;   // V cache dtype

        // YaRN / RoPE
        float rope_freq_base;
        float rope_freq_scale;
        uint32_t n_ctx_orig_yarn;
        float yarn_ext_factor;
        float yarn_attn_factor;
        float yarn_beta_fast;
        float yarn_beta_slow;

        // Defrag
        float defrag_thold;

        // Hadamard transforms on K/V (PHASE38 F territory; orthogonal to PHASE45)
        bool k_cache_hadamard;
        bool v_cache_hadamard;

        // MLA / split-mode
        int  mla_attn;
        bool split_mode_graph_scheduling;

        // Flash attention
        bool flash_attn;

        // Offload K/V to GPU
        bool offload_kqv;
    };

    LLAMA_API struct llama_session_params llama_session_default_params(void);

    LLAMA_API struct llama_session * llama_session_create(
            struct llama_model           * model,
            struct llama_session_params    params);

    // PHASE45 D6/D7 helper: wrap an externally-created llama_context as
    // a non-owning session. session_free will NOT free the wrapped ctx;
    // the caller retains ownership. Used while common's
    // llama_init_from_gpt_params remains the canonical context factory.
    // Removed at D10 once common is ported.
    LLAMA_API struct llama_session * llama_session_adopt(struct llama_context * ctx);

    LLAMA_API void llama_session_free(struct llama_session * session);

    // Accessors
    LLAMA_API const struct llama_model * llama_session_model    (const struct llama_session * session);
    LLAMA_API uint32_t                   llama_session_n_ctx    (const struct llama_session * session);
    LLAMA_API uint32_t                   llama_session_n_seq_max(const struct llama_session * session);
    LLAMA_API uint32_t                   llama_session_n_batch  (const struct llama_session * session);
    LLAMA_API uint32_t                   llama_session_n_ubatch (const struct llama_session * session);

    // KV cache operations (formerly llama_kv_self_*)
    LLAMA_API void llama_session_kv_clear  (struct llama_session * session);
    LLAMA_API bool llama_session_kv_seq_rm (struct llama_session * session, llama_seq_id seq, llama_pos p0, llama_pos p1);
    LLAMA_API void llama_session_kv_seq_cp (struct llama_session * session, llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1);
    LLAMA_API void llama_session_kv_seq_keep(struct llama_session * session, llama_seq_id seq);
    LLAMA_API void llama_session_kv_seq_add (struct llama_session * session, llama_seq_id seq, llama_pos p0, llama_pos p1, llama_pos delta);
    LLAMA_API void llama_session_kv_seq_div (struct llama_session * session, llama_seq_id seq, llama_pos p0, llama_pos p1, int d);
    LLAMA_API llama_pos llama_session_kv_seq_pos_max(struct llama_session * session, llama_seq_id seq);

    LLAMA_API void llama_session_kv_update(struct llama_session * session);
    LLAMA_API void llama_session_kv_defrag(struct llama_session * session);

    // State save/load (full session state, including all sequences)
    LLAMA_API size_t llama_session_state_get_size(struct llama_session * session);
    LLAMA_API size_t llama_session_state_get_data(struct llama_session * session, uint8_t * dst, size_t size);
    LLAMA_API size_t llama_session_state_set_data(struct llama_session * session, const uint8_t * src, size_t size);

    // Per-sequence state save/load
    LLAMA_API size_t llama_session_state_seq_get_size(struct llama_session * session, llama_seq_id seq);
    LLAMA_API size_t llama_session_state_seq_get_data(struct llama_session * session, uint8_t * dst, size_t size, llama_seq_id seq);
    LLAMA_API size_t llama_session_state_seq_set_data(struct llama_session * session, const uint8_t * src, size_t size, llama_seq_id seq);

    // Adapters (LoRA, control vectors)
    LLAMA_API int32_t llama_session_lora_adapter_set   (struct llama_session * session, struct llama_lora_adapter * adapter, float scale);
    LLAMA_API int32_t llama_session_lora_adapter_remove(struct llama_session * session, struct llama_lora_adapter * adapter);
    LLAMA_API void    llama_session_lora_adapter_clear (struct llama_session * session);

    LLAMA_API int32_t llama_session_control_vector_apply(
            struct llama_session * session,
            const float          * data,
            size_t                 len,
            int32_t                n_embd,
            int32_t                il_start,
            int32_t                il_end);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_SESSION_H
