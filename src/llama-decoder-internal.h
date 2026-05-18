//
// PHASE 45 D9.6b internal: full struct llama_decoder definition.
//
// Exposes the decoder struct so llama_context can hold a `default_decoder`
// member by value and back-reference into it before any user-created
// decoder exists. Without this, perf-counter accesses during ctx
// construction (warmup decode) would deref a null `decoder_ref`.
//
// NOT in the public include/ tree; libcommon and external consumers
// must not include this. Removed at D9.8 once llama_context is deleted.
//
#ifndef LLAMA_DECODER_INTERNAL_H
#define LLAMA_DECODER_INTERNAL_H

#include "llama.h"
#include "llama-decoder.h"
#include "llama-impl.h"   // PHASE45 D9.6d: full llama_split_tensor definition
#include "qnext-state-slot-allocator.h"  // PHASE45 D9.6f: qnext slot allocator type
#include "ggml.h"         // PHASE45 D9.6f: ggml_free for ~llama_decoder
#include "ggml-backend.h"

#include <vector>
#include <map>
#include <set>

struct llama_session;

struct llama_decoder {
    struct llama_session    * session = nullptr;
    struct llama_decoder_params params{};
    enum   llama_decoder_role role    = LLAMA_DECODER_PRIMARY;

    // PHASE45 D9.6b: perf counters (formerly on llama_context).
    // Verify and draft decoders track separately.
    int64_t t_p_eval_us       = 0;
    int64_t t_eval_us         = 0;
    int64_t t_compute_start_us = 0;
    int64_t n_queued_tokens   = 0;
    int32_t n_p_eval          = 0;
    int32_t n_eval            = 0;

    // PHASE45 D9.6c: output buffers (formerly on llama_context).
    ggml_backend_buffer_t buf_output = nullptr;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers
    size_t  output_size = 0; // capacity (of tokens positions) for the output buffers
    int32_t n_outputs   = 0; // number of actually-used outputs in the current ubatch or last logical batch

    bool logits_all = false;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0; // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // PHASE45 D9.6e: scheduler + compute-meta + abort callback (formerly
    // on llama_context). Today there is one sched per ctx, shared across
    // all decoders dispatching against the same session. Until per-decoder
    // graph reservation lands, the ctx-owned default_decoder holds these
    // fields; user decoders (verify, draft) leave them empty and reach
    // the shared sched via ctx->default_decoder.sched. The `owns_sched`
    // flag lets future per-decoder allocators set this true and trigger
    // free-on-destruct without breaking the shared default_decoder path.
    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_sched_t sched = nullptr;
    bool                 owns_sched = false;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    // PHASE45 D9.6d: recurrent state per layer (Qwen3Next / DeltaNet).
    // Architecturally per-decoder (verify and draft track independent
    // recurrent trajectories on accept/reject). Today the storage is
    // still allocated by llama_kv_cache_init alongside k_l/v_l so the
    // ggml_context lifetimes stay co-located; default_decoder mirrors
    // pointers from cache.s_l after init. Per-decoder allocation
    // contexts land at D9.6g/D9.7 when split_cache also splits.
    std::vector<struct ggml_tensor *> s_l;          // per layer
    // For split layers: per-device tensor pointers, mirrored from
    // kv_cache.split_s_l. Same lifetime caveat as s_l.
    std::vector<struct llama_split_tensor> split_s_l;

    // PHASE45 D9.6f: graph input tensors (formerly on llama_context).
    // Per-decoder because verify and draft build independent graphs.
    struct ggml_tensor * inp_tokens      = nullptr; // I32 [n_batch]
    struct ggml_tensor * inp_embd        = nullptr; // F32 [n_embd, n_batch]
    struct ggml_tensor * inp_pos         = nullptr; // I32 [n_batch]
    struct ggml_tensor * inp_out_ids     = nullptr; // I32 [n_outputs]
    struct ggml_tensor * inp_KQ_mask     = nullptr; // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_KQ_mask_swa = nullptr; // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_K_shift     = nullptr; // I32 [kv_size]
    struct ggml_tensor * inp_mean        = nullptr; // F32 [n_batch, n_batch]
    struct ggml_tensor * inp_cls         = nullptr; // I32 [n_batch]
    struct ggml_tensor * inp_s_copy      = nullptr; // I32 [kv_size]
    struct ggml_tensor * inp_s_mask      = nullptr; // F32 [1, n_kv]
    struct ggml_tensor * inp_s_seq       = nullptr; // I32 [n_kv, n_batch]
    struct ggml_tensor * inp_s_seq_qnext = nullptr; // I32 [1, n_batch]
    struct ggml_tensor * inp_pos_bucket    = nullptr; // I32 [n_batch|n_kv, n_batch]
    struct ggml_tensor * inp_KQ_mask_cross = nullptr; // F32 [n_outputs_enc, n_batch]
    struct ggml_tensor * inp_scale         = nullptr; // F32 [n_tokens]
    struct ggml_tensor * inp_mtp_states    = nullptr;
    // Per-row K-loop bound for the deterministic FA path (sm_75 only).
    // Populated each decode step: for each query row i, set bound[i] =
    // (kv_self->seq_pos_max(seq_id_of_row_i) + 1). Consumed as src[5] of
    // ggml_flash_attn_ext_per_slot_kv. See specs/deltanet/fattn-per-slot-kv-sm75.md §15.6.
    struct ggml_tensor * inp_per_row_k_bound = nullptr; // I32 [q->ne[1]] (n_tok)

    // PHASE45 D9.6f: per-seq slot allocator for the linear-attn recurrent
    // state buffer (s_l[il]); maps llama_seq_id -> slot index.
    qnext_state_slot_allocator qnext_slot_alloc;
    // Phase 3 instrumentation: count how many times the qwen3next mixed-seq
    // chunking sub-batch was triggered.
    uint64_t qnext_mixed_seq_fallback_count = 0;

    // PHASE45 D9.6f: PHASE36 chain-seed buffer + pointer.
    std::vector<float> draft_input_hidden_state_buf;
    const float *      draft_input_hidden_state = nullptr;

    // PHASE45 D9.6f: PHASE36 cycle counter (verify-decode tally).
    int64_t mtp_cycle_counter = 0;

    // PHASE45 D9.6f: MTP DRAFT_GEN device-side residual cache.
    void *  draft_residual_dev          = nullptr;
    size_t  draft_residual_dev_nbytes   = 0;
    size_t  draft_residual_dev_capacity = 0;
    int     draft_residual_dev_device   = -1;
    bool    draft_residual_dev_valid    = false;

    // PHASE45 D9.6f: MTP DRAFT_GEN argmax cache.
    bool                  draft_argmax_valid = false;
    int32_t               draft_argmax_n     = 0;
    std::vector<int32_t>  draft_argmax_ids;
    std::vector<float>    draft_argmax_probs;

    // PHASE45 D9.6f: optional top-2 cache (LLAMA_PROBE_TOP2 / tree-K=2 path).
    bool                  draft_top2_armed = false;
    std::vector<int32_t>  draft_argmax_top2_ids;

    // PHASE45 D9.6f: caller-controlled enable for the verify-step
    // argmax-cache fast path.
    bool fast_argmax_for_verify = false;

    // PHASE45 D9.6f: pre-final-norm residual stream tag from the main
    // forward graph (qwen35 / qwen35moe + cparams.mtp).
    struct ggml_tensor * t_h_pre_norm = nullptr;

    // DFlash per-layer residual snapshots. Populated by the eval-callback
    // installed when cparams.dflash_extract_count > 0; each slot matches
    // the index in cparams.dflash_extract_layers. n_elements is the
    // contiguous float count actually written for the last decode.
    // Cap mirrors dflash_extract_layers[80] in llama-cparams.h (Qwen 3.6 27B
    // 65-layer capture + headroom).
    //
    // Inner vector is indexed by primary seq_id of each row, sized to
    // cparams.n_seq_max at llama_set_dflash_extract_layers time. The
    // cb_eval hook demuxes rows into the correct per-seq buffer using
    // dflash_ubatch_row_seq (populated by the decode driver before sched
    // compute). At n_seq_max=1 the seq_id=0 buffer is the only one used
    // and behavior is byte-identical to the pre-Phase-3 flat storage.
    std::vector<std::vector<float>> dflash_extract_buf[80];
    std::vector<size_t>             dflash_extract_n[80];

    // Per-row primary seq_id (seq_id[i][0]) of the current ubatch.
    // Length == u_batch.n_tokens; populated by llama_decode_internal
    // before ggml_backend_sched_graph_compute_async so the DFlash
    // cb_eval hook can demux rows by seq_id at append time.
    std::vector<llama_seq_id> dflash_ubatch_row_seq;

    // PHASE45 D9.6f: PHASE36 Step 1 fused multi-draft cgraph counters.
    int32_t mtp_fused_last_compute_count = 0;

    int32_t mtp_fused_results_n = 0;
    llama_token mtp_fused_results_tokens[8] = {};
    float       mtp_fused_results_probs[8]  = {};

    int32_t mtp_fused_offset_buf[8 * 16 /*MAX_DEVICES*/] = {};
    struct ggml_tensor * mtp_fused_offset_t[8] = {};
    int32_t mtp_fused_offset_n_dev[8] = {};

    // PHASE45 D9.6f: per-chain-step residuals from the fused cgraph.
    struct ggml_tensor * mtp_fused_chain_residuals[8] = {};

    // PHASE45 D9.6f: deferred extraction state (Phase 38 E).
    bool                  mtp_fused_skip_extraction = false;
    struct ggml_cgraph *  mtp_fused_pending_gf      = nullptr;
    int                   mtp_fused_pending_n_steps = 0;
    int                   mtp_fused_async_guess     = -1;

    // PHASE45 D9.6f: persistent chain-residual buffer (Phase 38 B).
    struct ggml_context     * mtp_persist_ctx = nullptr;
    ggml_backend_buffer_t     mtp_persist_buf = nullptr;
    struct ggml_tensor      * mtp_persist[8]  = {};
    int                       mtp_persist_n   = 0;

    // PHASE45 D9.6f: chain-residual seed step request (Phase 37 #2.2).
    int32_t pending_chain_residual_step = -1;

    // PHASE45 D9.6f: MTP per-ubatch hook counters.
    uint64_t mtp_hook_fire_count    = 0;
    uint64_t mtp_inline_decode_count = 0;

    ~llama_decoder() {
        // PHASE45 D9.6c: decoder owns its output buffer; default_decoder
        // (held by-value in llama_context) frees on ctx teardown.
        if (buf_output != nullptr) {
            ggml_backend_buffer_free(buf_output);
            buf_output = nullptr;
        }
        // PHASE45 D9.6e: only the sched-owning decoder frees. Today
        // owns_sched is true only on default_decoder; future per-decoder
        // graph reservation will set true on user decoders too.
        if (owns_sched && sched != nullptr) {
            ggml_backend_sched_free(sched);
            sched = nullptr;
        }
        // PHASE45 D9.6f: persistent chain-residual buffer cleanup
        // (Phase 38 B3). Today only default_decoder allocates these.
        if (mtp_persist_buf != nullptr) {
            ggml_backend_buffer_free(mtp_persist_buf);
            mtp_persist_buf = nullptr;
        }
        if (mtp_persist_ctx != nullptr) {
            ggml_free(mtp_persist_ctx);
            mtp_persist_ctx = nullptr;
        }
    }
};

#endif // LLAMA_DECODER_INTERNAL_H
