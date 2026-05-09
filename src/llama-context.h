#pragma once

#include "llama-impl.h"
#include "llama-cparams.h"
#include "llama-sampling.h"
#include "qnext-state-slot-allocator.h"
#include "qnext-seq-pattern.h"

struct llama_model;

#include <vector>
#include <map>
#include <set>
#include <memory>

struct llama_kv_cell {
    llama_pos pos   = -1;
    llama_pos delta = 0;
    int32_t   src   = 0; // used by recurrent state models to copy states

    std::set<llama_seq_id> seq_id;

    bool has_seq_id(const llama_seq_id & id) const {
        return seq_id.find(id) != seq_id.end();
    }

    bool is_empty() const {
        return seq_id.empty();
    }

    bool is_same_seq(const llama_kv_cell & other) const {
        return seq_id == other.seq_id;
    }
};

// ring-buffer of cached KV data
struct llama_kv_cache {
    bool has_shift = false;
    bool do_defrag = false;
    bool do_copy   = false;
    bool recurrent = false; // with recurrent state models, a cell can hold the state for more than one past token
    bool hybrid    = false;
    bool v_trans   = true;  // the value tensor is transposed

    // Note: The value of head isn't only used to optimize searching
    // for a free KV slot. llama_decode_internal also uses it, so it
    // cannot be freely changed after a slot has been allocated.
    uint32_t head = 0;
    uint32_t size = 0;
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // computed before each graph build
    uint32_t n = 0;

    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;

    std::vector<llama_kv_cell> cells;

    std::vector<struct ggml_tensor *> k_l; // per layer
    std::vector<struct ggml_tensor *> v_l;
    std::vector<struct ggml_tensor *> s_l; // per layer recurrent state storage (Qwen3Next)

    // When true, the delta_net graph builder will enable per-step SSM state saves
    bool save_per_step_ssm = false;

    std::vector<llama_split_tensor> split_k_l;
    std::vector<llama_split_tensor> split_v_l;
    std::vector<llama_split_tensor> split_s_l;

    std::vector<struct ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    size_t total_size() const {
        size_t size = 0;
        for (ggml_backend_buffer_t buf : bufs) {
            size += ggml_backend_buffer_get_size(buf);
        }
        return size;
    }

    // GPU-resident checkpoint for recurrent/hybrid speculative decoding
    struct gpu_checkpoint {
        std::vector<llama_kv_cell> cells_snapshot;
        uint32_t head_snapshot = 0;
        uint32_t used_snapshot = 0;

        std::vector<ggml_tensor *> s_l_shadow;

        std::vector<std::vector<ggml_tensor *>> split_s_l_shadow;

        // Per-step SSM state checkpoints for speculative decoding.
        // For non-split layers: per_step_ssm[il] is the checkpoint tensor.
        // For split layers: per_step_ssm[il] is nullptr; use per_step_ssm_split[il][device].
        std::vector<ggml_tensor *> per_step_ssm;

        // Per-step conv feature buffer: stores qkv_mixed features from the
        // verification forward pass so conv state can be reconstructed at any step.
        // One tensor per recurrent layer, each sized [conv_dim * max_tokens].
        // For split layers: per_step_qkv[il] is nullptr; use per_step_qkv_split[il][device].
        std::vector<ggml_tensor *> per_step_qkv;

        // Per-device per-step checkpoints for split recurrent layers.
        std::vector<std::vector<ggml_tensor *>> per_step_ssm_split;
        std::vector<std::vector<ggml_tensor *>> per_step_qkv_split;

        int32_t per_step_n_tokens = 0;
        int32_t per_step_max_allocated = 0;
        int64_t per_step_ssm_state_size = 0;
        int64_t per_step_conv_state_dim = 0;
        int64_t per_step_conv_dim = 0;
        int32_t per_step_d_conv = 0;

        int selected_spec_mode = -1;

        // Serialised sequence state for CPU mode
        std::vector<uint8_t> cpu_state_data;

        // Separate storage for per-step allocations
        std::vector<struct ggml_context *>   per_step_ctxs;
        std::vector<ggml_backend_buffer_t>   per_step_bufs;

        std::vector<struct ggml_context *>   shadow_ctxs;
        std::vector<ggml_backend_buffer_t>   shadow_bufs;

        bool allocated = false;
        bool saved     = false;

        ~gpu_checkpoint() {
            for (struct ggml_context * ctx : shadow_ctxs) {
                ggml_free(ctx);
            }
            for (ggml_backend_buffer_t buf : shadow_bufs) {
                ggml_backend_buffer_free(buf);
            }
            for (struct ggml_context * ctx : per_step_ctxs) {
                ggml_free(ctx);
            }
            for (ggml_backend_buffer_t buf : per_step_bufs) {
                ggml_backend_buffer_free(buf);
            }
        }
    };

    gpu_checkpoint ckpt;

    bool checkpoint_alloc_shadows();
    bool checkpoint_supported() const;
    bool checkpoint_save();
    bool checkpoint_restore();
    void checkpoint_delete();

    // Per-step checkpoint: allocate, restore step k's full state (SSM + conv) to cache
    bool per_step_alloc(int max_tokens);
    bool per_step_restore(int step);

    ~llama_kv_cache() {
        for (struct ggml_context * ctx : ctxs) {
            ggml_free(ctx);
        }
        for (ggml_backend_buffer_t buf : bufs) {
            ggml_backend_buffer_free(buf);
        }
    }
};

struct llama_control_vector {
    std::vector<struct ggml_tensor *> tensors; // per layer
    std::vector<struct ggml_context *> ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    struct ggml_tensor * tensor_for(int il) const {
        if (il < 0 || il < layer_start || il > layer_end || (size_t) il >= tensors.size()) {
            return nullptr;
        }
        return tensors[il];
    }

    struct ggml_tensor * apply_to(struct ggml_context * ctx, struct ggml_tensor * cur, int  il) const {
        ggml_tensor * layer_dir = tensor_for(il);
        if (layer_dir != nullptr) {
            cur = ggml_add(ctx, cur, layer_dir);
        }
        return cur;
    }

    ~llama_control_vector() {
        for (struct ggml_context * ctx : ctxs) {
            ggml_free(ctx);
        }
        for (ggml_backend_buffer_t buf : bufs) {
            ggml_backend_buffer_free(buf);
        }
    }
};

// Forward declarations for the PHASE45 D9.6a back-reference pointers.
// llama_context will be deleted at D9.8 once all fields migrate out;
// these refs delete with it.
struct llama_session;

// PHASE45 D9.6b: include the full llama_decoder definition so we can
// hold a default_decoder member by value (perf counters and other
// migrated fields live on it before any user-created decoder exists).
#include "llama-decoder-internal.h"

struct llama_context {

    llama_context(const llama_model & model);

    ~llama_context();

    const struct llama_model & model;

    // PHASE45 D9.6a: back-references set by llama_session_adopt / _create
    // and llama_decoder_create. Internal helpers that take `lctx` reach
    // the new owning types via these (transitional; the helpers
    // themselves migrate to take session/decoder directly during D9.6b-h).
    struct llama_session * session_ref = nullptr;

    // PHASE45 D9.6b: default decoder owns perf counters etc. before
    // llama_decoder_create runs. ctx ctor points decoder_ref at it;
    // llama_decoder_create reassigns to the user's heap decoder.
    struct llama_decoder   default_decoder;
    struct llama_decoder * decoder_ref = nullptr;

    struct llama_cparams        cparams;
    struct llama_sampling       sampling;
    struct llama_kv_cache       kv_self;
    struct llama_control_vector cvec;

    std::vector<float> scale_data;

    std::unordered_map<struct llama_lora_adapter *, float> lora_adapters;

    std::vector<ggml_backend_t> backends;
#ifdef GGML_USE_METAL
    ggml_backend_t backend_metal = nullptr;
#endif
#ifdef GGML_USE_BLAS
    ggml_backend_t backend_blas = nullptr;
#endif
    ggml_backend_t backend_cpu = nullptr;

    bool has_evaluated_once = false;

    // PHASE45 D9.6b: lifecycle markers stay on ctx (session-shaped data;
    // distinct from per-decoder eval timing).
    int64_t t_start_us;
    int64_t t_load_us;

    // host buffer for the model output (logits and embeddings)
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

    // whether we are computing encoder output or decoder output
    bool is_encoding = false;

    // output of the encoder part of the encoder-decoder models
    std::vector<float> embd_enc;
    std::vector<std::set<llama_seq_id>> seq_ids_enc;

    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;
    ggml_backend_sched_t sched = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    // Phase 36 chain-seed fix: copy-on-set, not pointer-on-set.
    // The previous `const float *` design held a pointer into lctx.embd
    // captured before the next llama_decode. llama_decode's
    // llama_output_reserve repoints lctx.embd based on the new
    // n_outputs_max, leaving the captured pointer aimed at stale or
    // zeroed memory. Copying into a context-owned buffer at set time
    // immunises the fused/per-step chain seed against re-pointing.
    std::vector<float> draft_input_hidden_state_buf;
    const float *      draft_input_hidden_state = nullptr;

    // Phase 36 diagnostic: cycle counter increments on each verify
    // decode (MTP_OP_NONE && cparams.mtp). Threaded into fused/per-step
    // stats prints so cycle-aligned comparisons across paths are
    // possible without ad-hoc line counting.
    int64_t mtp_cycle_counter = 0;

    // MTP DRAFT_GEN device-side residual cache: populated by the embedding-extract
    // path on a CUDA backend (D2D copy from the cgraph's `embd` tensor) so the
    // next iteration's prepare_mtp_graph_inputs can refill inp_mtp_states without
    // the D2H+H2D bounce that ate ~1.5 ms per draft event.
    void *  draft_residual_dev          = nullptr;
    size_t  draft_residual_dev_nbytes   = 0;  // currently-populated payload size
    size_t  draft_residual_dev_capacity = 0;  // allocated buffer size (grows on demand)
    int     draft_residual_dev_device   = -1;
    bool    draft_residual_dev_valid    = false;

    // MTP DRAFT_GEN argmax cache: populated by the on-device CUDA kernel after
    // a DRAFT_GEN forward pass to skip the per-draft logits D2H. Sampler
    // (common_sampler_sample_speculative) checks draft_argmax_valid first.
    // Also reused for the verify-step (MTP_OP_NONE + cparams.mtp) when the
    // caller has set fast_argmax_for_verify and guarantees a trivial sampler.
    bool                  draft_argmax_valid = false;
    int32_t               draft_argmax_n     = 0;
    std::vector<int32_t>  draft_argmax_ids;       // [n_outputs] — top-1 token ids
    std::vector<float>    draft_argmax_probs;     // [n_outputs] — top-1 softmax probabilities

    // Optional top-2 cache (LLAMA_PROBE_TOP2 / tree-K=2 path). Non-empty only
    // when caller arms via llama_arm_draft_top2(). Allocating switches the
    // logits-extract kernel to its top-2 variant (~1.2% MTP tg cost when
    // armed, zero when off — default).
    bool                  draft_top2_armed = false;
    std::vector<int32_t>  draft_argmax_top2_ids;  // [n_outputs] — top-2 token ids

    // Caller-controlled enable for the verify-step argmax-cache fast path.
    // The server sets this true before decode iff the slot's effective sampler
    // is "trivial" (greedy temp=0, no penalties, no grammar, no biases, no
    // rbudget, no DRY/mirostat/XTC/etc.). When true, the verify decode skips
    // the full logits D2H and populates draft_argmax_* the same way DRAFT_GEN
    // does. Auto-cleared at the next decode entry; caller must re-arm per
    // decode.
    bool fast_argmax_for_verify = false;

    // input tensors
    struct ggml_tensor * inp_tokens;      // I32 [n_batch]
    struct ggml_tensor * inp_embd;        // F32 [n_embd, n_batch]
    struct ggml_tensor * inp_pos;         // I32 [n_batch]
    struct ggml_tensor * inp_out_ids;     // I32 [n_outputs]
    struct ggml_tensor * inp_KQ_mask;     // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_KQ_mask_swa; // F32 [kv_size, n_batch]
    struct ggml_tensor * inp_K_shift;     // I32 [kv_size]
    struct ggml_tensor * inp_mean;        // F32 [n_batch, n_batch]
    struct ggml_tensor * inp_cls;         // I32 [n_batch]
    struct ggml_tensor * inp_s_copy;      // I32 [kv_size]
    struct ggml_tensor * inp_s_mask;      // F32 [1, n_kv]
    struct ggml_tensor * inp_s_seq;       // I32 [n_kv, n_batch]
    struct ggml_tensor * inp_s_seq_qnext; // I32 [1, n_batch]
    qnext_state_slot_allocator qnext_slot_alloc; // per-seq slot allocator for the linear-attn recurrent state buffer (s_l[il]); maps llama_seq_id -> slot index
    // Phase 3 instrumentation: count how many times the qwen3next mixed-seq
    // chunking sub-batch was triggered (line ~4756). Should stay at 0 on
    // continuous-batched traffic after the per-seq fill site lands.
    uint64_t qnext_mixed_seq_fallback_count = 0;
    struct ggml_tensor * inp_pos_bucket;    // I32 [n_batch|n_kv, n_batch]
    struct ggml_tensor * inp_embd_enc;      // F32 [n_embd, n_outputs_enc]
    struct ggml_tensor * inp_KQ_mask_cross; // F32 [n_outputs_enc, n_batch]
    struct ggml_tensor * inp_scale = nullptr; // F32 [n_tokens]
    struct ggml_tensor * inp_mtp_states = nullptr;

    // Pre-final-norm residual stream from the main forward graph.
    // Tagged "h_pre_norm" by the qwen35 / qwen35moe builders when
    // cparams.mtp is set. Reset to nullptr at the start of each
    // llama_decode_internal so the per-ubatch hook can detect a
    // fresh build vs a stale pointer. Read via
    // llama_main_graph_h_pre_norm() (Phase 36 Step 3 hook plumbing).
    struct ggml_tensor * t_h_pre_norm = nullptr;

    // Phase 36 Step 1 (fused multi-draft cgraph): graph_compute call
    // count from the most recent llama_mtp_fused_draft_invoke. The
    // fused path expects this to be 1; per-step fallback expects
    // n_steps. Read via llama_mtp_fused_last_compute_count().
    int32_t mtp_fused_last_compute_count = 0;

    // Phase 36 Step 1: per-step results extracted from the fused graph
    // by llama_decode_internal post-compute. Read by
    // llama_mtp_fused_draft_invoke into the user's result struct.
    int32_t mtp_fused_results_n = 0;
    llama_token mtp_fused_results_tokens[8] = {};
    float       mtp_fused_results_probs[8]  = {};

    // Phase 36 Step 1: per-device offset constants for the per-device
    // argmax + reduction path. One i32 per step per device, filled at
    // graph build time and copied via set_inputs into the offsets
    // input tensor for that step.
    int32_t mtp_fused_offset_buf[8 * 16 /*MAX_DEVICES*/] = {};
    // Pointers to the per-step offset input tensors built by
    // build_qwen35_mtp_fused. set_inputs fills them from
    // mtp_fused_offset_buf. Sized [LLAMA_MTP_FUSED_MAX].
    struct ggml_tensor * mtp_fused_offset_t[8] = {};
    int32_t mtp_fused_offset_n_dev[8] = {};

    // Phase 37 #2.1 / Phase 38 B: per-chain-step residual outputs
    // (set_output) of the fused MTP cgraph. Each tensor [n_embd, 1]
    // holds h_pre_norm at position seed_pos + k + 1 (where seed_pos
    // is fused's seed's position). The sched-owned tensors are
    // short-lived (freed on next sched_reset); see mtp_persist_*
    // below for the persistent variant.
    struct ggml_tensor * mtp_fused_chain_residuals[8] = {};

    // Phase 38 E: when true, llama_decode_internal SKIPS the
    // post-compute extraction block (sched_synchronize +
    // tensor_get for argmax/prob + persist capture) for fused
    // decodes. The deferred work is run by
    // llama_mtp_fused_extract_results, called by the server after
    // the parallel verify dispatch completes. Cleared by
    // extract_results.
    bool mtp_fused_skip_extraction = false;

    // Phase 38 E: cached cgraph for the most recent async fused
    // dispatch. Held so extract_results can iterate the graph
    // nodes for argmax/prob/persist values. Set by
    // llama_decode_internal at the end of the fused dispatch when
    // skip_extraction is true. Reset by extract_results on
    // completion.
    struct ggml_cgraph * mtp_fused_pending_gf = nullptr;
    int mtp_fused_pending_n_steps = 0;

    // Phase 38 E (server-internal): the chain_residual_step that
    // was used as seed for the most recent async dispatch. After
    // verify completes and actual n_accepted is known, the
    // speculative path compares this guess to actual: match → use
    // async result; miss → discard + sequential redo. Set by
    // mtp_speculative_gen_draft when it dispatches async; read +
    // reset by the same function on the next cycle.
    int mtp_fused_async_guess = -1;

    // Phase 38 B: persistent chain-residual buffer. Outlives every
    // sched_reset (verify, UPDATE_ACCEPTED, fused-rebuild) so the
    // chain-residual seed plumbing in prepare_mtp_graph_inputs can
    // pull the prior fused decode's residuals D2D without going
    // through the host. Lazy-init on first fused compute (alloc
    // backend buffer + 8 tensors via ggml_backend_alloc_ctx_tensors_from_buft).
    // Populated by D2D copy from the sched-owned chain_residuals at
    // the end of post-compute extraction. persist_n is the count of
    // valid persist tensors (== n_steps of the most recent fused).
    struct ggml_context     * mtp_persist_ctx = nullptr;
    ggml_backend_buffer_t     mtp_persist_buf = nullptr;
    struct ggml_tensor      * mtp_persist[8]  = {};
    int                       mtp_persist_n   = 0;

    // Phase 37 #2.2: when set to a value in [0, n_steps-1], the next
    // MTP_OP_DRAFT_GEN_FUSED decode populates inp_mtp_states via D2D
    // copy from mtp_fused_chain_residuals[pending_chain_residual_step]
    // (the prior fused decode's residual at that chain step) instead
    // of the host-bounce path. Set to -1 by default to keep the host
    // path live as a fallback (e.g. when the prior decode wasn't
    // fused, or when n_accepted == n_steps and no chain residual at
    // that index exists). Read+cleared by prepare_mtp_graph_inputs;
    // resets to -1 every call so callers must re-arm per decode.
    int32_t pending_chain_residual_step = -1;

    // Phase 36 Step 3 (per-ubatch MTP KV hook): observability counters
    // for tests/mtp-ubatch-hook/. mtp_hook_fire_count = number of verify
    // ubatches that ran the kv-only hook; mtp_inline_decode_count =
    // number of MTP-block decodes triggered by the hook (today equals
    // hook fire count since each fire is one inline kv compute).
    uint64_t mtp_hook_fire_count = 0;
    uint64_t mtp_inline_decode_count = 0;

    ggml_backend_t ggml_backend_by_name(const char * name);

    struct Prev;
    std::unique_ptr<Prev> prev;

    void reset_scheduler();
    bool can_reuse_graph(const llama_batch & u_batch);

    struct CacheCopy {
        ggml_tensor * cpy = nullptr;
        size_t        step = 0;
    };
    std::vector<CacheCopy> cache_copies;

    bool update_cache_copies();

    bool prepare_mtp_graph_inputs(
        struct llama_context & lctx);
    void set_mtp_op_type(llama_mtp_op_type value);

};
