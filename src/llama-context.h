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
    // PHASE45 D9.6g: renamed from `kv_self` to reflect that the
    // K/V cache is the transformer's session-shared storage.
    // Architecturally session-owned; storage moves to llama_session
    // at D9.8 when llama_context deletes.
    struct llama_kv_cache       transformer_kv;
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

    // PHASE45 D9.6c: output buffers (logits/embd/output_ids/...) moved
    // to llama_decoder.

    // whether we are computing encoder output or decoder output
    bool is_encoding = false;

    // output of the encoder part of the encoder-decoder models
    std::vector<float> embd_enc;
    std::vector<std::set<llama_seq_id>> seq_ids_enc;

    // PHASE45 D9.6e: scheduler + compute-meta + abort callback moved to
    // llama_decoder (held by default_decoder; shared across all decoders
    // until per-decoder graph reservation lands).

    // PHASE45 D9.6f: encoder-output input tensor stays on ctx (the rest
    // of the inp_* tensors are graph inputs and live on llama_decoder).
    struct ggml_tensor * inp_embd_enc = nullptr; // F32 [n_embd, n_outputs_enc]

    // PHASE45 D9.6f: all MTP / draft / fused / persist / qnext / inp_*
    // (except inp_embd_enc) state moved to llama_decoder. The
    // ctx-owned default_decoder holds these fields between ctx ctor
    // and llama_decoder_create's swap to the user's decoder.

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
