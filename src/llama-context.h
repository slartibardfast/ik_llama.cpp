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

    // Per-stream allocator state for the n_stream KV layout
    // (PHASE_NSTREAM_KV_4D.md). K/V tensors are 4D
    // [head_dim, kv_size_per_stream, n_head_kv, n_stream].
    //
    // Each stream owns the contiguous range
    //   [s * kv_size_per_stream, (s + 1) * kv_size_per_stream)
    // within `cells`. `v_heads[s]` is the per-stream next-free cursor
    // expressed as a STREAM-LOCAL index in [0, kv_size_per_stream); the
    // flat (global) index is `s * kv_size_per_stream + v_heads[s]`.
    //
    // At n_stream == 1, the layout collapses to the legacy single-arena
    // model byte-for-byte: kv_size_per_stream == kv_size, v_heads[0]
    // mirrors `head`, K/V's ne[3] == 1 and ne[1] == kv_size so views
    // into 4D coincide with views into the legacy 2D shape.
    uint32_t n_stream            = 1;
    uint32_t kv_size_per_stream  = 0;
    std::vector<uint32_t>        v_heads;

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

// ─────────────────────────────────────────────────────────────────────
// PHASE45 D9.8 (PENDING) — `llama_context` is going away.
//
// All decoder-runtime fields (perf counters, output buffers, recurrent
// state, scheduler, MTP/draft/inp_*) extracted to `llama_decoder` at
// D9.6.b–h. The remaining fields below are architecturally session-
// scoped and will migrate to `llama_session` at D9.8. Until D9.8 lands:
//
//   ❌  NEW CODE MUST NOT add `lctx.field` reads or writes for the
//       fields listed below. Use the session API surface
//       (`llama_session_*`) or pass a `llama_session *` directly.
//   ✅  Existing `lctx.field` callsites stay until D9.8 ports them
//       en bloc; they're already counted in the ~365-callsite migration.
//
// Field → D9.8 migration target:
//   cparams, sampling             → session
//   transformer_kv (was kv_self)  → session.transformer_kv
//   cvec                          → session.cvec
//   scale_data                    → session.scale_data
//   lora_adapters                 → session.lora_adapters
//   backends, backend_{cpu,metal,blas} → session.backends
//   has_evaluated_once            → session
//   t_start_us, t_load_us         → session lifecycle markers
//   embd_enc, seq_ids_enc         → session
//   inp_embd_enc                  → session (encoder-only graph input)
//   prev (Prev), cache_copies     → session graph-cache state
//
// `session_ref` and `default_decoder` (added at D9.6a/b) are scaffolding
// and delete with the struct at D9.8. `decoder_ref` does too — its
// callers move to taking `llama_decoder *` directly.
// ─────────────────────────────────────────────────────────────────────
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

    // PHASE45 D9.6h: scale_data, lora_adapters, backends, backend_cpu,
    // backend_metal, backend_blas, embd_enc, seq_ids_enc are
    // architecturally session-owned. llama_init_from_model still
    // populates them on ctx today; llama_session_adopt() moves them
    // into session storage at adopt time. ctx fields are kept until
    // D9.8 to keep llama_init_from_model's allocation paths coherent.
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

    // PHASE45 D9.6h: encoder-output buffers — populated by ctx-level
    // encode path during llama_init_from_model and llama_encode; mirrored
    // onto llama_session at adopt time so consumer access via session
    // works without further ctx coupling. Kept on ctx today for the
    // legacy mtp graph-build path that reads lctx.embd_enc directly.
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

    // PHASE_NSTREAM_KV_PERF Tier 2: per-stream K/V READ view registry.
    //
    // Companion to specs/kv-cache/per_stream_read_view_patching.allium.
    // Parallels cache_copies (which patches the WRITE-side CPY nodes
    // per stream every tick). The READ views are the K/V slices in
    // llm_build_kqv that flash-attention / per-slot-kv / mul_mat read
    // from. At graph-build time these are constructed with stream_id=0
    // so the captured cuda graph's view->data points at the K/V base;
    // update_cache_copies() rewrites view->data per-tick to the current
    // stream's slice.
    //
    // Indexing convention: cache_read_views[2*il + 0] is the K read
    // view at layer il; cache_read_views[2*il + 1] is the V read view.
    // Under split-mode (ATTN/GRAPH), entries are per device: index
    // is 2*splits.size()*il + 2*device + 0/1.
    //
    // Sized in the llama_context constructor (same layout as
    // cache_copies). Populated by llm_build_kqv during graph build.
    struct CacheReadView {
        ggml_tensor * view = nullptr;
    };
    std::vector<CacheReadView> cache_read_views;

    bool update_cache_copies();

    bool prepare_mtp_graph_inputs(
        struct llama_context & lctx);
    void set_mtp_op_type(llama_mtp_op_type value);

    // DFlash drafter binding. Set by llama_set_dflash, cleared by
    // llama_free. Type is opaque from this header (forward-declared in
    // llama-dflash.cpp).
    // Spec: specs/dflash/dflash.allium.
    struct llama_dflash_ctx_state * dflash_state = nullptr;
};
