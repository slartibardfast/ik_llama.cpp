//
// PHASE 45 D6: llama_session implementation (Option A wrapper).
//
// A `llama_session` wraps an internal `llama_context`. Two construction
// paths:
//   1. llama_session_create(model, params) — creates a fresh context via
//      llama_init_from_model, owns it, frees it in llama_session_free.
//   2. llama_session_adopt(ctx) — wraps an externally-created context;
//      session_free does NOT free the ctx (caller retains ownership).
//      This is the bridge used while common/`llama_init_from_gpt_params`
//      is still the canonical context factory; deletes at D10.
//
// All KV-cache, state, lora, and control-vector ops forward to existing
// `llama_*(ctx, ...)` entry points. By construction the byte-output of
// any session-mediated forward matches the old API exactly.
//

#include "llama-session.h"
#include "llama.h"

#include <cstring>

struct llama_session {
    struct llama_context * ctx       = nullptr;
    bool                   owns_ctx  = false;
    struct llama_session_params params{};
};

extern "C" {

// Internal accessor: exposes the wrapped llama_context to llama-decoder.cpp.
// Not in the public header; deletes at D10.
struct llama_context * llama_session_internal_context(struct llama_session * session);

struct llama_session_params llama_session_default_params(void) {
    llama_session_params p{};
    p.n_ctx                       = 0;
    p.n_seq_max                   = 1;
    p.n_batch                     = 2048;
    p.n_ubatch                    = 512;
    p.type_k                      = GGML_TYPE_F16;
    p.type_v                      = GGML_TYPE_F16;
    p.rope_freq_base              = 0.0f;
    p.rope_freq_scale             = 0.0f;
    p.n_ctx_orig_yarn             = 0;
    p.yarn_ext_factor             = -1.0f;
    p.yarn_attn_factor            = 1.0f;
    p.yarn_beta_fast              = 32.0f;
    p.yarn_beta_slow              = 1.0f;
    p.defrag_thold                = -1.0f;
    p.k_cache_hadamard            = false;
    p.v_cache_hadamard            = false;
    p.mla_attn                    = 0;
    p.split_mode_graph_scheduling = false;
    p.flash_attn                  = false;
    p.offload_kqv                 = true;
    return p;
}

struct llama_session * llama_session_create(struct llama_model * model, struct llama_session_params params) {
    if (model == nullptr) return nullptr;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                       = params.n_ctx;
    cparams.n_seq_max                   = params.n_seq_max;
    cparams.n_batch                     = params.n_batch;
    cparams.n_ubatch                    = params.n_ubatch;
    cparams.type_k                      = params.type_k;
    cparams.type_v                      = params.type_v;
    cparams.rope_freq_base              = params.rope_freq_base;
    cparams.rope_freq_scale             = params.rope_freq_scale;
    cparams.yarn_orig_ctx               = params.n_ctx_orig_yarn;
    cparams.yarn_ext_factor             = params.yarn_ext_factor;
    cparams.yarn_attn_factor            = params.yarn_attn_factor;
    cparams.yarn_beta_fast              = params.yarn_beta_fast;
    cparams.yarn_beta_slow              = params.yarn_beta_slow;
    cparams.defrag_thold                = params.defrag_thold;
    cparams.k_cache_hadamard            = params.k_cache_hadamard;
    cparams.v_cache_hadamard            = params.v_cache_hadamard;
    cparams.mla_attn                    = params.mla_attn;
    cparams.split_mode_graph_scheduling = params.split_mode_graph_scheduling;
    cparams.flash_attn                  = params.flash_attn;
    cparams.offload_kqv                 = params.offload_kqv;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) return nullptr;

    auto * s = new llama_session;
    s->ctx      = ctx;
    s->owns_ctx = true;
    s->params   = params;
    return s;
}

struct llama_session * llama_session_adopt(struct llama_context * ctx) {
    if (ctx == nullptr) return nullptr;
    auto * s = new llama_session;
    s->ctx                  = ctx;
    s->owns_ctx             = false;
    s->params.n_ctx         = llama_n_ctx(ctx);
    s->params.n_seq_max     = llama_n_seq_max(ctx);
    s->params.n_batch       = llama_n_batch(ctx);
    s->params.n_ubatch      = llama_n_ubatch(ctx);
    return s;
}

void llama_session_free(struct llama_session * session) {
    if (session == nullptr) return;
    if (session->owns_ctx && session->ctx != nullptr) {
        llama_free(session->ctx);
    }
    delete session;
}

const struct llama_model * llama_session_model    (const struct llama_session * session) { return session ? llama_get_model(session->ctx)    : nullptr; }
uint32_t                   llama_session_n_ctx    (const struct llama_session * session) { return session ? llama_n_ctx    (session->ctx)    : 0; }
uint32_t                   llama_session_n_seq_max(const struct llama_session * session) { return session ? llama_n_seq_max(session->ctx)    : 0; }
uint32_t                   llama_session_n_batch  (const struct llama_session * session) { return session ? llama_n_batch  (session->ctx)    : 0; }
uint32_t                   llama_session_n_ubatch (const struct llama_session * session) { return session ? llama_n_ubatch (session->ctx)    : 0; }

void llama_session_kv_clear(struct llama_session * session) {
    if (session) llama_kv_cache_clear(session->ctx);
}
bool llama_session_kv_seq_rm(struct llama_session * session, llama_seq_id seq, llama_pos p0, llama_pos p1) {
    return session ? llama_kv_cache_seq_rm(session->ctx, seq, p0, p1) : false;
}
void llama_session_kv_seq_cp(struct llama_session * session, llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) {
    if (session) llama_kv_cache_seq_cp(session->ctx, src, dst, p0, p1);
}
void llama_session_kv_seq_keep(struct llama_session * session, llama_seq_id seq) {
    if (session) llama_kv_cache_seq_keep(session->ctx, seq);
}
void llama_session_kv_seq_add(struct llama_session * session, llama_seq_id seq, llama_pos p0, llama_pos p1, llama_pos delta) {
    if (session) llama_kv_cache_seq_add(session->ctx, seq, p0, p1, delta);
}
void llama_session_kv_seq_div(struct llama_session * session, llama_seq_id seq, llama_pos p0, llama_pos p1, int d) {
    if (session) llama_kv_cache_seq_div(session->ctx, seq, p0, p1, d);
}
llama_pos llama_session_kv_seq_pos_max(struct llama_session * session, llama_seq_id seq) {
    return session ? llama_kv_cache_seq_pos_max(session->ctx, seq) : -1;
}

void llama_session_kv_update(struct llama_session * session) {
    if (session) llama_kv_cache_update(session->ctx);
}
void llama_session_kv_defrag(struct llama_session * session) {
    if (session) llama_kv_cache_defrag(session->ctx);
}

size_t llama_session_state_get_size(struct llama_session * session) {
    return session ? llama_state_get_size(session->ctx) : 0;
}
size_t llama_session_state_get_data(struct llama_session * session, uint8_t * dst, size_t size) {
    return session ? llama_state_get_data(session->ctx, dst, size) : 0;
}
size_t llama_session_state_set_data(struct llama_session * session, const uint8_t * src, size_t size) {
    return session ? llama_state_set_data(session->ctx, src, size) : 0;
}

size_t llama_session_state_seq_get_size(struct llama_session * session, llama_seq_id seq) {
    return session ? llama_state_seq_get_size(session->ctx, seq, /*flags=*/0) : 0;
}
size_t llama_session_state_seq_get_data(struct llama_session * session, uint8_t * dst, size_t size, llama_seq_id seq) {
    return session ? llama_state_seq_get_data(session->ctx, dst, size, seq, /*flags=*/0) : 0;
}
size_t llama_session_state_seq_set_data(struct llama_session * session, const uint8_t * src, size_t size, llama_seq_id seq) {
    return session ? llama_state_seq_set_data(session->ctx, src, size, seq, /*flags=*/0) : 0;
}

int32_t llama_session_lora_adapter_set(struct llama_session * session, struct llama_lora_adapter * adapter, float scale) {
    return session ? llama_lora_adapter_set(session->ctx, adapter, scale) : -1;
}
int32_t llama_session_lora_adapter_remove(struct llama_session * session, struct llama_lora_adapter * adapter) {
    return session ? llama_lora_adapter_remove(session->ctx, adapter) : -1;
}
void llama_session_lora_adapter_clear(struct llama_session * session) {
    if (session) llama_lora_adapter_clear(session->ctx);
}

int32_t llama_session_control_vector_apply(
        struct llama_session * session,
        const float          * data,
        size_t                 len,
        int32_t                n_embd,
        int32_t                il_start,
        int32_t                il_end) {
    return session ? llama_control_vector_apply(session->ctx, data, len, n_embd, il_start, il_end) : -1;
}

// Internal-API: expose the underlying ctx pointer to the decoder. Not in
// the public header. D10 will delete this entirely.
struct llama_context * llama_session_internal_context(struct llama_session * session) {
    return session ? session->ctx : nullptr;
}

} // extern "C"
