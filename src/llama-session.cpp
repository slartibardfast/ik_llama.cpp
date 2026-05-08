//
// PHASE 45 D6 stub: llama_session implementation skeleton.
//
// D6 verifier is byte-identical greedy decode through main.cpp routed
// over the new API. To bind, llama_session_create must construct an
// internal llama_context (the existing implementation), and KV ops must
// forward to it. This iteration commits the skeleton; the next iteration
// wires the llama_context delegation.
//
// Bodies that aren't on the D6 forward path abort() so any unintended
// use surfaces immediately.
//

#include "llama-session.h"

#include <cstdlib>
#include <cstdio>

struct llama_session {
    struct llama_model *      model       = nullptr;
    struct llama_context *    ctx         = nullptr;  // delegate (filled in next D6 iteration)
    struct llama_session_params params;
};

[[noreturn]] static void session_unimplemented(const char * fn) {
    std::fprintf(stderr,
        "PHASE 45 D6 skeleton: %s body is the next iteration's work. "
        "Aborting to surface unintended use.\n", fn);
    std::abort();
}

extern "C" {

struct llama_session_params llama_session_default_params(void) {
    llama_session_params p{};
    p.n_ctx          = 0;
    p.n_seq_max      = 1;
    p.n_batch        = 512;
    p.n_ubatch       = 512;
    p.type_k         = GGML_TYPE_F16;
    p.type_v         = GGML_TYPE_F16;
    p.rope_freq_base = 0.0f;
    p.rope_freq_scale= 0.0f;
    p.n_ctx_orig_yarn= 0;
    p.yarn_ext_factor= -1.0f;
    p.yarn_attn_factor= 1.0f;
    p.yarn_beta_fast = 32.0f;
    p.yarn_beta_slow = 1.0f;
    p.defrag_thold   = -1.0f;
    p.k_cache_hadamard = false;
    p.v_cache_hadamard = false;
    p.mla_attn       = 0;
    p.split_mode_graph_scheduling = false;
    p.flash_attn     = false;
    p.offload_kqv    = true;
    return p;
}

struct llama_session * llama_session_create(struct llama_model * /*model*/, struct llama_session_params /*params*/) {
    session_unimplemented(__func__);
}

void llama_session_free(struct llama_session * session) {
    delete session;
}

const struct llama_model * llama_session_model    (const struct llama_session * session) { return session ? session->model : nullptr; }
uint32_t                   llama_session_n_ctx    (const struct llama_session * session) { return session ? session->params.n_ctx     : 0; }
uint32_t                   llama_session_n_seq_max(const struct llama_session * session) { return session ? session->params.n_seq_max : 0; }
uint32_t                   llama_session_n_batch  (const struct llama_session * session) { return session ? session->params.n_batch   : 0; }
uint32_t                   llama_session_n_ubatch (const struct llama_session * session) { return session ? session->params.n_ubatch  : 0; }

void llama_session_kv_clear   (struct llama_session * /*session*/) { session_unimplemented(__func__); }
bool llama_session_kv_seq_rm  (struct llama_session * /*session*/, llama_seq_id /*seq*/, llama_pos /*p0*/, llama_pos /*p1*/) { session_unimplemented(__func__); }
void llama_session_kv_seq_cp  (struct llama_session * /*session*/, llama_seq_id /*src*/, llama_seq_id /*dst*/, llama_pos /*p0*/, llama_pos /*p1*/) { session_unimplemented(__func__); }
void llama_session_kv_seq_keep(struct llama_session * /*session*/, llama_seq_id /*seq*/) { session_unimplemented(__func__); }
void llama_session_kv_seq_add (struct llama_session * /*session*/, llama_seq_id /*seq*/, llama_pos /*p0*/, llama_pos /*p1*/, llama_pos /*delta*/) { session_unimplemented(__func__); }
void llama_session_kv_seq_div (struct llama_session * /*session*/, llama_seq_id /*seq*/, llama_pos /*p0*/, llama_pos /*p1*/, int /*d*/) { session_unimplemented(__func__); }
llama_pos llama_session_kv_seq_pos_max(struct llama_session * /*session*/, llama_seq_id /*seq*/) { session_unimplemented(__func__); }

void llama_session_kv_update(struct llama_session * /*session*/) { session_unimplemented(__func__); }
void llama_session_kv_defrag(struct llama_session * /*session*/) { session_unimplemented(__func__); }

size_t llama_session_state_get_size(struct llama_session * /*session*/) { session_unimplemented(__func__); }
size_t llama_session_state_get_data(struct llama_session * /*session*/, uint8_t * /*dst*/, size_t /*size*/) { session_unimplemented(__func__); }
size_t llama_session_state_set_data(struct llama_session * /*session*/, const uint8_t * /*src*/, size_t /*size*/) { session_unimplemented(__func__); }

size_t llama_session_state_seq_get_size(struct llama_session * /*session*/, llama_seq_id /*seq*/) { session_unimplemented(__func__); }
size_t llama_session_state_seq_get_data(struct llama_session * /*session*/, uint8_t * /*dst*/, size_t /*size*/, llama_seq_id /*seq*/) { session_unimplemented(__func__); }
size_t llama_session_state_seq_set_data(struct llama_session * /*session*/, const uint8_t * /*src*/, size_t /*size*/, llama_seq_id /*seq*/) { session_unimplemented(__func__); }

int32_t llama_session_lora_adapter_set   (struct llama_session * /*session*/, struct llama_lora_adapter * /*adapter*/, float /*scale*/) { session_unimplemented(__func__); }
int32_t llama_session_lora_adapter_remove(struct llama_session * /*session*/, struct llama_lora_adapter * /*adapter*/) { session_unimplemented(__func__); }
void    llama_session_lora_adapter_clear (struct llama_session * /*session*/) { session_unimplemented(__func__); }

int32_t llama_session_control_vector_apply(
        struct llama_session * /*session*/,
        const float          * /*data*/,
        size_t                 /*len*/,
        int32_t                /*n_embd*/,
        int32_t                /*il_start*/,
        int32_t                /*il_end*/) { session_unimplemented(__func__); }

} // extern "C"
