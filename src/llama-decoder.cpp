//
// PHASE 45 D6: llama_decoder implementation (Option A wrapper).
//
// A `llama_decoder` borrows a session pointer and parameterizes execution
// by `role`. For D6 PRIMARY-role single-decoder usage, decode/get_logits/
// timings forward unchanged into the session's internal llama_context;
// behavior is byte-identical to old API by construction.
//
// Per-decoder state that today lives on llama_context (n_threads, causal,
// embeddings, mtp_op_type) is applied at decoder_create / decoder_decode
// time via existing setters. D7+ will add per-decoder-private storage so
// VERIFY and DRAFT_MTP can coexist on a single shared session.
//

#include "llama-decoder.h"
#include "llama-session.h"
#include "llama-session-internal.h"
#include "llama-context.h"  // PHASE45 D9.6a: set ctx->decoder_ref
#include "llama.h"

struct llama_decoder {
    struct llama_session    * session = nullptr;
    struct llama_decoder_params params{};
    enum   llama_decoder_role role    = LLAMA_DECODER_PRIMARY;
};

static void apply_decoder_params_to_ctx(struct llama_context * ctx, const struct llama_decoder_params & p) {
    if (ctx == nullptr) return;
    llama_set_n_threads (ctx, p.n_threads, p.n_threads_batch);
    llama_set_causal_attn(ctx, p.causal_attn);
    llama_set_embeddings (ctx, p.embeddings);
}

extern "C" {

struct llama_decoder_params llama_decoder_default_params(enum llama_decoder_role role) {
    llama_decoder_params p{};
    p.role                   = role;
    p.n_threads              = 4;
    p.n_threads_batch        = 4;
    p.causal_attn            = true;
    p.embeddings             = false;
    p.fused_moe_up_gate      = true;
    p.grouped_expert_routing = false;
    p.fused_up_gate          = true;
    p.fused_mmad             = true;
    p.rope_cache             = false;
    p.graph_reuse            = true;
    p.scheduler_async        = false;
    p.min_experts            = 0;
    p.thresh_experts         = 0.0f;
    p.mtp_fused_n_steps      = 0;
    p.mtp_fused_n_extend     = 0;
    p.mtp_inline_kv_hook     = false;
    p.cb_eval                = nullptr;
    p.cb_eval_user_data      = nullptr;
    p.tree_branch_id         = -1;
    p.tree_branch_min_prob   = 0.0f;
    return p;
}

struct llama_decoder * llama_decoder_create(struct llama_session * session, struct llama_decoder_params params) {
    if (session == nullptr) return nullptr;
    auto * d = new llama_decoder;
    d->session = session;
    d->params  = params;
    d->role    = params.role;

    llama_context * ctx = llama_session_internal_context(session);
    apply_decoder_params_to_ctx(ctx, params);
    if (ctx) ctx->decoder_ref = d;  // PHASE45 D9.6a: back-ref
    return d;
}

void llama_decoder_free(struct llama_decoder * decoder) {
    delete decoder;
}

enum llama_decoder_role llama_decoder_role_of(const struct llama_decoder * decoder) {
    return decoder ? decoder->role : LLAMA_DECODER_PRIMARY;
}

struct llama_session * llama_decoder_session(const struct llama_decoder * decoder) {
    return decoder ? decoder->session : nullptr;
}

const struct llama_model * llama_decoder_model(const struct llama_decoder * decoder) {
    if (!decoder || !decoder->session) return nullptr;
    return llama_session_model(decoder->session);
}

void llama_decoder_set_n_threads(struct llama_decoder * decoder, uint32_t n_threads, uint32_t n_threads_batch) {
    if (!decoder) return;
    decoder->params.n_threads       = n_threads;
    decoder->params.n_threads_batch = n_threads_batch;
    llama_set_n_threads(llama_session_internal_context(decoder->session), n_threads, n_threads_batch);
}
void llama_decoder_set_causal(struct llama_decoder * decoder, bool causal_attn) {
    if (!decoder) return;
    decoder->params.causal_attn = causal_attn;
    llama_set_causal_attn(llama_session_internal_context(decoder->session), causal_attn);
}
void llama_decoder_set_embeddings(struct llama_decoder * decoder, bool embeddings) {
    if (!decoder) return;
    decoder->params.embeddings = embeddings;
    llama_set_embeddings(llama_session_internal_context(decoder->session), embeddings);
}
void llama_decoder_set_warmup(struct llama_decoder * /*decoder*/, bool /*warmup*/) {
    // Warmup is consumed at ctx-create time by the existing factory; no
    // post-create runtime knob today. D7+ will expose if needed.
}

void llama_decoder_set_fast_argmax(struct llama_decoder * decoder, bool enabled) {
    if (!decoder || !decoder->session) return;
    llama_set_fast_argmax_for_verify(llama_session_internal_context(decoder->session), enabled);
}

int llama_decoder_spec_ckpt_init(struct llama_decoder * decoder, int mode, int max_tokens) {
    if (!decoder || !decoder->session) return 0;
    return llama_spec_ckpt_init(llama_session_internal_context(decoder->session), mode, max_tokens);
}
bool llama_decoder_spec_ckpt_save(struct llama_decoder * decoder, llama_seq_id seq_id) {
    if (!decoder || !decoder->session) return false;
    return llama_spec_ckpt_save(llama_session_internal_context(decoder->session), seq_id);
}
bool llama_decoder_spec_ckpt_restore(struct llama_decoder * decoder, llama_seq_id seq_id, llama_pos n_past, int step) {
    if (!decoder || !decoder->session) return false;
    return llama_spec_ckpt_restore(llama_session_internal_context(decoder->session), seq_id, n_past, step);
}
void llama_decoder_spec_ckpt_discard(struct llama_decoder * decoder) {
    if (!decoder || !decoder->session) return;
    llama_spec_ckpt_discard(llama_session_internal_context(decoder->session));
}

int32_t llama_decoder_decode(struct llama_decoder * decoder, struct llama_batch batch) {
    if (!decoder || !decoder->session) return -1;
    apply_decoder_params_to_ctx(llama_session_internal_context(decoder->session), decoder->params);
    return llama_decode(llama_session_internal_context(decoder->session), batch);
}

int32_t llama_decoder_encode(struct llama_decoder * decoder, struct llama_batch batch) {
    if (!decoder || !decoder->session) return -1;
    apply_decoder_params_to_ctx(llama_session_internal_context(decoder->session), decoder->params);
    return llama_encode(llama_session_internal_context(decoder->session), batch);
}

void llama_decoder_synchronize(struct llama_decoder * decoder) {
    if (!decoder || !decoder->session) return;
    llama_synchronize(llama_session_internal_context(decoder->session));
}

float * llama_decoder_get_logits(struct llama_decoder * decoder) {
    return decoder && decoder->session ? llama_get_logits(llama_session_internal_context(decoder->session)) : nullptr;
}
float * llama_decoder_get_logits_ith(struct llama_decoder * decoder, int32_t i) {
    return decoder && decoder->session ? llama_get_logits_ith(llama_session_internal_context(decoder->session), i) : nullptr;
}
float * llama_decoder_get_embeddings(struct llama_decoder * decoder) {
    return decoder && decoder->session ? llama_get_embeddings(llama_session_internal_context(decoder->session)) : nullptr;
}
float * llama_decoder_get_embeddings_ith(struct llama_decoder * decoder, int32_t i) {
    return decoder && decoder->session ? llama_get_embeddings_ith(llama_session_internal_context(decoder->session), i) : nullptr;
}
float * llama_decoder_get_embeddings_seq(struct llama_decoder * decoder, llama_seq_id seq_id) {
    return decoder && decoder->session ? llama_get_embeddings_seq(llama_session_internal_context(decoder->session), seq_id) : nullptr;
}

struct llama_timings llama_decoder_timings(const struct llama_decoder * decoder) {
    if (!decoder || !decoder->session) {
        llama_timings z{};
        return z;
    }
    return llama_get_timings(llama_session_internal_context(decoder->session));
}
void llama_decoder_perf_reset(struct llama_decoder * decoder) {
    if (!decoder || !decoder->session) return;
    llama_reset_timings(llama_session_internal_context(decoder->session));
}
void llama_decoder_perf_print(const struct llama_decoder * decoder) {
    if (!decoder || !decoder->session) return;
    llama_print_timings(llama_session_internal_context(decoder->session));
}

} // extern "C"
