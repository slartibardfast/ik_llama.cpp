//
// PHASE 45 D6 stub: llama_decoder implementation skeleton.
//
// Each decoder borrows a session pointer (no ownership) and parameterizes
// its execution by `role`. PRIMARY decoders cover today's standalone
// forward; VERIFY/DRAFT_MTP/TREE_BRANCH activate in D7-D9.
//
// D6 verifier (greedy decode through main.cpp via the new API) needs
// PRIMARY-role decode/get_logits to forward into the session's internal
// llama_context. That delegation lands in the next iteration.
//

#include "llama-decoder.h"
#include "llama-session.h"

#include <cstdlib>
#include <cstdio>

struct llama_decoder {
    struct llama_session    * session = nullptr;
    struct llama_decoder_params params;
    enum   llama_decoder_role role    = LLAMA_DECODER_PRIMARY;
};

[[noreturn]] static void decoder_unimplemented(const char * fn) {
    std::fprintf(stderr,
        "PHASE 45 D6 skeleton: %s body is the next iteration's work. "
        "Aborting to surface unintended use.\n", fn);
    std::abort();
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
}
void llama_decoder_set_causal    (struct llama_decoder * decoder, bool causal_attn) { if (decoder) decoder->params.causal_attn = causal_attn; }
void llama_decoder_set_embeddings(struct llama_decoder * decoder, bool embeddings)  { if (decoder) decoder->params.embeddings  = embeddings;  }
void llama_decoder_set_warmup    (struct llama_decoder * /*decoder*/, bool /*warmup*/) {
    // Warmup flag is consumed by the underlying llama_context at create-time
    // in the existing implementation; D6 forwarding lives in the next iteration.
}

int32_t llama_decoder_decode(struct llama_decoder * /*decoder*/, struct llama_batch /*batch*/) { decoder_unimplemented(__func__); }
int32_t llama_decoder_encode(struct llama_decoder * /*decoder*/, struct llama_batch /*batch*/) { decoder_unimplemented(__func__); }

void llama_decoder_synchronize(struct llama_decoder * /*decoder*/) { decoder_unimplemented(__func__); }

float * llama_decoder_get_logits        (struct llama_decoder * /*decoder*/) { decoder_unimplemented(__func__); }
float * llama_decoder_get_logits_ith    (struct llama_decoder * /*decoder*/, int32_t /*i*/) { decoder_unimplemented(__func__); }
float * llama_decoder_get_embeddings    (struct llama_decoder * /*decoder*/) { decoder_unimplemented(__func__); }
float * llama_decoder_get_embeddings_ith(struct llama_decoder * /*decoder*/, int32_t /*i*/) { decoder_unimplemented(__func__); }
float * llama_decoder_get_embeddings_seq(struct llama_decoder * /*decoder*/, llama_seq_id /*seq_id*/) { decoder_unimplemented(__func__); }

struct llama_timings llama_decoder_timings(const struct llama_decoder * /*decoder*/) {
    llama_timings data{};
    return data;
}
void llama_decoder_perf_reset(struct llama_decoder * /*decoder*/) { /* no-op until D6 wires perf forwarding */ }
void llama_decoder_perf_print(const struct llama_decoder * /*decoder*/) { /* no-op until D6 wires perf forwarding */ }

} // extern "C"
