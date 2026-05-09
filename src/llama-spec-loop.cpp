//
// PHASE 45 D6 stub: llama_spec_loop implementation skeleton.
//
// D6 verifier is `main.cpp` greedy decode — does NOT exercise spec_loop.
// Bodies are minimal placeholders that abort() if called, so any
// inadvertent use during D6/D7 fails loud.
// D8 fills in by porting common/speculative.cpp into this skeleton.
//

#include "llama-spec-loop.h"
#include "llama-decoder.h"
#include "llama-spec.h"
#include "llama-session.h"
#include "llama-session-internal.h"  // PHASE45 D10.b: ctx-equality check via session

#include <cstdlib>
#include <cstdio>
#include <vector>

struct llama_spec_loop {
    struct llama_decoder              * verify   = nullptr;
    std::vector<struct llama_decoder *> drafts;
    struct llama_spec_loop_params       params;

    int64_t                  n_drafted      = 0;
    int64_t                  n_accepted     = 0;
    int64_t                  n_verify_steps = 0;
    std::vector<llama_token> last_accepted;
};

static int32_t spec_loop_step_unimplemented(const char * fn) {
    std::fprintf(stderr,
        "%s: vertical spec_loop API has no implementation. "
        "Use llama_spec_loop_gen_drafts + llama_spec_loop_accept_n. "
        "Returning -1.\n", fn);
    return -1;
}

extern "C" {

struct llama_spec_loop_params llama_spec_loop_default_params(void) {
    llama_spec_loop_params p{};
    p.min_chain_prob  = 0.0f;
    p.max_draft_depth = 3;
    p.sampler         = nullptr;
    return p;
}

struct llama_spec_loop * llama_spec_loop_create(
        struct llama_decoder        * verify,
        struct llama_decoder       ** drafts,
        int                           n_drafts,
        struct llama_spec_loop_params params) {
    if (verify == nullptr || (n_drafts > 0 && drafts == nullptr)) return nullptr;
    auto * loop = new llama_spec_loop;
    loop->verify = verify;
    loop->drafts.assign(drafts, drafts + n_drafts);
    loop->params = params;
    return loop;
}

void llama_spec_loop_free(struct llama_spec_loop * loop) {
    delete loop;
}

int32_t llama_spec_loop_gen_drafts(
        struct llama_spec_loop * loop,
        llama_token              id_last,
        float                    p_min,
        int32_t                  n_draft_max,
        llama_seq_id             seq_id,
        llama_pos                n_past,
        llama_token            * drafts_out) {
    if (loop == nullptr || loop->verify == nullptr) return -1;
    if (loop->drafts.empty())                       return -1;
    if (drafts_out == nullptr || n_draft_max <= 0)  return -1;

    const int32_t n = llama_spec_mtp_draft(
            loop->verify,
            loop->drafts[0],
            id_last,
            p_min,
            n_draft_max,
            seq_id,
            n_past,
            drafts_out);

    if (n > 0) {
        loop->n_drafted += n;
        loop->last_accepted.assign(drafts_out, drafts_out + n);
    } else {
        loop->last_accepted.clear();
    }
    return n;
}

void llama_spec_loop_accept_n(struct llama_spec_loop * loop, int32_t n_accepted) {
    if (loop == nullptr || n_accepted <= 0) return;
    loop->n_accepted     += n_accepted;
    loop->n_verify_steps += 1;
    if ((size_t) n_accepted < loop->last_accepted.size()) {
        loop->last_accepted.resize(n_accepted);
    }
}

int32_t llama_spec_loop_gen_drafts_batched(
        struct llama_spec_loop      ** loops,
        int32_t                        n_loops,
        const llama_spec_mtp_slot_in * slots,
        float                          p_min,
        llama_token                  * drafts_out,
        int32_t                        drafts_out_stride,
        llama_spec_mtp_slot_out      * outs) {
    if (loops == nullptr || n_loops <= 0 || slots == nullptr) return -1;
    if (drafts_out == nullptr || outs == nullptr || drafts_out_stride <= 0) return -1;

    // All loops must share the underlying llama_context (post-D9.5
    // invariant — each slot wraps the same shared ctx_tgt via its own
    // session/decoder pair). The ctx check is what binds; the chosen
    // decoder pair (loops[0]'s) forwards to that single ctx.
    struct llama_decoder * verify = loops[0] ? loops[0]->verify : nullptr;
    struct llama_decoder * draft  = (loops[0] && !loops[0]->drafts.empty()) ? loops[0]->drafts[0] : nullptr;
    if (verify == nullptr || draft == nullptr) return -1;
    struct llama_context * verify_ctx = llama_session_internal_context(llama_decoder_session(verify));
    struct llama_context * draft_ctx  = llama_session_internal_context(llama_decoder_session(draft));
    for (int32_t i = 1; i < n_loops; ++i) {
        if (loops[i] == nullptr) return -1;
        if (llama_session_internal_context(llama_decoder_session(loops[i]->verify)) != verify_ctx) return -1;
        if (loops[i]->drafts.empty()) return -1;
        if (llama_session_internal_context(llama_decoder_session(loops[i]->drafts[0])) != draft_ctx) return -1;
    }

    const int32_t n = llama_spec_mtp_draft_batched(
            verify, draft,
            slots, n_loops,
            p_min,
            drafts_out, drafts_out_stride,
            outs);

    if (n > 0) {
        // Update per-loop stats; last_accepted is per-slot now (slot-major
        // slice). Each slot's drafted count is outs[i].n_drafted.
        for (int32_t i = 0; i < n_loops; ++i) {
            const int32_t k = outs[i].n_drafted;
            loops[i]->n_drafted += k;
            loops[i]->last_accepted.assign(
                drafts_out + (size_t) i * (size_t) drafts_out_stride,
                drafts_out + (size_t) i * (size_t) drafts_out_stride + (size_t) k);
        }
    } else {
        for (int32_t i = 0; i < n_loops; ++i) {
            loops[i]->last_accepted.clear();
        }
    }
    return n;
}

int32_t llama_spec_loop_step(struct llama_spec_loop * /*loop*/, struct llama_batch /*batch*/) {
    return spec_loop_step_unimplemented(__func__);
}

const llama_token * llama_spec_loop_last_accepted(const struct llama_spec_loop * loop, int32_t * n_accepted) {
    if (n_accepted) *n_accepted = (int32_t) loop->last_accepted.size();
    return loop->last_accepted.data();
}

float   llama_spec_loop_accept_rate    (const struct llama_spec_loop * loop) {
    if (loop->n_drafted == 0) return 0.0f;
    return (float) loop->n_accepted / (float) loop->n_drafted;
}
int64_t llama_spec_loop_n_drafted      (const struct llama_spec_loop * loop) { return loop->n_drafted;       }
int64_t llama_spec_loop_n_accepted     (const struct llama_spec_loop * loop) { return loop->n_accepted;      }
int64_t llama_spec_loop_n_verify_steps (const struct llama_spec_loop * loop) { return loop->n_verify_steps;  }

struct llama_decoder * llama_spec_loop_verify(const struct llama_spec_loop * loop) { return loop->verify; }
struct llama_decoder * llama_spec_loop_draft (const struct llama_spec_loop * loop, int idx) {
    if (idx < 0 || (size_t) idx >= loop->drafts.size()) return nullptr;
    return loop->drafts[idx];
}
int llama_spec_loop_n_draft_decoders(const struct llama_spec_loop * loop) {
    return (int) loop->drafts.size();
}

} // extern "C"
