//
// PHASE 45 D8 extraction: libllama-level MTP draft primitive.
//
// D8.1b/c: sync fused fast path + per-step fallback, both extracted from
// common/speculative.cpp's `mtp_speculative_gen_draft`. Draft sampling
// is greedy (argmax) — `common_sampler_sample_speculative` short-circuits
// the caller's sampler chain in favor of device-side argmax (fused
// path) or host-fallback argmax (per-step path). The "sampler" passed
// at the libcommon layer is irrelevant to draft selection; it matters
// only on the verify side.
//
// Decision: env-gate `LLAMA_MTP_FUSED` matches today's runtime knob —
// changing it would silently affect production behaviour. The bench
// configs in `scripts/bench-multiturn-pre-port.sh` (e.g.,
// "D_mtp_d3_ikv_fused_minprob") set this var; the +19% target the D8.4
// gate enforces depends on the fused path.
//
// Genuinely out of libllama scope:
//   - Async fused dispatch (LLAMA_MTP_FULL_2) — PHASE38 E was abandoned,
//     PHASE45 supersedes.
//   - Top-2 probe (LLAMA_PROBE_TOP2) — calls libcommon's
//     `probe_top2_push`; cannot live in libllama without breaking the
//     dependency direction. Re-add at the D8.3 libcommon shim layer.
//   - Autotune (spec-tuner) — application policy, libcommon home.
//
// Deferred but in libllama scope:
//   - Per-step decode profiling (LLAMA_PROFILE_DECODE) — diagnostic;
//     no architectural blocker. Add later if profiling tooling needs it.
//

#include "llama-spec.h"
#include "llama-session.h"
#include "llama-session-internal.h"
#include "llama-decoder.h"

#include <cmath>
#include <cstdlib>

// Argmax + softmax-prob over the last logits row. Matches
// `common_sampler_sample_speculative` semantics: argmax token id and
// the softmax probability of that token. Used when the device-side
// argmax fast path is unavailable.
static llama_token spec_argmax_with_prob(const float * logits, int n_vocab, float * out_prob) {
    if (logits == nullptr || n_vocab <= 0) {
        if (out_prob) *out_prob = 0.0f;
        return -1;
    }
    int   best_id = 0;
    float max_l   = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_l) { max_l = logits[i]; best_id = i; }
    }
    if (out_prob) {
        double sum = 0.0;
        for (int i = 0; i < n_vocab; ++i) {
            sum += std::exp((double) (logits[i] - max_l));
        }
        *out_prob = (sum > 0.0) ? (float) (1.0 / sum) : 0.0f;
    }
    return (llama_token) best_id;
}

extern "C" {

int32_t llama_spec_mtp_draft(
        struct llama_decoder * verify_decoder,
        struct llama_decoder * draft_decoder,
        llama_token            id_last,
        float                  p_min,
        int32_t                n_draft_max,
        llama_seq_id           seq_id,
        llama_pos              n_past,
        llama_token          * drafts_out) {

    if (verify_decoder == nullptr || draft_decoder == nullptr || drafts_out == nullptr) return -1;
    if (n_draft_max <= 0) return 0;

    // PHASE45 invariant: VERIFY and DRAFT decoders share the same session.
    // Both write to the same transformer K/V; the draft writes layer N-1
    // exclusively (see PHASE45_PHASE39_INTEGRATION.md §4).
    struct llama_session * session = llama_decoder_session(draft_decoder);
    if (session == nullptr || session != llama_decoder_session(verify_decoder)) return -1;

    struct llama_context * ctx = llama_session_internal_context(session);
    if (ctx == nullptr) return -1;

    const struct llama_model * model = llama_decoder_model(draft_decoder);
    if (model == nullptr) return -1;
    const int n_vocab = llama_n_vocab(model);
    if (n_vocab <= 0) return -1;

    // ---- Fast path: sync fused chain --------------------------------------
    //
    // PHASE36 Step 1.4: when the chain is greedy + within LLAMA_MTP_FUSED_MAX,
    // a single fused cgraph emits all draft tokens at once instead of N
    // sequential per-step decodes. Env-gated so behaviour matches today's
    // production server. On any failure (env not set, n outside fused range,
    // dispatch rc != 0, n_steps == 0), fall through to per-step.
    {
        static const bool fused_enabled = (std::getenv("LLAMA_MTP_FUSED") != nullptr);
        if (fused_enabled && n_draft_max > 1 && n_draft_max <= LLAMA_MTP_FUSED_MAX) {
            llama_set_mtp_op_type(ctx, MTP_OP_DRAFT_GEN_FUSED);
            llama_mtp_fused_result fr{};
            const int32_t rc = llama_mtp_fused_draft_invoke(
                    ctx, id_last, /*seed_hidden=*/nullptr, n_draft_max, &fr);
            llama_set_mtp_op_type(ctx, MTP_OP_NONE);

            if (rc == 0 && fr.n_steps > 0) {
                // p_min truncation on the per-step argmax probabilities
                // (host-side from ggml_backend_cuda_mtp_argmax_with_prob_to_host).
                int32_t n_use = fr.n_steps;
                if (p_min > 0.0f) {
                    for (int k = 0; k < fr.n_steps; ++k) {
                        if (fr.probs[k] < p_min) { n_use = k; break; }
                    }
                }
                for (int32_t k = 0; k < n_use; ++k) {
                    drafts_out[k] = fr.tokens[k];
                }
                // Fused path does not pollute KV-cell metadata (writes go
                // through the persist[] tensors, not the live cells), so
                // no kv_seq_rm purge is needed here.
                return n_use;
            }
            // Fall through to per-step on dispatch failure.
        }
    }

    // ---- Slow path: per-step chain ----------------------------------------

    llama_batch batch = llama_batch_init(/*n_tokens=*/1, /*embd=*/0, /*n_seq_max=*/1);

    // Flip op_type to DRAFT_GEN. Restored to NONE on exit. PHASE45.md
    // will eventually map this off decoder.role rather than via this
    // side channel.
    llama_set_mtp_op_type(ctx, MTP_OP_DRAFT_GEN);

    llama_token current_id  = id_last;
    llama_pos   current_pos = n_past;
    int32_t     n_drafts    = 0;

    for (int32_t i = 0; i < n_draft_max; ++i) {
        // Build single-token batch: token, pos, seq_id, logits=true.
        batch.n_tokens         = 1;
        batch.token   [0]      = current_id;
        batch.pos     [0]      = current_pos;
        batch.n_seq_id[0]      = 1;
        batch.seq_id  [0][0]   = seq_id;
        batch.logits  [0]      = 1;

        // Forward through the draft decoder. The decoder API enforces
        // its own params (n_threads, causal_attn, embeddings) before
        // forwarding to llama_decode.
        if (llama_decoder_decode(draft_decoder, batch) != 0) break;

        // Greedy: try the device-cached argmax first (populated during
        // the DRAFT_GEN forward, avoids the per-draft ~2 MB logits
        // D2H), fall back to host-side argmax when the cache is cold.
        llama_token id_next = -1;
        float       prob    = 0.0f;
        {
            int32_t cached_id  = -1;
            float   cached_prob = 0.0f;
            if (llama_get_draft_argmax(ctx, /*i=*/0, &cached_id, &cached_prob)) {
                id_next = (llama_token) cached_id;
                prob    = cached_prob;
            } else {
                const float * logits = llama_get_logits_ith(ctx, /*i=*/0);
                id_next = spec_argmax_with_prob(logits, n_vocab, &prob);
            }
        }
        if (id_next < 0 || id_next >= n_vocab) break;

        drafts_out[n_drafts++] = id_next;

        // Pull the hidden state and feed it back as next step's input —
        // this is the MTP head's recurrent seed for chain step i+1.
        const float * emb = llama_get_embeddings_ith(ctx, /*i=*/0);
        if (emb != nullptr) {
            llama_set_draft_input_hidden_state(ctx, emb);
        }

        current_id   = id_next;
        current_pos += 1;

        // p_min truncation: stop when the model's argmax probability
        // falls below the threshold. Each truncation saves one
        // verify-side forward at the cost of one fewer drafted token.
        // p_min == 0 disables truncation.
        if (p_min > 0.0f && prob < p_min) break;
    }

    llama_batch_free(batch);
    llama_set_mtp_op_type(ctx, MTP_OP_NONE);

    // Purge KV-cell metadata for the drafted positions. The verify
    // decoder's forward will re-populate these positions for the
    // accepted prefix; without the purge, we'd hold two cells
    // mapping to the same logical position.
    if (n_drafts > 0) {
        llama_kv_cache_seq_rm(ctx, seq_id, n_past, current_pos);
    }

    return n_drafts;
}

} // extern "C"
