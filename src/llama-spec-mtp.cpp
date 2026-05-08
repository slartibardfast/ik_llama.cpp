//
// PHASE 45 D8 extraction: libllama-level MTP draft primitive.
//
// D8.1a: signature + landing-zone stub. Aborts loudly if called so the
// extraction's downstream consumer (spec_loop_step at D8.2) fails fast
// rather than silently producing empty drafts. Body lands at D8.1b
// (port of `mtp_speculative_gen_draft`'s core path with sampler
// interaction adapted from `common_sampler` → `llama_sampler`).
//
// Out-of-scope for D8.1b (deferred to D8.1c+):
//   - Async fused dispatch (LLAMA_MTP_FULL_2 path)
//   - Autotune feedback (lives in libcommon)
//   - Top-2 probe instrumentation (LLAMA_PROBE_TOP2)
//
// In-scope for D8.1b:
//   - Sync fused dispatch via `llama_mtp_fused_draft_invoke`
//   - Per-step argmax sampling via `llama_sampler_sample`
//   - p_min truncation
//   - Hidden-state seeding via `llama_set_draft_input_hidden_state`
//

#include "llama-spec.h"

#include <cstdio>
#include <cstdlib>

extern "C" {

int32_t llama_spec_mtp_draft(
        struct llama_decoder * /*verify_decoder*/,
        struct llama_decoder * /*draft_decoder*/,
        struct llama_sampler * /*sampler*/,
        llama_token            /*id_last*/,
        float                  /*p_min*/,
        int32_t                /*n_draft_max*/,
        llama_seq_id           /*seq_id*/,
        llama_pos              /*n_past*/,
        llama_token          * /*drafts_out*/) {
    std::fprintf(stderr,
        "PHASE 45 D8.1a: llama_spec_mtp_draft is the extraction landing zone. "
        "Body fill is D8.1b work. Aborting to surface unintended use.\n");
    std::abort();
}

} // extern "C"
