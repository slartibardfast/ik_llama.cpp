//
// PHASE 45 D8 (extracted): Low-level speculation primitives for libllama.
//
// Today's `mtp_speculative_gen_draft` lives in `common/speculative.cpp`
// (libcommon) and depends on `common_sampler` (libcommon). That makes
// the algorithmic core unreachable from `llama_spec_loop` (libllama)
// because the dependency cannot run libllama → libcommon.
//
// This header begins the extraction. Bodies port the relevant logic
// down into libllama, taking `llama_sampler` (the chain primitive)
// directly rather than `common_sampler` (the libcommon glue).
// `common/speculative.cpp` will be refactored at D8.3 to call these
// directly; the libcommon layer keeps autotune (spec-tuner) and
// gpt_params parsing only.
//
// The functions are declared LLAMA_API so consumers (D8.2 spec_loop +
// the eventual common-shim path) can call them.
//

#ifndef LLAMA_SPEC_H
#define LLAMA_SPEC_H

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

    struct llama_decoder;

    // Generate up to n_draft_max speculative tokens from the MTP head
    // attached to the draft decoder, using the given sampler chain.
    //
    // Inputs:
    //   verify_decoder  — VERIFY-role decoder; its session is also where
    //                     the draft decoder was created (shared session).
    //                     Hidden state for the MTP seed comes from the
    //                     verify decoder's last forward.
    //   draft_decoder   — DRAFT_MTP-role decoder.
    //   sampler         — llama_sampler chain to apply at each draft step
    //                     (typically a greedy/argmax chain for MTP).
    //   id_last         — last token from the verify path (the seed token
    //                     for the MTP head's first step).
    //   p_min           — minimum chain probability; truncate the draft
    //                     when the per-step argmax probability drops below.
    //   n_draft_max     — maximum number of draft tokens to emit.
    //   seq_id          — sequence id (slot) the draft is for.
    //   n_past          — current position before the draft starts.
    //   drafts_out      — buffer to fill with draft token ids; capacity
    //                     must be ≥ n_draft_max.
    //
    // Returns: number of draft tokens emitted (0..n_draft_max). On
    // setup failure (nullptr decoders, mismatched session) returns -1.
    LLAMA_API int32_t llama_spec_mtp_draft(
            struct llama_decoder * verify_decoder,
            struct llama_decoder * draft_decoder,
            struct llama_sampler * sampler,
            llama_token            id_last,
            float                  p_min,
            int32_t                n_draft_max,
            llama_seq_id           seq_id,
            llama_pos              n_past,
            llama_token          * drafts_out);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_SPEC_H
