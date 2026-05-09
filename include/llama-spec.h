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
    // attached to the draft decoder. Draft sampling is greedy (argmax)
    // with the device-side argmax fast path used when available — this
    // matches the semantics of `common_sampler_sample_speculative` and
    // is the only mode `mtp_speculative_gen_draft` exercises in
    // production (PHASE36/37/38).
    //
    // Inputs:
    //   verify_decoder  — VERIFY-role decoder; its session is also where
    //                     the draft decoder was created (shared session).
    //                     Hidden state for the MTP seed comes from the
    //                     verify decoder's last forward.
    //   draft_decoder   — DRAFT_MTP-role decoder.
    //   id_last         — last token from the verify path (the seed token
    //                     for the MTP head's first step).
    //   p_min           — minimum chain probability; truncate the draft
    //                     when the per-step argmax probability drops below.
    //                     Use 0.0 to disable truncation (full chain).
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
            llama_token            id_last,
            float                  p_min,
            int32_t                n_draft_max,
            llama_seq_id           seq_id,
            llama_pos              n_past,
            llama_token          * drafts_out);

    // PHASE45 D10.b: batched MTP draft. Generates drafts for `n_slots`
    // active slots that share the verify_decoder + draft_decoder pair.
    // Each step issues ONE forward of n_slots tokens (one per alive slot)
    // through the draft decoder, replacing N sequential per-slot decodes.
    //
    // Input layout: `slots[0..n_slots-1]` describes one slot each
    // (seq_id + id_last + n_past + n_draft_max). All slots' draft
    // chains share the same `p_min` truncation threshold.
    //
    // Output layout: `drafts_out` is slot-major with stride
    // `drafts_out_stride` (in tokens) — slot i's drafts live at
    // drafts_out[i*stride + 0 .. i*stride + outs[i].n_drafted-1]. The
    // caller sets stride to max(slots[].n_draft_max).
    //
    // Per-slot output: `outs[i].n_drafted` is the number of valid tokens
    // produced for slot i (0 if no progress was possible, e.g. all slots
    // started dead). `outs[i].truncated` is true if p_min cut the chain
    // before reaching n_draft_max.
    //
    // LLAMA_MTP_FUSED is NOT consulted by this path — fused single-shot
    // batches a one-slot chain, not multi-slot. Callers wanting fused
    // for n_slots==1 must use the single-slot llama_spec_mtp_draft.
    //
    // Returns total drafts emitted across all slots, or negative on
    // setup error.
    typedef struct {
        llama_seq_id seq_id;
        llama_token  id_last;
        llama_pos    n_past;
        int32_t      n_draft_max;
    } llama_spec_mtp_slot_in;

    typedef struct {
        int32_t n_drafted;     // 0..n_draft_max
        bool    truncated;     // p_min cut chain short
    } llama_spec_mtp_slot_out;

    LLAMA_API int32_t llama_spec_mtp_draft_batched(
            struct llama_decoder              * verify_decoder,
            struct llama_decoder              * draft_decoder,
            const llama_spec_mtp_slot_in      * slots,
            int32_t                             n_slots,
            float                               p_min,
            llama_token                       * drafts_out,
            int32_t                             drafts_out_stride,
            llama_spec_mtp_slot_out           * outs);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_SPEC_H
