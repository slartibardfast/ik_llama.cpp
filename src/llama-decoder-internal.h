//
// PHASE 45 D9.6b internal: full struct llama_decoder definition.
//
// Exposes the decoder struct so llama_context can hold a `default_decoder`
// member by value and back-reference into it before any user-created
// decoder exists. Without this, perf-counter accesses during ctx
// construction (warmup decode) would deref a null `decoder_ref`.
//
// NOT in the public include/ tree; libcommon and external consumers
// must not include this. Removed at D9.8 once llama_context is deleted.
//
#ifndef LLAMA_DECODER_INTERNAL_H
#define LLAMA_DECODER_INTERNAL_H

#include "llama.h"
#include "llama-decoder.h"

struct llama_session;

struct llama_decoder {
    struct llama_session    * session = nullptr;
    struct llama_decoder_params params{};
    enum   llama_decoder_role role    = LLAMA_DECODER_PRIMARY;

    // PHASE45 D9.6b: perf counters (formerly on llama_context).
    // Verify and draft decoders track separately.
    int64_t t_p_eval_us       = 0;
    int64_t t_eval_us         = 0;
    int64_t t_compute_start_us = 0;
    int64_t n_queued_tokens   = 0;
    int32_t n_p_eval          = 0;
    int32_t n_eval            = 0;
};

#endif // LLAMA_DECODER_INTERNAL_H
