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
#include "ggml-backend.h"

#include <vector>
#include <map>
#include <set>

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

    // PHASE45 D9.6c: output buffers (formerly on llama_context).
    ggml_backend_buffer_t buf_output = nullptr;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers
    size_t  output_size = 0; // capacity (of tokens positions) for the output buffers
    int32_t n_outputs   = 0; // number of actually-used outputs in the current ubatch or last logical batch

    bool logits_all = false;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0; // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    ~llama_decoder() {
        // PHASE45 D9.6c: decoder owns its output buffer; default_decoder
        // (held by-value in llama_context) frees on ctx teardown.
        if (buf_output != nullptr) {
            ggml_backend_buffer_free(buf_output);
            buf_output = nullptr;
        }
    }
};

#endif // LLAMA_DECODER_INTERNAL_H
