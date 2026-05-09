//
// PHASE 45 D6/D7/D8/D9.6h internal bridge.
//
// Exposes the wrapped llama_context pointer from llama_session for
// other libllama-internal .cpps that need direct ctx access (decoder
// forward, spec primitives), AND (D9.6h) the full llama_session
// struct definition so llama_context can hold pointer access into
// session-owned fields (lora_adapters, cvec, backends, scale_data,
// embd_enc, seq_ids_enc) before storage formally moves at D9.8.
//
// NOT in the public include/ tree; libcommon and external consumers
// must not include this. Removed at D9.8 once llama_session owns the
// fields directly and llama_context is deleted.
//

#ifndef LLAMA_SESSION_INTERNAL_H
#define LLAMA_SESSION_INTERNAL_H

#include "llama.h"
#include "llama-session.h"

#ifdef __cplusplus

#include "ggml-backend.h"

#include <unordered_map>
#include <vector>
#include <set>

struct llama_context;
struct llama_lora_adapter;

// Forward-decl from llama-context.h (full def lives there).
struct llama_control_vector;

struct llama_session {
    struct llama_context * ctx       = nullptr;
    bool                   owns_ctx  = false;
    struct llama_session_params params{};

    // PHASE45 D9.6h: session-owned fields. After llama_session_create /
    // _adopt runs, llama_context's matching fields are moved into here
    // (vector/map move semantics) so the session is the canonical
    // owner. Internal helpers reach these via ctx->session_ref->FIELD;
    // ctx's destructor no longer frees them. Storage formally
    // disconnects from ctx at D9.8.
    std::unordered_map<struct llama_lora_adapter *, float> lora_adapters;
    std::vector<float>                                     scale_data;
    std::vector<ggml_backend_t>                            backends;
#ifdef GGML_USE_METAL
    ggml_backend_t                                         backend_metal = nullptr;
#endif
#ifdef GGML_USE_BLAS
    ggml_backend_t                                         backend_blas  = nullptr;
#endif
    ggml_backend_t                                         backend_cpu   = nullptr;
    std::vector<float>                                     embd_enc;
    std::vector<std::set<llama_seq_id>>                    seq_ids_enc;
};

extern "C" {
#endif

    struct llama_context * llama_session_internal_context(struct llama_session * session);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_SESSION_INTERNAL_H
