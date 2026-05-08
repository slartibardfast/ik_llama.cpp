//
// PHASE 45 D6/D7/D8 internal bridge.
//
// Exposes the wrapped llama_context pointer from llama_session for
// other libllama-internal .cpps that need direct ctx access (decoder
// forward, spec primitives). NOT in the public include/ tree;
// libcommon and external consumers must not include this. Removed at
// D10 once llama_session owns the fields directly and llama_context
// is deleted.
//

#ifndef LLAMA_SESSION_INTERNAL_H
#define LLAMA_SESSION_INTERNAL_H

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

    struct llama_session;

    struct llama_context * llama_session_internal_context(struct llama_session * session);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_SESSION_INTERNAL_H
