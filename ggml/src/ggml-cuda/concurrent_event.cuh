#pragma once

#include "common.cuh"
#include "ggml-backend.h"

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Per-slot CUDA stream handle for the active fan-out interval.
// Caller must have invoked ggml_backend_concurrent_event_fork(event) before
// dispatching kernels on this stream and ggml_backend_concurrent_event_join(event)
// after, so the main stream observes the per-slot work in dependency order.
//
// Returns nullptr only if event is non-NULL but has been freed (programmer error).
// Aborts on out-of-range slot_idx.
cudaStream_t ggml_backend_cuda_concurrent_event_stream(ggml_backend_concurrent_event_t event, int slot_idx);

#ifdef __cplusplus
}
#endif
