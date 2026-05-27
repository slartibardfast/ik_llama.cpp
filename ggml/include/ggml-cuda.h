#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_HIPBLAS
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_API GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device, const void * params);

GGML_API GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_API GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_API GGML_CALL int  ggml_backend_cuda_get_device_count(void);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_API GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_API GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_API GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer);

// PHASE46 B.5 P2: peer-access verification. Returns true iff
// cudaDeviceCanAccessPeer(from_device, to_device) returns true at the
// driver level. Multi-GPU CLIP under tensor-split requires this on
// every (from, to) pair in the participating device set; an explicit
// check at init time gives a clean error rather than a cryptic
// failure during the first cross-device transfer. Formal contract:
// yarn-agentic specs/mgpu-split/ClipCrossDeviceFlow.tla.
GGML_API GGML_CALL bool ggml_backend_cuda_can_access_peer(int from_device, int to_device);

GGML_API void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data);

// Number of cached CUDA graphs currently held by this backend's context. The
// graph cache is keyed on cgraph topology pointer; its size grows as decode
// shapes vary. Exposed for tests that bound cache growth.
GGML_API GGML_CALL size_t ggml_backend_cuda_graph_cache_size(ggml_backend_t backend);

// PHASE_CUDA_NATIVE_DISPATCH C2: introspection hook for the eager-init
// invariant. Returns true iff the four PD1 lazy-create surfaces are
// already populated on this backend's context:
//   - copy_event (single cudaEvent_t per context)
//   - streams[device][stream] for device < device_count, stream < MAX_STREAMS
//   - cublas_handles[device] for device < device_count
//   - pools[device] for device < device_count
// Tests assert this returns true immediately after ggml_backend_cuda_init.
// Returns false if backend is null or not a CUDA backend.
GGML_API GGML_CALL bool ggml_backend_cuda_eager_init_complete(ggml_backend_t backend);

// PHASE_CUDA_NATIVE_DISPATCH C3: outer-capture state.
// C4's wrapper around ggml_backend_sched_compute_splits issues
// cudaStreamBeginCapture(Relaxed) on the primary backend's stream
// and sets this flag; cudaStreamEndCapture clears it. Per-backend
// ggml_backend_cuda_graph_compute reads the flag and skips its own
// capture / instantiate / launch lifecycle, letting kernels stream
// into the outer captured graph instead.
//
// Thread-local: C1's HostThreadIsExactlyOne invariant guarantees a
// single dispatcher thread owns this state.
GGML_API void ggml_cuda_outer_capture_set(bool active);
GGML_API bool ggml_cuda_outer_capture_active(void);

// PHASE_CUDA_NATIVE_DISPATCH C4: outer-capture orchestration.
//
// _begin starts a multi-device capture rooted on primary_backend's
// stream, then pulls each backend in secondary_backends into the
// capture via cross-stream event-chain (cudaEventRecord on primary,
// cudaStreamWaitEvent on each secondary). On success, sets the C3
// outer-capture flag so per-backend ggml_backend_cuda_graph_compute
// short-circuits its own capture/launch. Returns false if any
// participating backend is not a CUDA backend or if
// cudaStreamBeginCapture fails; caller MUST dispatch eagerly on false.
//
// _end_and_launch issues fan-in event-chain (every secondary backend
// records its completion event; primary stream waits on each), then
// cudaStreamEndCapture + cudaGraphInstantiate + cudaGraphLaunch +
// cudaGraphDestroy + cudaGraphExecDestroy. Returns false on any
// failure; on false return the caller's eager dispatch results are
// already in flight. Clears the C3 outer-capture flag.
//
// Counter: ggml_cuda_outer_capture_count() returns the number of
// successful _begin → _end_and_launch round-trips since process start
// (or since the last _count_reset).
GGML_API bool   ggml_cuda_outer_capture_begin(
                    ggml_backend_t   primary_backend,
                    ggml_backend_t * secondary_backends,
                    int              n_secondary);
GGML_API bool   ggml_cuda_outer_capture_end_and_launch(
                    ggml_backend_t   primary_backend,
                    ggml_backend_t * secondary_backends,
                    int              n_secondary);
GGML_API size_t ggml_cuda_outer_capture_count(void);
GGML_API void   ggml_cuda_outer_capture_count_reset(void);

// PHASE_CUDA_NATIVE_DISPATCH C5: split end-and-launch into end-capture
// + launch-exec so the sched can cache the instantiated graph. On cache
// hit the sched skips _begin entirely and calls _launch_exec on the
// cached cudaGraphExec_t handle (replayed against the primary stream
// with the same cross-stream join + fan-in baked into the captured
// graph at C4).
//
// _end_capture: fan-in event-chain + cudaStreamEndCapture +
//   cudaGraphInstantiate. Returns the instantiated executable via
//   out_exec (caller owns; destroy with ggml_cuda_outer_capture_destroy
//   _exec when evicting). Returns false on any failure; clears the
//   outer-capture flag either way.
// _launch_exec: cudaGraphLaunch(exec, primary->stream()). Increments
//   the capture counter. Returns false on launch error.
// _destroy_exec: cudaGraphExecDestroy. Called by the sched on
//   eviction / sched_free.
//
// Opaque to consumers: use the ggml_cuda_graph_exec_t alias rather
// than dereferencing.
typedef void * ggml_cuda_graph_exec_t;

GGML_API bool ggml_cuda_outer_capture_end_capture(
                    ggml_backend_t            primary_backend,
                    ggml_backend_t          * secondary_backends,
                    int                       n_secondary,
                    ggml_cuda_graph_exec_t  * out_exec);
GGML_API bool ggml_cuda_outer_capture_launch_exec(
                    ggml_backend_t           primary_backend,
                    ggml_cuda_graph_exec_t   exec);
GGML_API void ggml_cuda_outer_capture_destroy_exec(
                    ggml_cuda_graph_exec_t   exec);

// CUDA graph cache instrumentation surface. All functions return 0 / -1
// when GGML_CUDA_GRAPH_PROBE is unset or when the underlying probe step
// has not yet landed. Tests treat any zero return as RED.
GGML_API GGML_CALL size_t ggml_backend_cuda_graph_topology_class_count(ggml_backend_t backend);
GGML_API GGML_CALL size_t ggml_backend_cuda_graph_disable_vram_pressure_count(ggml_backend_t backend);
GGML_API GGML_CALL size_t ggml_backend_cuda_graph_update_failure_count(ggml_backend_t backend);
GGML_API GGML_CALL int    ggml_backend_cuda_graph_probe_flush(ggml_backend_t backend);
GGML_API GGML_CALL int    ggml_backend_cuda_graph_probe_active(void);

// Device-side argmax + softmax-prob over [n_rows, n_vocab] logits.
// Replaces the host-side argmax loop in common_sampler_sample_speculative when
// running greedy MTP draft sampling on a CUDA backend. Eliminates per-draft
// logits D2H (~2 MB at vocab=248,320 × 2 outputs) in favour of a tiny 8 B/row
// (id, prob) result.
GGML_API GGML_CALL void ggml_backend_cuda_mtp_argmax_with_prob(
    const void * logits_dev,    // float*, [n_rows, n_vocab] device pointer
    int n_rows,
    int n_vocab,
    void * dst_ids_dev,         // int32_t*, [n_rows] device pointer
    void * dst_probs_dev,       // float*,   [n_rows] device pointer
    void * dst_top2_ids_dev,    // int32_t*, [n_rows] device pointer; nullptr for default top-1-only path
    int device);

// Single-call variant: handles per-device scratch alloc + kernel launch + D2H to
// caller-provided host buffers. Synchronous (host buffers are valid on return).
// Pass host_top2_ids_out=nullptr for the default top-1-only fast path.
GGML_API GGML_CALL void ggml_backend_cuda_mtp_argmax_with_prob_to_host(
    const void * logits_dev,
    int n_rows,
    int n_vocab,
    int32_t * host_ids_out,
    float   * host_probs_out,
    int32_t * host_top2_ids_out,  // optional; nullptr to skip top-2 D2H
    int device);

// Synchronous device-to-device copy on the legacy default stream.
// Used by llama_kv_cache / MTP plumbing in src/llama.cpp without needing a
// cuda_runtime.h dependency.
GGML_API GGML_CALL void ggml_backend_cuda_memcpy_d2d(
    void * dst_dev,
    const void * src_dev,
    size_t nbytes,
    int device);

// Allocate persistent device buffer (cudaMalloc) of `nbytes` and return the pointer.
// Returns nullptr on failure. Caller frees with ggml_backend_cuda_free.
GGML_API GGML_CALL void * ggml_backend_cuda_malloc(size_t nbytes, int device);
GGML_API GGML_CALL void   ggml_backend_cuda_free(void * ptr);

// Test hook: drive a single allocation through the backend's scratch
// pool and return the pointer (or nullptr on soft-fail). Returns the
// allocated byte count via *actual_size_out. On success, the buffer
// must be released with ggml_backend_cuda_pool_free_test. Used by
// test-cuda-pool-graceful-oom to exercise the pool soft-fail
// path independent of which op happens to call into the pool.
GGML_API GGML_CALL void * ggml_backend_cuda_pool_alloc_test(ggml_backend_t backend,
                                                           size_t nbytes,
                                                           size_t * actual_size_out);
GGML_API GGML_CALL void   ggml_backend_cuda_pool_free_test(ggml_backend_t backend,
                                                          void * ptr,
                                                          size_t size);

// Device-side per-step state restore for delta-net / SSM checkpoint rollback.
// Reconstructs n_layers worth of recurrent state (conv portion + ssm portion) entirely
// on-device, eliminating the CPU-roundtrip that the host-side fallback in
// llama_kv_cache::per_step_restore performed via ggml_backend_tensor_get/set per layer.
//
// All pointer arrays must point to device memory on `device`. Pass nullptr for any
// shadow_ptrs[i] when no shadow conv state is available (early-step path).
GGML_API GGML_CALL void ggml_backend_cuda_per_step_restore_layers(
    int n_layers,
    void * const * dst_ptrs,            // s_l[il]->data
    const void * const * ssm_ptrs,      // per_step_ssm[il]->data (full per-step buffer)
    const void * const * qkv_ptrs,      // per_step_qkv[il]->data
    const void * const * shadow_ptrs,   // s_l_shadow[il]->data (or nullptr)
    int step,
    int64_t conv_state_dim,             // = (d_conv-1) * conv_dim (elements)
    int64_t conv_dim,
    int32_t d_conv,
    int64_t ssm_state_dim,              // elements per layer
    int device);

#ifdef  __cplusplus
}
#endif
