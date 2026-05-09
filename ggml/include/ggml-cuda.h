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

GGML_API void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data);

// Number of cached CUDA graphs currently held by this backend's context. The
// graph cache is keyed on cgraph topology pointer; its size grows as decode
// shapes vary. Exposed for tests that bound cache growth.
GGML_API GGML_CALL size_t ggml_backend_cuda_graph_cache_size(ggml_backend_t backend);

// PHASE46 E.1 — opt-in async-overlap gate.
//
// Returns nonzero if the caller (typically a draft-graph dispatch site)
// should schedule its work on the per-device draft_streams[] with a
// fork/join event guarding the merge into the main stream. Off by default
// (Phase 38 E ground truth: on sm_75, decode is memory-bandwidth-bound
// and async overlap measures negative). On sm_80+ where SM slack exists
// the user can opt in via LLAMA_DRAFT_OVERLAP=1.
//
// The flag is read once at first call and cached. Callers should test it
// at the dispatch boundary; downstream the fork/join is enforced by
// ggml_backend_concurrent_event_{fork,join} so the merge is deterministic
// regardless of overlap.
GGML_API GGML_CALL int ggml_backend_cuda_draft_overlap_enabled(void);

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
