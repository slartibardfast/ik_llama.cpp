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
