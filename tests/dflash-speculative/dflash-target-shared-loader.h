// dflash-target-shared-loader.h
//
// Loads the SHARED-with-drafter tensors from the production target GGUF:
//   - token_embd.weight   F16  [vocab, hidden]   — anchor + mask embeddings
//   - output.weight       BF16 [vocab, hidden]   — drafter lm_head
//   - output_norm.weight  F32  [hidden]          — drafter pre-lm_head norm
//
// Per @SharedEmbedAndLMHead, these three tensors are read once at server
// init and used by both target and drafter. In production they're the
// "same memory" — here they're uploaded to GPU once and addressed by
// both kernel paths via the same device pointers.
//
// Spec: specs/dflash/kernel-design.md §6.1 "Kernel boundary — lm_head".
//
// Skips all other target tensors (quantized layer weights, etc.) —
// we don't need them on the drafter path.

#pragma once

#include "ggml.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dflash_reference {

struct TargetSharedWeights {
    const __half        * token_embd  = nullptr;  // F16  [vocab, hidden]
    const __nv_bfloat16 * lm_head     = nullptr;  // BF16 [vocab, hidden]
    const float         * output_norm = nullptr;  // F32  [hidden]
    int vocab_size  = 0;
    int hidden_size = 0;

    std::vector<void *>   gpu_buffers;
    struct gguf_context * gguf_ctx = nullptr;
    struct ggml_context * ggml_ctx = nullptr;
};

inline bool load_target_shared(const char * gguf_path, TargetSharedWeights & t) {
    struct gguf_init_params params{};
    params.no_alloc = false;
    params.ctx      = &t.ggml_ctx;
    t.gguf_ctx = gguf_init_from_file(gguf_path, params);
    if (t.gguf_ctx == nullptr) {
        std::fprintf(stderr, "load_target_shared: gguf_init_from_file failed for %s\n", gguf_path);
        return false;
    }

    auto upload = [&](const char * name) -> std::pair<void *, std::size_t> {
        struct ggml_tensor * tn = ggml_get_tensor(t.ggml_ctx, name);
        if (tn == nullptr) {
            std::fprintf(stderr, "load_target_shared: tensor not found: %s\n", name);
            return {nullptr, 0};
        }
        const std::size_t nb = ggml_nbytes(tn);
        void * dev = nullptr;
        cudaError_t err = cudaMalloc(&dev, nb);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "load_target_shared: cudaMalloc(%zu) failed: %s\n",
                         nb, cudaGetErrorString(err));
            return {nullptr, 0};
        }
        cudaMemcpy(dev, tn->data, nb, cudaMemcpyHostToDevice);
        t.gpu_buffers.push_back(dev);
        std::printf("  loaded %-25s  type=%-10s  bytes=%zu\n",
                    name, ggml_type_name(tn->type), nb);
        if (std::strcmp(name, "token_embd.weight") == 0) {
            t.vocab_size  = static_cast<int>(tn->ne[1]);
            t.hidden_size = static_cast<int>(tn->ne[0]);
        }
        return {dev, nb};
    };

    auto p_te = upload("token_embd.weight");  if (!p_te.first)  return false;
    auto p_lh = upload("output.weight");      if (!p_lh.first)  return false;
    auto p_on = upload("output_norm.weight"); if (!p_on.first)  return false;
    t.token_embd  = static_cast<const __half *>(p_te.first);
    t.lm_head     = static_cast<const __nv_bfloat16 *>(p_lh.first);
    t.output_norm = static_cast<const float *>(p_on.first);

    // Verify dtypes match what we expect.
    {
        struct ggml_tensor * tn = ggml_get_tensor(t.ggml_ctx, "token_embd.weight");
        if (tn->type != GGML_TYPE_F16) {
            std::fprintf(stderr, "load_target_shared: token_embd.weight is %s, expected F16\n",
                         ggml_type_name(tn->type));
            return false;
        }
    }
    {
        struct ggml_tensor * tn = ggml_get_tensor(t.ggml_ctx, "output.weight");
        if (tn->type != GGML_TYPE_BF16) {
            std::fprintf(stderr, "load_target_shared: output.weight is %s, expected BF16\n",
                         ggml_type_name(tn->type));
            return false;
        }
    }
    return true;
}

inline void free_target_shared(TargetSharedWeights & t) {
    for (void * p : t.gpu_buffers) if (p) cudaFree(p);
    t.gpu_buffers.clear();
    if (t.gguf_ctx) { gguf_free(t.gguf_ctx); t.gguf_ctx = nullptr; }
    if (t.ggml_ctx) { ggml_free(t.ggml_ctx); t.ggml_ctx = nullptr; }
}

} // namespace dflash_reference
