// llama-dflash.cpp
//
// DFlash speculative decoding stub implementation.
//
// RED-first scaffold: every entry point returns
// LLAMA_DFLASH_NOT_IMPLEMENTED (or a corresponding -1 / -2 sentinel)
// until the real implementation lands. The behavioural tests under
// tests/dflash-speculative/ bind on the eventual correct return
// values; today they assert RED.
//
// Spec: specs/dflash/dflash.allium
// Design: specs/dflash/DESIGN.md

#include "llama.h"

int32_t llama_set_dflash(
        struct llama_context * /*ctx_tgt*/,
        struct llama_model   * /*model_dft*/) {
    return LLAMA_DFLASH_NOT_IMPLEMENTED;
}

int32_t llama_dflash_n_source_layers(const struct llama_model * /*model_dft*/) {
    return -1;
}

int32_t llama_dflash_block_size(const struct llama_model * /*model_dft*/) {
    return -1;
}

llama_token llama_dflash_mask_token_id(const struct llama_model * /*model_dft*/) {
    return -1;
}

int32_t llama_dflash_swa_window(
        const struct llama_model * /*model_dft*/,
        int32_t                    /*layer_idx*/) {
    return -2;
}

enum llama_dflash_layer_type llama_dflash_layer_type_at(
        const struct llama_model * /*model_dft*/,
        int32_t                    /*layer_idx*/) {
    // Stub returns full_attention so symbol-surface tests can
    // bind on the enum value. Behavioural tests assert the
    // drafter's actual layer composition; they fail RED until
    // the implementation reads from GGUF metadata.
    return LLAMA_DFLASH_LAYER_FULL_ATTENTION;
}
