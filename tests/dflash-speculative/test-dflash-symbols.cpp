// test-dflash-symbols.cpp
//
// Compile-only RED test for dflash_speculative.allium API surface.
//
// Drives:
//   - existence of enum llama_dflash_layer_type {
//         LLAMA_DFLASH_LAYER_FULL_ATTENTION,
//         LLAMA_DFLASH_LAYER_SLIDING_ATTENTION
//     } and the spec's DrafterLayerType enum.
//   - existence of enum llama_dflash_status with the seven sentinel
//     values the contract returns (OK + 6 error codes).
//   - existence and linkability of:
//         llama_set_dflash
//         llama_dflash_n_source_layers
//         llama_dflash_block_size
//         llama_dflash_mask_token_id
//         llama_dflash_swa_window
//         llama_dflash_layer_type_at
//
// This test does not exercise behaviour. It fails to compile (or
// link) until the implementation adds the named symbols. Once it
// compiles + links the runtime body returns 0; the per-invariant
// tests in this bundle then drive behavioural slices.
//
// Spec: yarn-agentic/dflash_speculative.allium

#include "llama.h"

#include <cstdio>

int main() {
    // 1. Layer-type enum: both variants must exist and be distinct.
    //    Per DrafterIsDenseQwen3 invariant: only these two are valid.
    static_assert(LLAMA_DFLASH_LAYER_FULL_ATTENTION    != LLAMA_DFLASH_LAYER_SLIDING_ATTENTION,
                  "DFlash layer-type variants must be distinct");

    // 2. Status enum: all seven sentinels must exist and be distinct.
    static_assert(LLAMA_DFLASH_OK                != LLAMA_DFLASH_NOT_IMPLEMENTED, "");
    static_assert(LLAMA_DFLASH_NOT_IMPLEMENTED   != LLAMA_DFLASH_INVALID_DRAFTER, "");
    static_assert(LLAMA_DFLASH_INVALID_DRAFTER   != LLAMA_DFLASH_VOCAB_MISMATCH,  "");
    static_assert(LLAMA_DFLASH_VOCAB_MISMATCH    != LLAMA_DFLASH_HYBRID_DRAFTER,  "");
    static_assert(LLAMA_DFLASH_HYBRID_DRAFTER    != LLAMA_DFLASH_MISSING_METADATA, "");
    static_assert(LLAMA_DFLASH_MISSING_METADATA  != LLAMA_DFLASH_MULTIMODAL_PROMPT, "");

    // 3. Function symbols: take addresses to force link-time binding.
    //    Per spec contracts ExtractFeatures, ProjectAndFuse,
    //    DraftBlockEmit, and AdvanceState — all surface through this
    //    API at the C boundary.
    typedef int32_t (*set_dflash_fn)(struct llama_context *, struct llama_model *);
    typedef int32_t (*n_source_layers_fn)(const struct llama_model *);
    typedef int32_t (*block_size_fn)(const struct llama_model *);
    typedef llama_token (*mask_token_fn)(const struct llama_model *);
    typedef int32_t (*swa_window_fn)(const struct llama_model *, int32_t);
    typedef enum llama_dflash_layer_type (*layer_type_at_fn)(const struct llama_model *, int32_t);

    set_dflash_fn       p1 = &llama_set_dflash;
    n_source_layers_fn  p2 = &llama_dflash_n_source_layers;
    block_size_fn       p3 = &llama_dflash_block_size;
    mask_token_fn       p4 = &llama_dflash_mask_token_id;
    swa_window_fn       p5 = &llama_dflash_swa_window;
    layer_type_at_fn    p6 = &llama_dflash_layer_type_at;

    if (p1 == nullptr || p2 == nullptr || p3 == nullptr ||
        p4 == nullptr || p5 == nullptr || p6 == nullptr) {
        fprintf(stderr, "FAIL: one or more DFlash symbols null after &-take\n");
        return 1;
    }

    printf("=== test-dflash-symbols ===\n");
    printf("  llama_dflash_layer_type   enum present (2 variants)\n");
    printf("  llama_dflash_status       enum present (7 sentinels)\n");
    printf("  llama_set_dflash          linkable\n");
    printf("  llama_dflash_n_source_layers linkable\n");
    printf("  llama_dflash_block_size   linkable\n");
    printf("  llama_dflash_mask_token_id linkable\n");
    printf("  llama_dflash_swa_window   linkable\n");
    printf("  llama_dflash_layer_type_at linkable\n");
    printf("  GREEN — symbol surface exists\n");
    return 0;
}
