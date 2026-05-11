// test-dflash-load-source-layer-count.cpp
//
// Drives: dflash_speculative.allium @ invariant
//         SourceLayerCountMatchesDrafterTraining.
//
// Property: the count of target source layers the drafter expects
// features from is read from GGUF metadata key
// dflash.target_layer_ids; the implementation must not hardcode 5.
//
// Upstream llama.cpp PR #22105 hardcodes this — set_dflash logs
// indices [0]..[4] unconditionally at llama-context.cpp:1226-1231,
// which array-out-of-bounds the log line if a drafter ships ≠5.
// Per paper §5 most drafters use 5; for Qwen3.6-27B-DFlash specifically
// we cannot verify the count until OQ-DRAFTER-CONFIG-GATED resolves.
// The implementation must read whatever count is in the GGUF, not
// assume 5.
//
// Test strategy (when implementation lands):
//   1. Load drafter z-lab/Qwen3.6-27B-DFlash GGUF.
//   2. Read llama_dflash_n_source_layers(model_dft).
//   3. Assert result > 0 AND <= some sane bound (say 16).
//   4. Cross-check: the count equals the length of dflash.target_layer_ids
//      array in the GGUF metadata (verified independently via
//      llama_model_meta_count / llama_model_meta_val_str path).
//   5. Negative case: a drafter GGUF lacking dflash.target_layer_ids
//      metadata makes llama_set_dflash return
//      LLAMA_DFLASH_MISSING_METADATA.
//
// Currently RED: llama_dflash_n_source_layers returns -1 (stub).

#include "llama.h"

#include <cstdio>

int main() {
    // RED until implementation lands.
    const int32_t stub = llama_dflash_n_source_layers(nullptr);
    if (stub != -1) {
        fprintf(stderr,
                "test-dflash-load-source-layer-count: stub no longer returns -1 (got %d).\n"
                "  Implementation has begun; fill in the body per the header plan.\n",
                stub);
        return 1;
    }

    fprintf(stderr,
            "TODO: implement SourceLayerCountMatchesDrafterTraining test\n"
            "      once a real Qwen3.6-27B-DFlash GGUF is loadable.\n");
    return 77;
}
