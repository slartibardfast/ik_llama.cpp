// test-dflash-load-dense-qwen3.cpp
//
// Drives: dflash_speculative.allium @ invariant DrafterIsDenseQwen3.
//
// Property: a DFlash drafter is dense Qwen3 — its per-layer
// attention type is either FULL_ATTENTION or SLIDING_ATTENTION,
// never linear_attention (DeltaNet). If a drafter declares any
// linear_attention layer in its GGUF metadata or HF config,
// llama_set_dflash must reject with LLAMA_DFLASH_HYBRID_DRAFTER.
//
// This bind mirrors vLLM's _DFLASH_VALID_LAYER_TYPES = {
// "full_attention", "sliding_attention" } in qwen3_dflash.py:55.
//
// Test strategy (when implementation lands):
//   1. Load a valid drafter — expect llama_set_dflash → OK.
//   2. For every layer index i in [0, n_layers):
//      assert llama_dflash_layer_type_at(model_dft, i) ∈
//             { LLAMA_DFLASH_LAYER_FULL_ATTENTION,
//               LLAMA_DFLASH_LAYER_SLIDING_ATTENTION }
//   3. Negative case: fabricate (or take from a hypothetical
//      pathological GGUF) a drafter that declares layer_types
//      containing "linear_attention" in metadata.
//      llama_set_dflash must return LLAMA_DFLASH_HYBRID_DRAFTER.
//
// Currently RED: llama_dflash_layer_type_at returns
// LLAMA_DFLASH_LAYER_FULL_ATTENTION unconditionally (stub does not
// read metadata). Real impl reads dflash.layer_types from GGUF.

#include "llama.h"

#include <cstdio>

int main() {
    // RED until implementation lands. Stub always returns
    // FULL_ATTENTION regardless of which model/layer is queried.
    const enum llama_dflash_layer_type stub =
        llama_dflash_layer_type_at(nullptr, /*layer_idx=*/0);

    // Sanity: the stub returns a well-formed enum value.
    if (stub != LLAMA_DFLASH_LAYER_FULL_ATTENTION &&
        stub != LLAMA_DFLASH_LAYER_SLIDING_ATTENTION) {
        fprintf(stderr,
                "FAIL: llama_dflash_layer_type_at returned out-of-band value %d\n",
                (int) stub);
        return 1;
    }

    fprintf(stderr,
            "TODO: implement DrafterIsDenseQwen3 test once a real\n"
            "      Qwen3.6-27B-DFlash GGUF is loadable. Requires:\n"
            "        - GGUF metadata reader for dflash.layer_types\n"
            "        - llama_set_dflash with layer-type validation\n"
            "        - LLAMA_DFLASH_HYBRID_DRAFTER reject path\n");
    return 77;
}
