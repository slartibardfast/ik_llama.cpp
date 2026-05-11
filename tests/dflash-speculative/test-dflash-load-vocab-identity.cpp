// test-dflash-load-vocab-identity.cpp
//
// Drives: dflash_speculative.allium @ invariant DrafterTargetVocabIdentity.
//
// Property: when a DFlash drafter is bound to a target context via
// llama_set_dflash, the two models must share exact vocabulary —
// same vocab_size and same token-id space. This makes the
// SharedEmbedAndLMHead invariant sound (the LM head emits indices
// into a vocab the drafter's drafts must also use).
//
// Test strategy (when implementation lands):
//   1. Load target Qwen3.6-27B at IQ4_KS.
//   2. Load drafter z-lab/Qwen3.6-27B-DFlash GGUF.
//   3. Bind via llama_set_dflash(ctx_tgt, model_dft).
//   4. Assert llama_set_dflash returned LLAMA_DFLASH_OK.
//   5. Assert llama_n_vocab(model_tgt) == llama_n_vocab(model_dft).
//   6. Assert special token ids match (BOS, EOS, vision-start /
//      vision-end / image / video for Qwen3.6).
//   7. Negative case: load a drafter whose tokenizer differs;
//      llama_set_dflash must return LLAMA_DFLASH_VOCAB_MISMATCH.
//
// Currently RED: llama_set_dflash returns LLAMA_DFLASH_NOT_IMPLEMENTED
// (stub). Exit 77 (CTest skip) until a real drafter GGUF is
// converted via convert_hf_to_gguf.py's DFlashModel class.
//
// Spec: yarn-agentic/dflash_speculative.allium @ invariant
//       DrafterTargetVocabIdentity
//       SharedEmbedAndLMHead

#include "llama.h"

#include <cstdio>

int main() {
    // RED until implementation lands. The eventual implementation
    // binds llama_set_dflash to a real drafter and verifies the
    // vocab match before declaring OK.
    const int32_t stub = llama_set_dflash(nullptr, nullptr);
    if (stub != LLAMA_DFLASH_NOT_IMPLEMENTED) {
        // Stub is no longer present — implementation has begun.
        // Replace this body with the real test plan above.
        fprintf(stderr,
                "test-dflash-load-vocab-identity: stub status changed to %d.\n"
                "  Implementation has begun; this test needs to be filled in\n"
                "  with the steps in the file header.\n",
                stub);
        return 1;
    }

    fprintf(stderr,
            "TODO: implement DrafterTargetVocabIdentity test once a real\n"
            "      Qwen3.6-27B-DFlash GGUF is loadable. Requires:\n"
            "        - Converter DFlashModel class\n"
            "        - llama_set_dflash with vocab-cross-check at bind\n");
    return 77;
}
