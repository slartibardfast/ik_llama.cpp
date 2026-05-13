// test-dflash-load-vocab-identity.cpp
//
// Drives: specs/dflash/dflash.allium @ invariant DrafterTargetVocabIdentity.
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
// Spec: specs/dflash/dflash.allium @ invariant
//       DrafterTargetVocabIdentity
//       SharedEmbedAndLMHead

#include "llama.h"

#include <cstdio>

int main() {
    // T5 status check. Real implementation rejects null pointers with
    // LLAMA_DFLASH_INVALID_DRAFTER (sidecar drafter API). Once a real
    // Qwen3.6-27B-DFlash GGUF is loadable end-to-end, this body should
    // be replaced with the full DrafterTargetVocabIdentity test in the
    // file header.
    const int32_t rc = llama_set_dflash(nullptr, nullptr);
    if (rc != LLAMA_DFLASH_INVALID_DRAFTER && rc != LLAMA_DFLASH_NOT_IMPLEMENTED) {
        fprintf(stderr,
                "test-dflash-load-vocab-identity: unexpected status %d for null args.\n"
                "  Expected LLAMA_DFLASH_INVALID_DRAFTER (-2) or LLAMA_DFLASH_NOT_IMPLEMENTED (-1).\n",
                rc);
        return 1;
    }

    fprintf(stderr,
            "TODO: implement DrafterTargetVocabIdentity test against a real\n"
            "      Qwen3.6-27B-DFlash GGUF loaded via llama_dflash_drafter_load.\n"
            "      Currently stubbed at null-pointer null-check level.\n");
    return 77;
}
