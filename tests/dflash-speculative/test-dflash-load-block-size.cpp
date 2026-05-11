// test-dflash-load-block-size.cpp
//
// Drives: specs/dflash/dflash.allium @ invariant
//         BlockSizeFixedPerDeployment.
//
// Property: DraftBlock.block_size is a deployment-time constant
// read from drafter GGUF metadata (key dflash.block_size). It is
// stable across queries and cannot be reconfigured mid-session
// without a fresh context. specs/dflash/DESIGN.md commits to
// starting at 8 (small ne[1] verify shape known-deterministic on
// sm_75) and moving to 16 after Gate-5 binds determinism at the
// larger shape.
//
// Per spec: changing block_size mid-session would invalidate the
// drafter's positional mask assumptions and the verify-batch shape
// contract. Implementation must read once at load and refuse
// changes thereafter.
//
// Test strategy (when implementation lands):
//   1. Load drafter z-lab/Qwen3.6-27B-DFlash GGUF.
//   2. block_size_a = llama_dflash_block_size(model_dft).
//      Assert block_size_a > 0 (sane lower bound) and <= 64
//      (sane upper; published values are 8, 10, 15, 16).
//   3. block_size_b = llama_dflash_block_size(model_dft).
//      Assert block_size_a == block_size_b (idempotent read).
//   4. The mask_token_id is part of the same metadata bundle:
//      mask_id = llama_dflash_mask_token_id(model_dft).
//      Assert mask_id >= 0 AND mask_id < llama_n_vocab(model_tgt).
//      (DrafterTargetVocabIdentity supports this.)
//
// Currently RED: llama_dflash_block_size returns -1 (stub).

#include "llama.h"

#include <cstdio>

int main() {
    // RED until implementation lands.
    const int32_t bs = llama_dflash_block_size(nullptr);
    const llama_token mid = llama_dflash_mask_token_id(nullptr);

    if (bs != -1) {
        fprintf(stderr,
                "test-dflash-load-block-size: stub block_size no longer -1 (got %d).\n",
                bs);
        return 1;
    }
    if (mid != -1) {
        fprintf(stderr,
                "test-dflash-load-block-size: stub mask_token no longer -1 (got %d).\n",
                (int) mid);
        return 1;
    }

    fprintf(stderr,
            "TODO: implement BlockSizeFixedPerDeployment test once a\n"
            "      real Qwen3.6-27B-DFlash GGUF is loadable.\n");
    return 77;
}
