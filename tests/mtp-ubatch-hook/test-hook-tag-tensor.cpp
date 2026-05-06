// test-hook-tag-tensor.cpp
//
// Drives:
//   - mtp_ubatch_hook.allium THPreNormTensorTagged (contract invariant)
//
// The main forward graph for Qwen 3.5 dense, Qwen 3.5 MoE, and
// Qwen 3.6 27B/35B-A3B must expose a tagged tensor named
// "h_pre_norm" that points at the residual stream immediately
// before the final RMS norm, with shape (n_embd, n_tokens). The
// hook reads this tensor; main graph builders that omit the tag
// are out of contract.
//
// Today the tag does not exist. After build_qwen35() and
// build_qwen35moe() add `cb(cur, "h_pre_norm", -1)` and assign
// `res->t_h_pre_norm = cur` before the final norm, this test
// can find the tagged tensor and assert its shape.

#include "llama.h"
#include "ggml.h"

#include <cassert>
#include <cstdio>

int main() {
    // TODO(spec-driven, RED until implementation lands):
    //
    // 1. Load the test fixture model (Qwen 3.5 0.8B Q8_0 MTP at
    //    /opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0-MTP.gguf).
    // 2. Decode one prompt token to trigger graph build.
    // 3. Call llama_main_graph_h_pre_norm(ctx) — must return non-null.
    // 4. Assert tensor->ne[0] == n_embd from hparams.
    // 5. Assert tensor->ne[1] == 1 (single-token decode).
    // 6. Assert tensor->type == GGML_TYPE_F32.
    // 7. Strcmp tensor->name (or its ggml_get_name() value) against
    //    "h_pre_norm" — must match exactly. The cb() macro tags
    //    tensors with this name.
    //
    // For multi-ubatch decode (a longer prompt), the tensor's ne[1]
    // must equal the ubatch n_tokens. Add a second decode with a
    // 32-token prompt and re-check the shape.

    fprintf(stderr,
            "TODO: implement tag-tensor test once build_qwen35/moe\n"
            "      tags h_pre_norm and llama_main_graph_h_pre_norm\n"
            "      lookup API exists.\n");

    return 77;
}
