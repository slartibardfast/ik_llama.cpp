// test-mtp-fused-determinism.cpp
//
// Drives:
//   - mtp_fused_draft.allium DeterminismUnderArgmax (system-wide)
//
// Two invocations of FusedDraftStep with byte-identical inputs and
// identical model state must produce byte-identical token sequences
// when mode = argmax_only. Binds against non-deterministic atomic
// kernel ordering or RNG-state leakage.
//
// Test strategy:
//   1. Run invoke A with (seed_token=42, hidden=H, n_steps=4).
//   2. Reset MTP-layer KV (or use a fresh context).
//   3. Run invoke B with the same arguments.
//   4. Assert A.tokens[0..n_steps) == B.tokens[0..n_steps) elementwise.

#include "llama.h"

#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
    // TODO(spec-driven, RED until implementation lands):
    // 1. Load test fixture model (see test-mtp-fused-single-compute).
    // 2. Capture a reproducible seed_hidden by running one main
    //    forward over a fixed prompt and pulling the last hidden
    //    row to host.
    // 3. Run llama_mtp_fused_draft_invoke twice with byte-identical
    //    inputs. The second invoke must use a clean MTP-KV state
    //    matching the first invoke's pre-state (seq_rm to the
    //    pre-round position).
    // 4. memcmp(out_a.tokens, out_b.tokens, n_steps * sizeof(int32_t))
    //    must be 0.
    // 5. probs need not be byte-identical at the FP level (different
    //    kernel scheduling can produce different last-bit ULP); the
    //    invariant binds on tokens, not on probs.
    //
    // Note on multi-GPU: production runs split across CUDA0+CUDA1.
    // Cross-device kernel scheduling order is implementation-defined.
    // The argmax winner is the only deterministic property the spec
    // promises; this test must NOT assert byte-equality on probs.

    fprintf(stderr,
            "TODO: implement DeterminismUnderArgmax test once the fused\n"
            "      API and the test fixture are available.\n");

    return 77;
}
