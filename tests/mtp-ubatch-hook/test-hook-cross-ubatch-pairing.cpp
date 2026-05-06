// test-hook-cross-ubatch-pairing.cpp
//
// Drives:
//   - mtp_ubatch_hook.allium CrossUbatchPairingOnContinuation
//   - mtp_ubatch_hook.allium CrossUbatchResetOnDiscontinuation
//
// When ubatch B follows ubatch A in the same sequence with
// B.pos_first = A.pos_last + 1, the hook's first MTP-block input
// for B is the pair (A's last hidden row, B's first token). When B
// does not continue A (different sequence or position gap), the
// pending state from A is discarded and no spurious MTP write is
// emitted at the boundary.

#include "llama.h"

#include <cassert>
#include <cstdio>

int main() {
    // TODO(spec-driven, RED until implementation lands):
    //
    // The implementation needs an inspection point that lets the test
    // observe the hook's pending state. Options:
    //   (a) llama_mtp_hook_pending_pos(ctx) returning the cached
    //       pending_pos (or -1 if discarded);
    //   (b) inspecting the first row of the next hook's input batch
    //       directly via a debug callback.
    // Pick (a) — it's a thin observability hook and fits the existing
    // pattern.
    //
    // CONTINUATION CASE:
    //   1. Decode ubatch A: pos in [10, 19] (n_tokens=10), seq_id=0.
    //   2. After decode, pending_pos must equal 19.
    //   3. Decode ubatch B: pos in [20, 24] (n_tokens=5), seq_id=0.
    //   4. The hook's MTP input for B's first row must use A's last
    //      hidden + B's first token. Implementation detail covered
    //      by the lockstep test, but observable here as: after B's
    //      decode completes, MTP-KV at pos 20 must be populated. If
    //      the pairing failed, the hook would have skipped pos 20
    //      (no pre-token hidden available) and pos 20 would be
    //      missing.
    //   5. Assert llama_mtp_kv_has_pos(ctx, 0, 20) == true.
    //
    // DISCONTINUATION CASE:
    //   1. Decode ubatch A: pos in [10, 19] seq_id=0.
    //   2. After decode, pending_pos must equal 19.
    //   3. Decode ubatch C: pos in [50, 54] seq_id=0 (gap from A).
    //   4. Implementation must discard pending state because
    //      C.pos_first (50) != pending_pos+1 (20).
    //   5. Assert llama_mtp_hook_pending_pos(ctx) was reset to -1
    //      between A and C (or expose a "discard count" counter and
    //      assert it incremented).
    //   6. After C's decode, MTP-KV at pos 50 must be populated
    //      (the hook fires for C and uses an in-ubatch pairing for
    //      C's interior rows; the boundary row for pos 50 has no
    //      pre-token hidden, so spec OQ-2 says it's special-cased
    //      — verify the implementation's chosen handling matches
    //      the expectations established by the lockstep test).
    //
    // SEQUENCE-DIFFERENCE CASE:
    //   1. Decode ubatch A: pos in [10, 19] seq_id=0.
    //   2. Decode ubatch D: pos in [20, 24] seq_id=1 (different seq).
    //   3. Even though pos_first matches (20 = 19+1), seq_id differs
    //      — pairing must NOT happen. Pending state for seq 0 must
    //      be preserved; new state for seq 1 must start fresh.

    fprintf(stderr,
            "TODO: implement cross-ubatch pairing test once the hook,\n"
            "      pending_pos observability, and MTP-KV inspection\n"
            "      land.\n");

    return 77;
}
