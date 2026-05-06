// test-accept-bonus-position.cpp
//
// Drives:
//   - mtp_verify_accept.allium BonusPosIsSeedPlusNAcceptedPlusOne
//
// Property: bonus_pos = pos_seed + n_accepted + 1. The bonus token
// occupies the position immediately after the last accepted draft
// (or after the seed, if zero drafts accepted). This is the
// position that becomes the next round's pos_seed + 0.
//
// Test strategy:
//   1. Vary pos_seed over a few values (0, 17, 4096).
//   2. Vary n_accepted over [0, n_drafts] inclusive (controlled via
//      synthesised draft_tokens vs argmax indices).
//   3. Assert decision.bonus_pos == pos_seed + decision.n_accepted + 1
//      in every case.

#include "llama.h"

#include <cstdio>

int main() {
    // TODO(spec-driven, RED until implementation lands).
    fprintf(stderr,
            "TODO: implement BonusPosIsSeedPlusNAcceptedPlusOne test once\n"
            "      llama_mtp_accept_verify is callable with stubbed logits.\n");
    return 77;
}
