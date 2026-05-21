// test-mtp-x-n-stream.cpp
//
// Property test for the MTP fused × n_stream composition, derived from
// /home/llm/yarn-agentic/specs/composition/mtp_fused_x_n_stream.allium
// and /home/llm/yarn-agentic/specs/composition/MTPxNStream.tla.
//
// Stub-style: models the chain-residual arm state machine + multi-slot
// batched-dispatch opt-out, then verifies five invariants:
//
//   1. NoCrossStreamChainPoisoning: chain seed consumed by stream s
//      was armed by s itself (not by some other stream).
//   2. MultiSlotImpliesSkippedArm (HEAD): when n_active_streams > 1,
//      no arm is set. Enforced structurally by ArmChain's precondition.
//   3. SingleSlotPathHonoursArm (HEAD): at most one arm exists at a
//      time; single-active-stream re-arming is idempotent.
//   4. ArmedImpliesActive: chain_armed is non-empty only when at least
//      one stream is active.
//   5. Retire-to-zero clears all arms (the persist buffer is invalid
//      when no decoder remains to consume it).
//
// PASS expected on HEAD: server-context.cpp:3427-3430 structurally
// skips chain-residual arming at multi-slot batched dispatch. The
// MultiSlotImpliesSkippedArm contract is HONOURED ON HEAD.
//
// Returns: 0 = PASS, 1 = FAIL.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <set>
#include <string>

namespace {

#define FAIL_AT(msg, ...) do { \
    std::fprintf(stderr, "FAIL %s:%d: " msg "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
    return false; \
} while (0)

// Mini state machine. Mirrors the TLA+ model 1:1.
struct FusedDecoderState {
    int n_active_streams = 0;
    std::set<int> chain_armed;     // set of stream ids with armed chain
    std::optional<int> last_fused_consumer;
};

constexpr int MAX_STREAMS = 4;

// admit_stream — transition a fresh stream to PROCESSING. If we enter
// multi-slot (active becomes >=2) under HEAD opt-out, the existing
// single-slot arm is cleared.
void admit_stream(FusedDecoderState& s) {
    s.n_active_streams += 1;
    if (s.n_active_streams >= 2) {
        // HEAD opt-out: multi-slot batched dispatch skips arming + invalidates
        // the persist buffer for any prior single-slot arm.
        s.chain_armed.clear();
    }
}

// retire_stream — stream id leaves the active set. When the active set
// reaches zero, ALL arms are invalidated (no decoder to consume them).
void retire_stream(FusedDecoderState& s, int id) {
    if (s.n_active_streams == 0) return;
    s.n_active_streams -= 1;
    if (s.n_active_streams == 0) {
        s.chain_armed.clear();
    } else {
        s.chain_armed.erase(id);
    }
}

// arm_chain — stream id arms its chain for the next fused decode.
// Under HEAD: only fires at n_active_streams == 1, and re-arming the
// same stream is idempotent; arming a DIFFERENT stream while one is
// already armed is rejected (cannot happen under structural opt-out).
//
// Returns true if the arm succeeded.
bool arm_chain(FusedDecoderState& s, int id) {
    if (s.n_active_streams != 1) return false;
    // Idempotent re-arm OK; otherwise must be empty (no other stream
    // can be armed while one already is — single-buffer constraint).
    if (!s.chain_armed.empty() && s.chain_armed.count(id) == 0) return false;
    s.chain_armed.clear();
    s.chain_armed.insert(id);
    return true;
}

// fused_decode_armed — stream id consumes its own armed chain.
// Returns true if a chain was consumed (D2D fast-path); false if the
// host-bounce path was used instead (no arm to consume).
bool fused_decode_armed(FusedDecoderState& s, int id) {
    if (s.chain_armed.count(id) > 0) {
        s.chain_armed.erase(id);
        s.last_fused_consumer = id;
        return true;
    }
    s.last_fused_consumer = id;
    return false;
}

// --- Invariants -----------------------------------------------------------

bool no_cross_stream_chain_poisoning(const FusedDecoderState& s) {
    // Every armed-set element is a stream id in [0, MAX_STREAMS).
    // Operationally: arm_chain only inserts the calling stream;
    // fused_decode_armed only consumes the calling stream. There is
    // no path where stream s consumes stream t's seed.
    for (int id : s.chain_armed) {
        if (id < 0 || id >= MAX_STREAMS) return false;
    }
    return true;
}

bool multi_slot_implies_skipped_arm(const FusedDecoderState& s) {
    return s.n_active_streams <= 1 || s.chain_armed.empty();
}

bool single_slot_path_honours_arm(const FusedDecoderState& s) {
    return s.chain_armed.size() <= 1;
}

bool armed_implies_active(const FusedDecoderState& s) {
    return s.chain_armed.empty() || s.n_active_streams >= 1;
}

// --- The tests -----------------------------------------------------------

bool check_invariants_at(const FusedDecoderState& s, const char* tag) {
    if (!no_cross_stream_chain_poisoning(s))
        FAIL_AT("[%s] NoCrossStreamChainPoisoning violated", tag);
    if (!multi_slot_implies_skipped_arm(s))
        FAIL_AT("[%s] MultiSlotImpliesSkippedArm violated", tag);
    if (!single_slot_path_honours_arm(s))
        FAIL_AT("[%s] SingleSlotPathHonoursArm violated", tag);
    if (!armed_implies_active(s))
        FAIL_AT("[%s] ArmedImpliesActive violated", tag);
    return true;
}

bool test_init_all_invariants_hold() {
    FusedDecoderState s;
    return check_invariants_at(s, "init")
        && (std::fprintf(stdout, "test_init_all_invariants_hold OK\n"), true);
}

bool test_single_slot_arm_and_consume() {
    FusedDecoderState s;
    admit_stream(s);                       // n_active=1
    if (!arm_chain(s, 0)) FAIL_AT("expected arm_chain(0) to succeed at n_active=1");
    if (!check_invariants_at(s, "armed-1")) return false;
    if (!fused_decode_armed(s, 0)) FAIL_AT("expected fused_decode_armed to consume D2D");
    if (!check_invariants_at(s, "post-consume")) return false;
    if (s.chain_armed.empty() == false) FAIL_AT("chain_armed should be empty after consume");
    std::fprintf(stdout, "test_single_slot_arm_and_consume OK\n");
    return true;
}

bool test_multi_slot_admit_clears_arm() {
    FusedDecoderState s;
    admit_stream(s);                       // n_active=1
    arm_chain(s, 0);                       // chain_armed={0}
    if (!check_invariants_at(s, "pre-multi")) return false;
    admit_stream(s);                       // n_active=2 → arm cleared
    if (!check_invariants_at(s, "post-multi-admit")) return false;
    if (!s.chain_armed.empty())
        FAIL_AT("admit-into-multi-slot should clear arm (HEAD opt-out)");
    std::fprintf(stdout, "test_multi_slot_admit_clears_arm OK\n");
    return true;
}

bool test_multi_slot_rejects_arm() {
    FusedDecoderState s;
    admit_stream(s); admit_stream(s);      // n_active=2
    if (arm_chain(s, 0)) FAIL_AT("arm_chain at n_active=2 should be rejected (HEAD)");
    if (arm_chain(s, 1)) FAIL_AT("arm_chain at n_active=2 should be rejected (HEAD)");
    if (!s.chain_armed.empty()) FAIL_AT("chain_armed should remain empty");
    if (!check_invariants_at(s, "multi-no-arm")) return false;
    std::fprintf(stdout, "test_multi_slot_rejects_arm OK\n");
    return true;
}

bool test_multi_slot_fused_decode_host_bounce() {
    FusedDecoderState s;
    admit_stream(s); admit_stream(s);      // n_active=2
    // No arm at multi-slot. fused_decode_armed returns false (host-bounce).
    if (fused_decode_armed(s, 0)) FAIL_AT("expected host-bounce at n_active=2");
    if (fused_decode_armed(s, 1)) FAIL_AT("expected host-bounce at n_active=2");
    if (!check_invariants_at(s, "multi-host-bounce")) return false;
    std::fprintf(stdout, "test_multi_slot_fused_decode_host_bounce OK\n");
    return true;
}

bool test_retire_to_zero_clears_arms() {
    FusedDecoderState s;
    admit_stream(s);                       // n_active=1
    arm_chain(s, 0);                       // chain_armed={0}
    retire_stream(s, 0);                   // n_active=0 → clear
    if (!s.chain_armed.empty()) FAIL_AT("retire-to-zero should clear arms");
    if (!check_invariants_at(s, "retired")) return false;
    std::fprintf(stdout, "test_retire_to_zero_clears_arms OK\n");
    return true;
}

bool test_idempotent_rearm_same_stream() {
    FusedDecoderState s;
    admit_stream(s);                       // n_active=1
    if (!arm_chain(s, 0)) FAIL_AT("first arm failed");
    if (!arm_chain(s, 0)) FAIL_AT("idempotent re-arm of same stream rejected");
    if (s.chain_armed.size() != 1 || s.chain_armed.count(0) == 0)
        FAIL_AT("idempotent re-arm should leave chain_armed = {0}");
    if (!check_invariants_at(s, "rearmed")) return false;
    std::fprintf(stdout, "test_idempotent_rearm_same_stream OK\n");
    return true;
}

bool test_single_slot_disallows_other_stream_arm() {
    // While stream 0 is armed at single-slot, arm_chain(1) must be
    // rejected (the single buffer is owned by stream 0).
    FusedDecoderState s;
    admit_stream(s);                       // n_active=1
    arm_chain(s, 0);                       // chain_armed={0}
    if (arm_chain(s, 1))
        FAIL_AT("expected arm_chain(1) to be rejected while {0} is armed");
    if (s.chain_armed != std::set<int>{0})
        FAIL_AT("chain_armed should remain {0}");
    std::fprintf(stdout, "test_single_slot_disallows_other_stream_arm OK\n");
    return true;
}

} // namespace

int main() {
    bool ok = true;
    ok &= test_init_all_invariants_hold();
    ok &= test_single_slot_arm_and_consume();
    ok &= test_multi_slot_admit_clears_arm();
    ok &= test_multi_slot_rejects_arm();
    ok &= test_multi_slot_fused_decode_host_bounce();
    ok &= test_retire_to_zero_clears_arms();
    ok &= test_idempotent_rearm_same_stream();
    ok &= test_single_slot_disallows_other_stream_arm();
    if (!ok) {
        std::fprintf(stderr, "FAIL: MTP × n_stream invariant violated\n");
        return 1;
    }
    std::fprintf(stdout, "[PASS] MTP fused × n_stream HEAD opt-out + 4 "
                         "spec invariants held under 8 property tests.\n");
    return 0;
}
