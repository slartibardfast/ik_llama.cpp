// test-batch-composition-gates.cpp
//
// Property test for the Bug C closure scheduler gates, derived from
// /home/llm/yarn-agentic/specs/scheduler/batch_composition.allium and
// /home/llm/yarn-agentic/specs/multislot/BatchComposition.tla.
//
// Drives the gate logic as a stub — replicates the two server-context.cpp
// gate computations against synthetic slot state and asserts:
//
//   1. PrefillSerialisationGate (v1, batch_pending_prompt):
//      at most ONE slot in LOAD_PROMPT contributes per tick; continuation
//      slots win over new prompts.
//   2. DecodeHoldGate (Bug C fix, add_sampled_tokens):
//      if any slot is LOAD_PROMPT, no PROCESSING slot's decode is added
//      to this tick's batch.
//   3. MixedBatchProhibition: every tick's batch is either pure-prefill
//      or pure-decode — never mixed.
//
// The stub mirrors the implementation at
// ik_llama.cpp/examples/server/server-context.cpp:3194 (add_sampled_tokens)
// and ik_llama.cpp/examples/server/server-context.cpp:3626 (batch_pending_prompt).
// If the gate code in server-context.cpp diverges from this stub, the
// test will not catch it directly (it tests the SPEC); the live trace
// validation in S5 binds on the actual implementation.
//
// Returns: 0 = PASS, 1 = FAIL.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <optional>

namespace {

// Mini-types mirroring server-context.h's enums and fields.
enum class SlotState { IDLE, PROCESSING };
enum class SlotCommand { NONE, LOAD_PROMPT, RELEASE };

struct StubSlot {
    int id;
    SlotState state;
    SlotCommand command;
    int n_prompt_tokens;
    int n_prompt_tokens_processed;
    bool pending_decode;
};

// Mirror of PrefillSerialisationGate (active_pp_slot_id selection in
// batch_pending_prompt). Returns the id of the contributing slot, or
// std::nullopt when no prefill is selected.
std::optional<int> prefill_serialisation_gate(const std::vector<StubSlot>& slots) {
    std::optional<int> continuation;
    std::optional<int> first_load_prompt;
    for (const auto& s : slots) {
        if (s.command != SlotCommand::LOAD_PROMPT) continue;
        if (s.n_prompt_tokens_processed > 0) {
            if (!continuation.has_value()) continuation = s.id;
        } else {
            if (!first_load_prompt.has_value()) first_load_prompt = s.id;
        }
    }
    if (continuation.has_value()) return continuation;
    return first_load_prompt;
}

// Mirror of DecodeHoldGate (early-return in add_sampled_tokens). Returns
// true iff decode contributions MAY be added this tick.
bool decode_hold_gate_permits(const std::vector<StubSlot>& slots) {
    for (const auto& s : slots) {
        if (s.command == SlotCommand::LOAD_PROMPT) return false;
    }
    return true;
}

// Compose the batch under both gates. Returns (prefill_slots, decode_slots).
struct ComposedBatch {
    std::vector<int> prefill_slots;
    std::vector<int> decode_slots;
};

ComposedBatch compose_batch(const std::vector<StubSlot>& slots) {
    ComposedBatch b;
    auto prefill_id = prefill_serialisation_gate(slots);
    if (prefill_id.has_value()) b.prefill_slots.push_back(*prefill_id);
    if (decode_hold_gate_permits(slots)) {
        for (const auto& s : slots) {
            if (s.state == SlotState::PROCESSING && s.pending_decode) {
                b.decode_slots.push_back(s.id);
            }
        }
    }
    return b;
}

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s — %s\n", __FILE__, __LINE__, #cond, msg); \
        std::exit(1); \
    } \
} while (0)

// Property 1: AtMostOnePrefillSlotPerBatch.
void test_prefill_serialisation_singleton() {
    std::vector<StubSlot> slots = {
        {0, SlotState::IDLE, SlotCommand::LOAD_PROMPT, 100, 0, false},
        {1, SlotState::IDLE, SlotCommand::LOAD_PROMPT, 200, 0, false},
        {2, SlotState::IDLE, SlotCommand::LOAD_PROMPT, 300, 0, false},
    };
    auto b = compose_batch(slots);
    ASSERT(b.prefill_slots.size() <= 1, "AtMostOnePrefillSlotPerBatch");
    ASSERT(b.decode_slots.empty(), "DecodeHoldGate suppresses decodes here");
}

// Property 2: PrefillContinuationPriority.
void test_prefill_continuation_priority() {
    std::vector<StubSlot> slots = {
        {0, SlotState::IDLE, SlotCommand::LOAD_PROMPT, 100, 0, false},
        {1, SlotState::IDLE, SlotCommand::LOAD_PROMPT, 200, 50, false},  // continuation
        {2, SlotState::IDLE, SlotCommand::LOAD_PROMPT, 300, 0, false},
    };
    auto b = compose_batch(slots);
    ASSERT(b.prefill_slots.size() == 1, "exactly one prefill picked");
    ASSERT(b.prefill_slots[0] == 1, "continuation wins over fresh prompts");
}

// Property 3: DecodeHoldGate when any slot is LOAD_PROMPT.
void test_decode_held_when_prefill_pending() {
    std::vector<StubSlot> slots = {
        {0, SlotState::PROCESSING, SlotCommand::NONE,        0, 0, true},
        {1, SlotState::PROCESSING, SlotCommand::NONE,        0, 0, true},
        {2, SlotState::IDLE,       SlotCommand::LOAD_PROMPT, 50, 0, false},
    };
    auto b = compose_batch(slots);
    ASSERT(b.prefill_slots.size() == 1, "the new prompt fills the prefill seat");
    ASSERT(b.decode_slots.empty(), "DecodeHoldGate suppresses both decodes");
}

// Property 4: Pure-decode tick when no slot is LOAD_PROMPT.
void test_pure_decode_when_no_prefill() {
    std::vector<StubSlot> slots = {
        {0, SlotState::PROCESSING, SlotCommand::NONE, 0, 0, true},
        {1, SlotState::PROCESSING, SlotCommand::NONE, 0, 0, true},
        {2, SlotState::IDLE,       SlotCommand::NONE, 0, 0, false},
    };
    auto b = compose_batch(slots);
    ASSERT(b.prefill_slots.empty(), "no prefill candidates");
    ASSERT(b.decode_slots.size() == 2, "both PROCESSING slots contribute decode");
}

// Property 5: MixedBatchProhibition — the load-bearing invariant.
// Sweep a broader space of slot configurations and assert the composed
// batch is never mixed.
void test_mixed_batch_prohibition_sweep() {
    auto each_state    = { SlotState::IDLE, SlotState::PROCESSING };
    auto each_command  = { SlotCommand::NONE, SlotCommand::LOAD_PROMPT, SlotCommand::RELEASE };
    auto each_n_done   = { 0, 25, 100 };
    auto each_pending  = { false, true };

    int configs_tested = 0;
    for (auto s_a : each_state) {
      for (auto c_a : each_command) {
        for (auto n_a : each_n_done) {
          for (auto p_a : each_pending) {
            for (auto s_b : each_state) {
              for (auto c_b : each_command) {
                for (auto n_b : each_n_done) {
                  for (auto p_b : each_pending) {
                    std::vector<StubSlot> slots = {
                        {0, s_a, c_a, 100, n_a, p_a},
                        {1, s_b, c_b, 100, n_b, p_b},
                    };
                    auto batch = compose_batch(slots);
                    bool mixed = !batch.prefill_slots.empty()
                              && !batch.decode_slots.empty();
                    ASSERT(!mixed, "MixedBatchProhibition under sweep");
                    ++configs_tested;
                  }
                }
              }
            }
          }
        }
      }
    }
    std::fprintf(stdout, "swept %d slot configurations\n", configs_tested);
}

}  // namespace

int main() {
    test_prefill_serialisation_singleton();
    test_prefill_continuation_priority();
    test_decode_held_when_prefill_pending();
    test_pure_decode_when_no_prefill();
    test_mixed_batch_prohibition_sweep();
    std::fprintf(stdout, "test-batch-composition-gates: PASS\n");
    return 0;
}
