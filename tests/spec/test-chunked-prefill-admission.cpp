// test-chunked-prefill-admission.cpp
//
// Property test for the Tier 4 chunked-prefill admission contracts,
// derived from
// /home/llm/yarn-agentic/specs/scheduler/batch_composition.allium and
// /home/llm/yarn-agentic/specs/multislot/BatchComposition.tla
// (post-T4 rewrite, 2026-05-22).
//
// Drives a stub admission function that mirrors the production loop
// landing in T4.2 / T4.4. Asserts the four new contracts:
//
//   1. TokenBudgetPerUbatch — total tokens admitted per tick <= K.
//   2. DecodePriorityAdmission — if any prefill admitted, every
//      decode-eligible slot is also in batch_decodes.
//   3. ChunkedPrefillAdmission — per-slot admitted count is bounded
//      by min(residual_budget, remaining_prefill); slot persists in
//      LOAD_PROMPT while remaining > 0.
//   4. PrefillCarryProgressesMonotonically — across consecutive
//      ticks, a slot's n_past_prompt is non-decreasing and reaches
//      n_prompt_tokens in finitely many ticks.
//
// This test is a STUB property test (no server, no model, no GGUF).
// The end-to-end server behaviour is bound by the trace validator
// (scripts/validate-batch-composition-trace.py) under T4.6's gate
// sweep on a real llama-server NDJSON dump. This file binds the
// algorithmic shape of the admission loop independently of the
// runtime path it sits in.
//
// Returns: 0 = PASS, 1 = FAIL.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

namespace {

// ============================================================
// Mini-types mirroring server-context.h's enums and the
// batch_composition.allium external entities.
// ============================================================

enum class SlotState   { IDLE, PROCESSING };
enum class SlotCommand { NONE, LOAD_PROMPT, RELEASE };

struct StubSlot {
    int          id;
    SlotState    state;
    SlotCommand  command;
    int          n_prompt_tokens;      // total prompt length
    int          n_past_prompt;        // tokens admitted so far (carry)
    bool         pending_decode;       // PROCESSING slots only
};

// Per-tick admission outcome.
struct AdmissionOutcome {
    // per-slot prefill chunk count this tick. Slots absent from this
    // map admitted 0 prefill.
    std::map<int, int> prefill_counts;
    // PROCESSING slots that contributed a decode token this tick.
    std::vector<int>   decode_slots;
    // Budget K used.
    int                budget_k;
};

// ============================================================
// Stub admission loop — mirrors the production loop that lands in
// T4.2/T4.4. Decode-priority + round-robin chunked prefill.
// ============================================================

AdmissionOutcome compose_t4_batch(std::vector<StubSlot>& slots, int K) {
    AdmissionOutcome out;
    out.budget_k = K;
    int remaining = K;

    // Phase 1 — admit decode tokens from every decode-eligible slot.
    for (const auto& s : slots) {
        if (s.state == SlotState::PROCESSING && s.pending_decode) {
            if (remaining == 0) break;  // budget exhausted
            out.decode_slots.push_back(s.id);
            --remaining;
        }
    }

    // Phase 2 — round-robin admit prefill chunks from LOAD_PROMPT slots.
    // Each round visits LOAD_PROMPT slots in id order and admits up to
    // (remaining // n_active_load_prompt) tokens, then re-evaluates.
    // For the stub we use a simpler equal-share single-round split
    // followed by greedy fill — equivalent behaviour for typical cases.
    std::vector<int> lp_ids;
    for (const auto& s : slots) {
        if (s.command == SlotCommand::LOAD_PROMPT
            && s.n_past_prompt < s.n_prompt_tokens) {
            lp_ids.push_back(s.id);
        }
    }
    while (remaining > 0 && !lp_ids.empty()) {
        bool any_admitted_this_round = false;
        for (int sid : lp_ids) {
            if (remaining == 0) break;
            StubSlot* s = nullptr;
            for (auto& t : slots) if (t.id == sid) { s = &t; break; }
            if (!s) continue;
            int slot_remaining = s->n_prompt_tokens - s->n_past_prompt
                                 - out.prefill_counts[sid];
            if (slot_remaining <= 0) continue;
            int take = (slot_remaining < remaining) ? slot_remaining : remaining;
            // Cap per-round per-slot at a small chunk for fairness;
            // for the stub we cap at 16 tokens per round (production
            // tunes via K only). The cap is internal to the stub and
            // does not appear in the spec.
            if (take > 16) take = 16;
            out.prefill_counts[sid] += take;
            remaining -= take;
            any_admitted_this_round = true;
        }
        if (!any_admitted_this_round) break;
    }

    return out;
}

// Apply admission outcome to slots: advance n_past_prompt; clear
// pending_decode. Returns the post-tick slot vector.
void apply_outcome(std::vector<StubSlot>& slots, const AdmissionOutcome& out) {
    for (auto& s : slots) {
        auto it = out.prefill_counts.find(s.id);
        if (it != out.prefill_counts.end()) {
            s.n_past_prompt += it->second;
        }
        for (int d : out.decode_slots) {
            if (d == s.id) s.pending_decode = false;
        }
    }
}

// Transition slots that finished prefill: LOAD_PROMPT -> NONE,
// IDLE -> PROCESSING, arm pending_decode.
void complete_prefills(std::vector<StubSlot>& slots) {
    for (auto& s : slots) {
        if (s.command == SlotCommand::LOAD_PROMPT
            && s.n_past_prompt == s.n_prompt_tokens
            && s.n_prompt_tokens > 0) {
            s.command = SlotCommand::NONE;
            s.state = SlotState::PROCESSING;
            s.pending_decode = true;
        }
    }
}

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s — %s\n", __FILE__, __LINE__, #cond, msg); \
        std::exit(1); \
    } \
} while (0)

// ============================================================
// Spec property assertions.
// ============================================================

int total_admitted(const AdmissionOutcome& o) {
    int total = 0;
    for (const auto& kv : o.prefill_counts) total += kv.second;
    total += (int)o.decode_slots.size();
    return total;
}

void assert_t4_contracts(const std::vector<StubSlot>& slots_before,
                          const AdmissionOutcome& o,
                          const char* label) {
    // (1) TokenBudgetPerUbatch.
    int total = total_admitted(o);
    ASSERT(total <= o.budget_k, label);

    // (2) PerTokenFlagExclusivity — no slot in both prefill and decode.
    for (const auto& kv : o.prefill_counts) {
        if (kv.second == 0) continue;
        for (int d : o.decode_slots) {
            ASSERT(d != kv.first, label);
        }
    }

    // (3) DecodePriorityAdmission — if any prefill admitted, every
    // decode-eligible slot is in batch_decodes.
    bool any_prefill = false;
    for (const auto& kv : o.prefill_counts) {
        if (kv.second > 0) { any_prefill = true; break; }
    }
    if (any_prefill) {
        for (const auto& s : slots_before) {
            if (s.state == SlotState::PROCESSING && s.pending_decode) {
                bool found = false;
                for (int d : o.decode_slots) if (d == s.id) { found = true; break; }
                ASSERT(found, label);
            }
        }
    }

    // (4) ChunkedPrefillAdmission per-slot bound.
    for (const auto& kv : o.prefill_counts) {
        const StubSlot* s = nullptr;
        for (const auto& t : slots_before) if (t.id == kv.first) { s = &t; break; }
        ASSERT(s != nullptr, label);
        int remaining = s->n_prompt_tokens - s->n_past_prompt;
        ASSERT(kv.second <= remaining, label);
        if (kv.second > 0) {
            ASSERT(s->command == SlotCommand::LOAD_PROMPT, label);
        }
    }
}

// ============================================================
// Test cases.
// ============================================================

// Case 1 — synthetic admission shape from the plan:
// 8 slots; 4 PROCESSING + 4 LOAD_PROMPT (n_prompt=128). K=64.
// Tick 0 emits 4 decode + 60 prefill spread across 4 LOAD_PROMPT
// slots. Subsequent ticks drain remaining prefill.
void test_synthetic_admission_shape() {
    std::vector<StubSlot> slots = {
        {0, SlotState::PROCESSING, SlotCommand::NONE,         0,   0, true},
        {1, SlotState::PROCESSING, SlotCommand::NONE,         0,   0, true},
        {2, SlotState::PROCESSING, SlotCommand::NONE,         0,   0, true},
        {3, SlotState::PROCESSING, SlotCommand::NONE,         0,   0, true},
        {4, SlotState::IDLE,       SlotCommand::LOAD_PROMPT, 128,  0, false},
        {5, SlotState::IDLE,       SlotCommand::LOAD_PROMPT, 128,  0, false},
        {6, SlotState::IDLE,       SlotCommand::LOAD_PROMPT, 128,  0, false},
        {7, SlotState::IDLE,       SlotCommand::LOAD_PROMPT, 128,  0, false},
    };
    const int K = 64;
    auto before = slots;
    auto out = compose_t4_batch(slots, K);
    assert_t4_contracts(before, out, "synthetic_admission_shape_tick0");
    ASSERT(out.decode_slots.size() == 4, "all 4 PROCESSING slots admitted");
    ASSERT(total_admitted(out) <= K, "budget bound");
    // Prefill should fill the remaining 60-token budget.
    int prefill_total = 0;
    for (const auto& kv : out.prefill_counts) prefill_total += kv.second;
    ASSERT(prefill_total == 60, "prefill fills residual budget exactly");
    apply_outcome(slots, out);

    // Drain remaining prefill in subsequent ticks. Decode slots have
    // pending_decode cleared; we don't re-arm in this test (the
    // production server re-arms via ProduceSample).
    int max_ticks = 64;  // safety bound
    while (max_ticks-- > 0) {
        bool more_prefill = false;
        for (const auto& s : slots) {
            if (s.command == SlotCommand::LOAD_PROMPT
                && s.n_past_prompt < s.n_prompt_tokens) {
                more_prefill = true;
                break;
            }
        }
        if (!more_prefill) break;
        auto before2 = slots;
        auto out2 = compose_t4_batch(slots, K);
        assert_t4_contracts(before2, out2, "synthetic_admission_shape_drain");
        apply_outcome(slots, out2);
        complete_prefills(slots);
    }
    for (const auto& s : slots) {
        if (s.id >= 4) {
            ASSERT(s.n_past_prompt == s.n_prompt_tokens, "prefill complete");
            ASSERT(s.state == SlotState::PROCESSING, "transitioned to PROCESSING");
        }
    }
}

// Case 2 — single-slot prefill larger than budget.
// 1 slot, n_prompt=512, K=64. Eight consecutive ticks each admit
// 64 prefill tokens; n_past_prompt monotone {64, 128, ..., 512}.
void test_single_slot_prefill_larger_than_budget() {
    std::vector<StubSlot> slots = {
        {0, SlotState::IDLE, SlotCommand::LOAD_PROMPT, 512, 0, false},
    };
    const int K = 64;
    std::vector<int> n_past_history;
    n_past_history.push_back(slots[0].n_past_prompt);
    for (int t = 0; t < 16; ++t) {
        auto before = slots;
        auto out = compose_t4_batch(slots, K);
        assert_t4_contracts(before, out, "single_slot_512_K64");
        ASSERT(out.decode_slots.empty(), "no decode (single LOAD_PROMPT only)");
        apply_outcome(slots, out);
        complete_prefills(slots);
        n_past_history.push_back(slots[0].n_past_prompt);
        if (slots[0].state == SlotState::PROCESSING) break;
    }
    // Monotonic non-decrease.
    for (size_t i = 1; i < n_past_history.size(); ++i) {
        ASSERT(n_past_history[i] >= n_past_history[i-1],
               "n_past_prompt monotone");
    }
    ASSERT(slots[0].state == SlotState::PROCESSING,
           "completed in finite ticks");
}

// Case 3 — mixed-state regression sweep. Vary slot counts/states and
// assert spec invariants over many configurations.
void test_mixed_state_regression_sweep() {
    auto each_K = { 1, 2, 4, 8, 16, 64, 512 };
    auto each_n_prompt = { 0, 1, 32, 128, 1024 };
    auto each_n_past = { 0, 16, 64, 128 };
    int configs_tested = 0;
    for (int K : each_K) {
        for (int n_prompt_a : each_n_prompt) {
            for (int n_prompt_b : each_n_prompt) {
                for (int n_past_a : each_n_past) {
                    if (n_past_a > n_prompt_a) continue;
                    std::vector<StubSlot> slots = {
                        {0, SlotState::IDLE,       SlotCommand::LOAD_PROMPT,
                         n_prompt_a, n_past_a, false},
                        {1, SlotState::IDLE,       SlotCommand::LOAD_PROMPT,
                         n_prompt_b, 0,        false},
                        {2, SlotState::PROCESSING, SlotCommand::NONE,
                         0,          0,        true},
                        {3, SlotState::PROCESSING, SlotCommand::NONE,
                         0,          0,        false},
                    };
                    auto before = slots;
                    auto out = compose_t4_batch(slots, K);
                    assert_t4_contracts(before, out,
                                        "mixed_state_regression_sweep");
                    ++configs_tested;
                }
            }
        }
    }
    std::fprintf(stdout, "swept %d slot configurations\n", configs_tested);
}

// Case 4 — DecodePriorityAdmission negative sanity: a tick with
// only a budget-exhausting prefill must NOT include decodes WHEN no
// prefill is admitted (i.e., decode-priority is order-dependent in
// admission, not unconditional in batch shape).
// Specifically: when there's PROCESSING+pending_decode slots AND
// LOAD_PROMPT slots, and budget K=2, the 2 decodes are admitted
// first; no budget remains for prefill; this is decode-only-mixed.
void test_decode_priority_budget_full() {
    std::vector<StubSlot> slots = {
        {0, SlotState::PROCESSING, SlotCommand::NONE,        0, 0,  true},
        {1, SlotState::PROCESSING, SlotCommand::NONE,        0, 0,  true},
        {2, SlotState::IDLE,       SlotCommand::LOAD_PROMPT, 50, 0, false},
    };
    const int K = 2;
    auto before = slots;
    auto out = compose_t4_batch(slots, K);
    assert_t4_contracts(before, out, "decode_priority_budget_full");
    ASSERT(out.decode_slots.size() == 2,
           "both decodes admitted, no room for prefill");
    int prefill_total = 0;
    for (const auto& kv : out.prefill_counts) prefill_total += kv.second;
    ASSERT(prefill_total == 0,
           "prefill admitted nothing — decode priority + budget exhaustion");
}

}  // namespace

int main() {
    test_synthetic_admission_shape();
    test_single_slot_prefill_larger_than_budget();
    test_mixed_state_regression_sweep();
    test_decode_priority_budget_full();
    std::fprintf(stdout, "test-chunked-prefill-admission: PASS\n");
    return 0;
}
