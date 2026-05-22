// llama-paged-kv-trace.h
//
// T5.0 — stub header for the paged KV NDJSON trace producer.
//
// The trace producer emits per-tick BlockAllocEvent records to an
// NDJSON stream when LLAMA_T5_TRACE=1 is set. Records are consumed by
// scripts/validate-paged-allocator-trace.py to verify the four
// allocator invariants over a real session:
//
//   BlockUniquelyOwned, FreeListDisjoint, AllocLazy, DefragPreservesOwnership.
//
// Per PHASE_NSTREAM_KV_PERF.md §"Trace producer + validator":
//   - Producer wired in T5.4 (allocator + WRITE path land) — until then,
//     the entry points below are no-ops.
//   - Validator script lands at T5.0 in
//     /home/llm/yarn-agentic/scripts/validate-paged-allocator-trace.py
//     and is gate-driven at T5.4 + T5.8 closure.
//
// Per [[feedback_bake_measurement_env_gates]]: LLAMA_T5_TRACE is a
// measurement-only knob. It MUST be removed in the same commit that
// bakes the verified Tier 5 behaviour (T5.8 closure).

#ifndef LLAMA_PAGED_KV_TRACE_H
#define LLAMA_PAGED_KV_TRACE_H

#ifdef __cplusplus
extern "C" {
#endif

// Op kinds for the trace event.
typedef enum llama_paged_kv_trace_op {
    LLAMA_PAGED_KV_TRACE_ALLOC       = 0,
    LLAMA_PAGED_KV_TRACE_FREE        = 1,
    LLAMA_PAGED_KV_TRACE_DEFRAG_MOVE = 2,
} llama_paged_kv_trace_op_t;

// Initialise the trace producer. Called once at llama_new_context_with_model
// when LLAMA_T5_TRACE=1; otherwise no-op.
//
// Returns 0 on success, non-zero on failure to open the trace stream.
//
// Stub (T5.0): always returns 0; emits nothing.
// Implementation lands at T5.4.
int llama_paged_kv_trace_init(void);

// Emit one BlockAllocEvent. Called from the allocator's alloc / free /
// defrag-move sites in T5.4.
//
// tick: monotonic per-decode counter (0 at session start).
// seq: seq id this op applies to (-1 for global ops like defrag start).
// block_id: physical block id touched by this op.
// op: alloc / free / defrag_move.
// prev_block_id: only meaningful for DEFRAG_MOVE; the previous block_id
//                of the seq's logical position before the move.
//
// Stub (T5.0): no-op. Implementation lands at T5.4.
void llama_paged_kv_trace_event(
    int tick,
    int seq,
    int block_id,
    llama_paged_kv_trace_op_t op,
    int prev_block_id);

// Flush + close the trace stream. Called at context destroy when
// LLAMA_T5_TRACE=1.
//
// Stub (T5.0): no-op.
void llama_paged_kv_trace_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif  // LLAMA_PAGED_KV_TRACE_H
