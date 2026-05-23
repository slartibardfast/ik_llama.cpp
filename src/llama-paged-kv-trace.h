// llama-paged-kv-trace.h
//
// Paged KV NDJSON trace producer header.
//
// T5.8 closure (2026-05-23) bake-out per [[feedback_bake_measurement_env_gates]]:
// the LLAMA_T5_TRACE runtime env-gate has been removed. The producer is
// now compile-time-gated by the LLAMA_T5_TRACE_BUILD macro:
//
//   - Default build (production): LLAMA_T5_TRACE_BUILD undefined →
//     trace_event / trace_init / trace_shutdown are inline no-ops, zero
//     runtime cost, no env-var check.
//   - Developer build for trace validation:
//     -DLLAMA_T5_TRACE_BUILD=1 enables the producer (opens a file at
//     trace_init based on LLAMA_T5_TRACE_PATH or default ./trace-paged-kv.ndjson;
//     trace_event writes one NDJSON record per allocator op).
//
// Records are consumed by scripts/validate-paged-allocator-trace.py to
// verify the four allocator invariants over a real session:
//
//   BlockUniquelyOwned, FreeListDisjoint, AllocLazy, DefragPreservesOwnership.

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

#ifdef LLAMA_T5_TRACE_BUILD

// Initialise the trace producer. Opens the trace file at
// LLAMA_T5_TRACE_PATH (default: ./trace-paged-kv.ndjson). Returns 0 on
// success, non-zero on failure.
int llama_paged_kv_trace_init(void);

// Emit one BlockAllocEvent.
//
// tick: monotonic per-decode counter (0 at session start).
// seq: seq id this op applies to (-1 for global ops like defrag start).
// block_id: physical block id touched by this op.
// op: alloc / free / defrag_move.
// prev_block_id: only meaningful for DEFRAG_MOVE.
void llama_paged_kv_trace_event(
    int tick,
    int seq,
    int block_id,
    llama_paged_kv_trace_op_t op,
    int prev_block_id);

// Flush + close the trace stream.
void llama_paged_kv_trace_shutdown(void);

#else  // LLAMA_T5_TRACE_BUILD undefined — production path, no-ops.

static inline int  llama_paged_kv_trace_init(void) { return 0; }
static inline void llama_paged_kv_trace_event(int, int, int, llama_paged_kv_trace_op_t, int) {}
static inline void llama_paged_kv_trace_shutdown(void) {}

#endif  // LLAMA_T5_TRACE_BUILD

#ifdef __cplusplus
}
#endif

#endif  // LLAMA_PAGED_KV_TRACE_H
