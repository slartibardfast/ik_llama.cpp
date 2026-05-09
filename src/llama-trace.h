// PHASE46 B.4: NDJSON trace emission for determinism replay.
//
// Buffered writer; one JSON object per line. File path is taken from
// $LLAMA_TRACE_NDJSON at first call (read once + cached). When the env is
// unset, llama_trace_emit() is a no-op and llama_trace_enabled() returns
// false. The output stream is line-buffered so partial traces are readable
// during long runs and flushed on process shutdown.
//
// Event codes are stable wire identifiers — do not renumber. The replay
// tool (tools/replay-trace.cpp) and slot-permutation harness
// (tools/permute-slots.cpp) both consume this format.
#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define LLAMA_TRACE_EV_FORK_DRAFT   0
#define LLAMA_TRACE_EV_JOIN_DRAFT   1
#define LLAMA_TRACE_EV_FORK_VERIFY  2
#define LLAMA_TRACE_EV_JOIN_VERIFY  3
#define LLAMA_TRACE_EV_ACCEPT       4
#define LLAMA_TRACE_EV_REJECT       5

bool llama_trace_enabled(void);

// Emit one NDJSON line. step/slot_id/pos identify the event in time + lane;
// n_drafted and n_accepted carry per-step counters (use -1 to omit).
void llama_trace_emit(int event_code,
                      int     slot_id,
                      int64_t step,
                      int64_t pos,
                      int     n_drafted,
                      int     n_accepted);

// Force flush — for tests that want to read the trace mid-run.
void llama_trace_flush(void);

#ifdef __cplusplus
}
#endif
