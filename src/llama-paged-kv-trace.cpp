// llama-paged-kv-trace.cpp
//
// T5.3 — paged KV NDJSON trace producer implementation.
//
// Producer is gated on the LLAMA_T5_TRACE environment variable. When
// LLAMA_T5_TRACE=1 at process start, _init opens the trace file
// (LLAMA_T5_TRACE_PATH or default ./trace-paged-kv.ndjson) and every
// allocator alloc/free emits one BlockAllocEvent line. When unset or
// not "1", every entry point is a no-op and the allocator pays zero
// per-event cost.
//
// Per [[feedback_bake_measurement_env_gates]]: this env knob is a
// measurement-only artefact. T5.8 closure REMOVES the env gate (and
// this file, or compiles it out behind a build flag); the validator
// remains because it captures the binding behaviour.

#include "llama-paged-kv-trace.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

std::FILE *      g_trace_fp       = nullptr;
std::mutex       g_trace_mu;
std::atomic<int> g_trace_event_id{0};

const char * op_name(llama_paged_kv_trace_op_t op) {
    switch (op) {
        case LLAMA_PAGED_KV_TRACE_ALLOC:       return "ALLOC";
        case LLAMA_PAGED_KV_TRACE_FREE:        return "FREE";
        case LLAMA_PAGED_KV_TRACE_DEFRAG_MOVE: return "DEFRAG_MOVE";
    }
    return "UNKNOWN";
}

}  // namespace

extern "C" int llama_paged_kv_trace_init(void) {
    const char * enabled = std::getenv("LLAMA_T5_TRACE");
    if (!enabled || std::strcmp(enabled, "1") != 0) {
        // Disabled — leave g_trace_fp == nullptr; events are no-ops.
        return 0;
    }
    std::lock_guard<std::mutex> lk(g_trace_mu);
    if (g_trace_fp) {
        // Already initialised. Idempotent.
        return 0;
    }
    const char * path = std::getenv("LLAMA_T5_TRACE_PATH");
    if (!path || path[0] == '\0') path = "trace-paged-kv.ndjson";
    g_trace_fp = std::fopen(path, "w");
    if (!g_trace_fp) {
        std::fprintf(stderr,
                     "llama_paged_kv_trace_init: failed to open '%s'\n",
                     path);
        return 1;
    }
    g_trace_event_id.store(0);
    return 0;
}

extern "C" void llama_paged_kv_trace_event(
        int tick,
        int seq,
        int block_id,
        llama_paged_kv_trace_op_t op,
        int prev_block_id) {
    // Fast path: no producer initialised → no-op.
    if (!g_trace_fp) return;

    // Allocate event id outside the lock (atomic).
    const int event_id = g_trace_event_id.fetch_add(1);

    std::lock_guard<std::mutex> lk(g_trace_mu);
    if (!g_trace_fp) return;  // re-check after lock acquisition.
    if (op == LLAMA_PAGED_KV_TRACE_DEFRAG_MOVE) {
        std::fprintf(g_trace_fp,
                     "{\"tick\":%d,\"event_id\":%d,\"seq\":%d,"
                     "\"block_id\":%d,\"op\":\"%s\",\"prev_block_id\":%d}\n",
                     tick, event_id, seq, block_id,
                     op_name(op), prev_block_id);
    } else {
        std::fprintf(g_trace_fp,
                     "{\"tick\":%d,\"event_id\":%d,\"seq\":%d,"
                     "\"block_id\":%d,\"op\":\"%s\"}\n",
                     tick, event_id, seq, block_id, op_name(op));
    }
}

extern "C" void llama_paged_kv_trace_shutdown(void) {
    std::lock_guard<std::mutex> lk(g_trace_mu);
    if (g_trace_fp) {
        std::fflush(g_trace_fp);
        std::fclose(g_trace_fp);
        g_trace_fp = nullptr;
    }
}
