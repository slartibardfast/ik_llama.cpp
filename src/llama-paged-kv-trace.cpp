// llama-paged-kv-trace.cpp
//
// Paged KV NDJSON trace producer.
//
// T5.8 closure (2026-05-23) bake-out per [[feedback_bake_measurement_env_gates]]:
// the LLAMA_T5_TRACE runtime env-gate has been removed. This file is
// compile-time-gated by LLAMA_T5_TRACE_BUILD. In production builds the
// header provides inline no-op stubs; this TU is only meaningful in
// developer builds that set -DLLAMA_T5_TRACE_BUILD=1.

#include "llama-paged-kv-trace.h"

#ifdef LLAMA_T5_TRACE_BUILD

#include <atomic>
#include <cstdio>
#include <cstdlib>
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
    std::lock_guard<std::mutex> lk(g_trace_mu);
    if (g_trace_fp) {
        return 0;  // idempotent.
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
    if (!g_trace_fp) return;

    const int event_id = g_trace_event_id.fetch_add(1);

    std::lock_guard<std::mutex> lk(g_trace_mu);
    if (!g_trace_fp) return;
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

#endif  // LLAMA_T5_TRACE_BUILD
