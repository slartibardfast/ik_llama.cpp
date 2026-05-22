// server-trace-ndjson.h
//
// NDJSON trace emission for batch-composition + graph-cache invariants.
//
// Gated on LLAMA_TRACE_NDJSON_DIR. When unset, every call is a no-op
// branch in the hot path. When set, one NDJSON line is appended per
// emit call to <dir>/server-trace.ndjson.
//
// Schema is defined alongside the validator at
// /home/llm/yarn-agentic/scripts/validate-batch-composition-trace.py.
//
// Under T4 chunked-prefill admission this trace is the binding gate
// for the scheduler invariants (TokenBudgetRespected,
// DecodePriorityAdmission, PerTokenFlagExclusivity, PrefillCarryProgresses)
// defined in specs/scheduler/batch_composition.allium and
// specs/multislot/BatchComposition.tla. It is not a diagnostic-only
// scaffold — keep enabled in any verify run that exercises GP4.m.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace server_trace_ndjson {

inline const char* trace_dir_env() {
    static const char* val = std::getenv("LLAMA_TRACE_NDJSON_DIR");
    return val;
}

inline bool enabled() {
    const char* d = trace_dir_env();
    return d != nullptr && d[0] != '\0';
}

inline std::FILE* trace_file() {
    static std::FILE* fp = []() -> std::FILE* {
        const char* d = trace_dir_env();
        if (!d || d[0] == '\0') return nullptr;
        std::string path = std::string(d) + "/server-trace.ndjson";
        std::FILE* f = std::fopen(path.c_str(), "a");
        if (!f) {
            std::fprintf(stderr,
                "server-trace-ndjson: failed to open %s for append\n",
                path.c_str());
        }
        return f;
    }();
    return fp;
}

inline std::mutex& trace_mutex() {
    static std::mutex m;
    return m;
}

inline void emit_set(std::string& out, const std::vector<int>& xs) {
    out += "[";
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(xs[i]);
    }
    out += "]";
}

// Emit a TickDispatch record. Call once per server tick (one
// update_slots() iteration), AFTER the batch has been fully composed
// (add_sampled_tokens + batch_pending_prompt) and BEFORE
// process_batch_tokens dispatches it. Captures the tick-level batch
// composition under T4 chunked-prefill admission:
//   prefill_counts        — per-slot count of prefill tokens admitted
//                           this tick (slot_id -> count, count > 0)
//   decode_slots          — slot ids whose decode token is in this
//                           tick's batch
//   processing_set_at_start_of_tick — slots in PROCESSING state at
//                           tick start (before add_sampled_tokens)
//   loading_prompt_set_at_start_of_tick — slots in LOAD_PROMPT command
//                           at tick start (before batch_pending_prompt)
//   budget_k              — per-tick token budget K
//                           (params.prefill_chunk_budget, default
//                           n_ubatch)
// Schema and invariants in
// /home/llm/yarn-agentic/scripts/validate-batch-composition-trace.py
// (T4 form).
inline void emit_tick_dispatch(int tick,
                               const std::map<int, int>& prefill_counts,
                               const std::vector<int>& decode_slots,
                               const std::vector<int>& processing_set_at_start_of_tick,
                               const std::vector<int>& loading_prompt_set_at_start_of_tick,
                               int budget_k) {
    if (!enabled()) return;
    std::FILE* fp = trace_file();
    if (!fp) return;
    std::string line = "{\"action\": \"TickDispatch\", \"tick\": "
                     + std::to_string(tick)
                     + ", \"prefill_counts\": {";
    bool first = true;
    for (const auto & kv : prefill_counts) {
        if (!first) line += ", ";
        first = false;
        line += "\"" + std::to_string(kv.first) + "\": "
              + std::to_string(kv.second);
    }
    line += "}, \"decode_slots\": ";
    emit_set(line, decode_slots);
    line += ", \"processing_set_at_start_of_tick\": ";
    emit_set(line, processing_set_at_start_of_tick);
    line += ", \"loading_prompt_set_at_start_of_tick\": ";
    emit_set(line, loading_prompt_set_at_start_of_tick);
    line += ", \"budget_k\": " + std::to_string(budget_k);
    line += "}\n";
    std::lock_guard<std::mutex> lk(trace_mutex());
    std::fwrite(line.data(), 1, line.size(), fp);
    std::fflush(fp);
}

// P0.B.S5 extension — CUDA graph cache events. Companion to
// /home/llm/yarn-agentic/specs/graphs/cuda_graph_reuse.allium and
// /home/llm/yarn-agentic/specs/graphs/CUDAGraphReuse.tla.
//
// These emit functions are the API surface for a future ggml-cuda.cu
// instrumentation pass: at the cache-decision sites in ggml-cuda.cu
// (~4500-4830) call emit_graph_capture / emit_graph_exec_update /
// emit_graph_evict from the host-side dispatcher. The validator at
// scripts/validate-batch-composition-trace.py then checks the
// CUDAGraphReuse invariants (CacheBounded, DtypeStrictness,
// AddressToleranceScopedToViewCpy) against the live trace.
//
// Instrumentation wiring is deferred until Tier 2's cache-hit-rate
// measurement requires it (GP3.j in PHASE_NSTREAM_KV_PERF.md). The
// header API is landed now to keep the spec/code contract surface
// stable.

// CaptureGraph — emitted when a fresh topology miss causes a graph
// capture + instantiate. topology_hash is the integer hash key the
// cache uses; dtype is the per-node dst dtype (string label).
inline void emit_graph_capture(int tick,
                               uint64_t topology_hash,
                               const std::string& dtype,
                               int cache_size_post) {
    if (!enabled()) return;
    std::FILE* fp = trace_file();
    if (!fp) return;
    std::string line = "{\"action\": \"CaptureGraph\", \"tick\": "
                     + std::to_string(tick)
                     + ", \"topology_hash\": " + std::to_string(topology_hash)
                     + ", \"dtype\": \"" + dtype + "\""
                     + ", \"cache_size_post\": " + std::to_string(cache_size_post)
                     + "}\n";
    std::lock_guard<std::mutex> lk(trace_mutex());
    std::fwrite(line.data(), 1, line.size(), fp);
    std::fflush(fp);
}

// UpdateGraphExec — emitted when cudaGraphExecUpdate patches an
// existing entry. cache_index is the entry's position in the FIFO;
// dtype is the per-node dst dtype (must match the cached entry's
// dtype — DtypeStrictness).
inline void emit_graph_exec_update(int tick,
                                   uint64_t topology_hash,
                                   const std::string& dtype,
                                   int cache_index) {
    if (!enabled()) return;
    std::FILE* fp = trace_file();
    if (!fp) return;
    std::string line = "{\"action\": \"UpdateGraphExec\", \"tick\": "
                     + std::to_string(tick)
                     + ", \"topology_hash\": " + std::to_string(topology_hash)
                     + ", \"dtype\": \"" + dtype + "\""
                     + ", \"cache_index\": " + std::to_string(cache_index)
                     + "}\n";
    std::lock_guard<std::mutex> lk(trace_mutex());
    std::fwrite(line.data(), 1, line.size(), fp);
    std::fflush(fp);
}

// EvictGraphCacheEntry — emitted when FIFO eviction drops the
// oldest entry to make room for a fresh capture. evicted_topology
// is the topology hash of the dropped entry; cache_size_post is
// the size after the eviction (before the follow-up capture).
inline void emit_graph_evict(int tick,
                             uint64_t evicted_topology_hash,
                             int cache_size_post) {
    if (!enabled()) return;
    std::FILE* fp = trace_file();
    if (!fp) return;
    std::string line = "{\"action\": \"EvictGraphCacheEntry\", \"tick\": "
                     + std::to_string(tick)
                     + ", \"evicted_topology_hash\": "
                     + std::to_string(evicted_topology_hash)
                     + ", \"cache_size_post\": " + std::to_string(cache_size_post)
                     + "}\n";
    std::lock_guard<std::mutex> lk(trace_mutex());
    std::fwrite(line.data(), 1, line.size(), fp);
    std::fflush(fp);
}

// WarmUpRunIndex — partition the trace into discrete warm-up runs
// (R1, R2, R3, ...) so the validator can verify byte-identity of
// the dispatch sequence across runs. GP3.g in
// PHASE_NSTREAM_KV_PERF.md requires R1 ≡ R2 ≡ R3 byte-identical.
//
// Call once at the start of each run. Validator splits the trace
// at these markers and compares record-by-record across runs.
inline void emit_warmup_run_marker(int run_index) {
    if (!enabled()) return;
    std::FILE* fp = trace_file();
    if (!fp) return;
    std::string line = "{\"action\": \"WarmUpRunIndex\", \"run_index\": "
                     + std::to_string(run_index) + "}\n";
    std::lock_guard<std::mutex> lk(trace_mutex());
    std::fwrite(line.data(), 1, line.size(), fp);
    std::fflush(fp);
}

}  // namespace server_trace_ndjson
