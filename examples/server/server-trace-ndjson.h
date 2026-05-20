// server-trace-ndjson.h
//
// S5 — NDJSON trace emission for the Bug C closure scheduler invariants.
//
// Gated on LLAMA_TRACE_NDJSON_DIR. When unset, every call is a no-op
// branch in the hot path. When set, one NDJSON line is appended per
// emit call to <dir>/server-trace.ndjson.
//
// Schema is defined alongside the validator at
// /home/llm/yarn-agentic/scripts/validate-batch-composition-trace.py.
//
// Per feedback_bake_measurement_env_gates: this is a diagnostic that
// should be deleted (header + emit calls + env var) once the spec/code
// agreement is established and the n_stream 4D port (N3) removes the
// load-bearing decode-side gate. Tracked in
// /home/llm/yarn-agentic/PHASE_NSTREAM_KV_4D.md S5 closure.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
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

// Emit a TickDispatch record. Call exactly once per llama_decode
// dispatched by the server tick — right before the decode runs, so the
// recorded sets reflect what's about to be processed.
inline void emit_tick_dispatch(int tick,
                               const std::vector<int>& prefill_slots,
                               const std::vector<int>& decode_slots,
                               const std::vector<int>& loading_prompt_set) {
    if (!enabled()) return;
    std::FILE* fp = trace_file();
    if (!fp) return;
    std::string line = "{\"action\": \"TickDispatch\", \"tick\": "
                     + std::to_string(tick)
                     + ", \"prefill_slots\": ";
    emit_set(line, prefill_slots);
    line += ", \"decode_slots\": ";
    emit_set(line, decode_slots);
    line += ", \"loading_prompt_set_at_start_of_tick\": ";
    emit_set(line, loading_prompt_set);
    line += "}\n";
    std::lock_guard<std::mutex> lk(trace_mutex());
    std::fwrite(line.data(), 1, line.size(), fp);
    std::fflush(fp);
}

}  // namespace server_trace_ndjson
