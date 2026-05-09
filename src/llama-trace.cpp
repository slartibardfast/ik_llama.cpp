// PHASE46 B.4: NDJSON trace emission implementation.

#include "llama-trace.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace {

static const char * EVENT_NAMES[] = {
    "FORK_DRAFT",
    "JOIN_DRAFT",
    "FORK_VERIFY",
    "JOIN_VERIFY",
    "ACCEPT",
    "REJECT",
};

struct trace_state {
    std::mutex mu;
    FILE *     fp      = nullptr;
    bool       checked = false;
    bool       enabled = false;
};

// Singleton — function-local static avoids static-init-order issues with
// other translation units that call llama_trace_emit during their init.
static trace_state & state() {
    static trace_state s;
    return s;
}

static void ensure_open(trace_state & s) {
    if (s.checked) {
        return;
    }
    const char * path = std::getenv("LLAMA_TRACE_NDJSON");
    if (path != nullptr && *path != '\0') {
        s.fp = std::fopen(path, "ab");
        if (s.fp != nullptr) {
            // Line buffered so partial traces are tail-able during long runs.
            std::setvbuf(s.fp, nullptr, _IOLBF, 0);
            s.enabled = true;
        } else {
            std::fprintf(stderr,
                "llama_trace: cannot open '%s' for append; tracing disabled\n",
                path);
        }
    }
    s.checked = true;
}

static double now_seconds() {
    using clock = std::chrono::steady_clock;
    static const auto epoch = clock::now();
    auto delta = clock::now() - epoch;
    return std::chrono::duration<double>(delta).count();
}

} // anonymous

extern "C" {

bool llama_trace_enabled(void) {
    auto & s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    ensure_open(s);
    return s.enabled;
}

void llama_trace_emit(int event_code,
                      int     slot_id,
                      int64_t step,
                      int64_t pos,
                      int     n_drafted,
                      int     n_accepted) {
    auto & s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    ensure_open(s);
    if (!s.enabled) {
        return;
    }
    const int ec = (event_code < 0 || event_code > LLAMA_TRACE_EV_REJECT)
                       ? -1
                       : event_code;
    const char * name = (ec >= 0) ? EVENT_NAMES[ec] : "UNKNOWN";
    std::fprintf(s.fp,
        "{\"t\":%.6f,\"ev\":\"%s\",\"slot\":%d,\"step\":%lld,\"pos\":%lld,\"n_drafted\":%d,\"n_accepted\":%d}\n",
        now_seconds(),
        name,
        slot_id,
        (long long) step,
        (long long) pos,
        n_drafted,
        n_accepted);
}

void llama_trace_flush(void) {
    auto & s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.fp != nullptr) {
        std::fflush(s.fp);
    }
}

} // extern "C"
