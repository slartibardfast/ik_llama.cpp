// PHASE46 — trace-validate
//
// Reads an NDJSON trace produced by llama_trace (LLAMA_TRACE_NDJSON=<path>)
// and verifies structural invariants needed by replay / permutation tools:
//
//   * each line parses as a JSON object with the expected keys
//     (t, ev, slot, step, pos, n_drafted, n_accepted)
//   * timestamps are monotonic non-decreasing
//   * for every (slot, step) where FORK_DRAFT fires there is a matching
//     JOIN_DRAFT later in the trace; FORK_VERIFY/JOIN_VERIFY likewise
//   * ACCEPT events have n_accepted >= 1; REJECT events have n_accepted == 0
//
// Exits 0 on PASS, 1 on FAIL. The full replay (re-decode via llama API
// and diff token IDs) is a future tool — this is the static-verification
// half of B.5 and is independently useful when triaging traces from a
// live server.

// Wire-format event names produced by src/llama-trace.cpp. Kept inline here
// rather than #including llama-trace.h because this tool only consumes the
// serialized form (NDJSON) — it does not link the trace module.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

struct event_rec {
    double      t        = 0.0;
    std::string ev;
    int         slot     = 0;
    int64_t     step     = 0;
    int64_t     pos      = 0;
    int         n_drafted   = -1;
    int         n_accepted  = -1;
    size_t      line_no  = 0;
};

static double get_number(const std::string & s, const char * key) {
    std::string needle = std::string("\"") + key + "\":";
    auto p = s.find(needle);
    if (p == std::string::npos) {
        return -1.0;
    }
    return std::strtod(s.c_str() + p + needle.size(), nullptr);
}

static std::string get_string(const std::string & s, const char * key) {
    std::string needle = std::string("\"") + key + "\":\"";
    auto p = s.find(needle);
    if (p == std::string::npos) {
        return {};
    }
    p += needle.size();
    auto q = s.find('"', p);
    if (q == std::string::npos) {
        return {};
    }
    return s.substr(p, q - p);
}

static bool parse_line(const std::string & line, size_t line_no, event_rec & out) {
    if (line.empty() || line.front() != '{' || line.back() != '}') {
        std::fprintf(stderr, "line %zu: not a JSON object\n", line_no);
        return false;
    }
    out.t   = get_number(line, "t");
    out.ev  = get_string(line, "ev");
    out.slot = (int)    get_number(line, "slot");
    out.step = (int64_t)get_number(line, "step");
    out.pos  = (int64_t)get_number(line, "pos");
    out.n_drafted   = (int)get_number(line, "n_drafted");
    out.n_accepted  = (int)get_number(line, "n_accepted");
    out.line_no = line_no;
    if (out.ev.empty()) {
        std::fprintf(stderr, "line %zu: missing ev field\n", line_no);
        return false;
    }
    return true;
}

} // anonymous

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <trace.ndjson>\n", argv[0]);
        return 2;
    }

    FILE * fp = std::fopen(argv[1], "rb");
    if (fp == nullptr) {
        std::perror(argv[1]);
        return 2;
    }

    std::vector<event_rec> events;
    char buf[8192];
    size_t line_no = 0;
    int parse_errors = 0;
    while (std::fgets(buf, sizeof(buf), fp) != nullptr) {
        ++line_no;
        std::string s(buf);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
            s.pop_back();
        }
        if (s.empty()) {
            continue;
        }
        event_rec e;
        if (!parse_line(s, line_no, e)) {
            ++parse_errors;
            continue;
        }
        events.push_back(std::move(e));
    }
    std::fclose(fp);

    if (events.empty()) {
        std::fprintf(stderr, "no events parsed\n");
        return 1;
    }

    int errors = parse_errors;

    // Monotonic timestamps
    for (size_t i = 1; i < events.size(); ++i) {
        if (events[i].t < events[i-1].t) {
            std::fprintf(stderr,
                "line %zu: timestamp regressed (%.6f < %.6f)\n",
                events[i].line_no, events[i].t, events[i-1].t);
            ++errors;
        }
    }

    // FORK/JOIN balance per (slot, step). Track open forks and require a
    // matching join later. Excess joins or unmatched forks are errors.
    using key_t = std::pair<int, int64_t>;
    std::map<key_t, int> open_draft;   // (slot, step) → open count
    std::map<key_t, int> open_verify;
    int accept_total  = 0;
    int reject_total  = 0;

    for (const auto & e : events) {
        if (e.ev == "FORK_DRAFT") {
            open_draft[{e.slot, e.step}]++;
        } else if (e.ev == "JOIN_DRAFT") {
            auto it = open_draft.find({e.slot, e.step});
            if (it == open_draft.end() || it->second == 0) {
                std::fprintf(stderr,
                    "line %zu: JOIN_DRAFT without matching FORK (slot=%d step=%lld)\n",
                    e.line_no, e.slot, (long long) e.step);
                ++errors;
            } else {
                if (--it->second == 0) {
                    open_draft.erase(it);
                }
            }
        } else if (e.ev == "FORK_VERIFY") {
            open_verify[{e.slot, e.step}]++;
        } else if (e.ev == "JOIN_VERIFY") {
            auto it = open_verify.find({e.slot, e.step});
            if (it == open_verify.end() || it->second == 0) {
                std::fprintf(stderr,
                    "line %zu: JOIN_VERIFY without matching FORK (slot=%d step=%lld)\n",
                    e.line_no, e.slot, (long long) e.step);
                ++errors;
            } else {
                if (--it->second == 0) {
                    open_verify.erase(it);
                }
            }
        } else if (e.ev == "ACCEPT") {
            ++accept_total;
            if (e.n_accepted < 1) {
                std::fprintf(stderr,
                    "line %zu: ACCEPT with n_accepted=%d (expected >=1)\n",
                    e.line_no, e.n_accepted);
                ++errors;
            }
        } else if (e.ev == "REJECT") {
            ++reject_total;
            if (e.n_accepted != 0) {
                std::fprintf(stderr,
                    "line %zu: REJECT with n_accepted=%d (expected 0)\n",
                    e.line_no, e.n_accepted);
                ++errors;
            }
        } else {
            std::fprintf(stderr,
                "line %zu: unknown event '%s'\n",
                e.line_no, e.ev.c_str());
            ++errors;
        }
    }

    if (!open_draft.empty()) {
        for (const auto & kv : open_draft) {
            std::fprintf(stderr,
                "unmatched FORK_DRAFT remains open: slot=%d step=%lld count=%d\n",
                kv.first.first, (long long) kv.first.second, kv.second);
            ++errors;
        }
    }
    if (!open_verify.empty()) {
        for (const auto & kv : open_verify) {
            std::fprintf(stderr,
                "unmatched FORK_VERIFY remains open: slot=%d step=%lld count=%d\n",
                kv.first.first, (long long) kv.first.second, kv.second);
            ++errors;
        }
    }

    std::printf("trace-validate: %zu events, %d ACCEPT, %d REJECT, %d errors\n",
                events.size(), accept_total, reject_total, errors);
    return errors == 0 ? 0 : 1;
}
