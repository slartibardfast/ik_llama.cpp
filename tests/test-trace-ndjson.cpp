// Unit test for llama_trace NDJSON emission.
//
// Verifies in two forked children (so the singleton's cached env-check
// state never bleeds across modes):
//   1. with LLAMA_TRACE_NDJSON unset, llama_trace_enabled() is false and
//      emit() produces no file output;
//   2. with LLAMA_TRACE_NDJSON set, emit() writes one well-formed NDJSON
//      line per call, with the expected event-name + counters and
//      monotonic non-decreasing timestamps.

#include "llama-trace.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(cond) do {                                         \
    if (!(cond)) {                                               \
        std::fprintf(stderr, "Assertion failed at %s:%d: %s\n",  \
                __FILE__, __LINE__, #cond);                      \
        std::abort();                                            \
    }                                                            \
} while (0)

static std::vector<std::string> read_lines(const char * path) {
    std::vector<std::string> lines;
    FILE * fp = std::fopen(path, "rb");
    if (fp == nullptr) {
        return lines;
    }
    char buf[4096];
    while (std::fgets(buf, sizeof(buf), fp) != nullptr) {
        std::string s(buf);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
            s.pop_back();
        }
        if (!s.empty()) {
            lines.push_back(std::move(s));
        }
    }
    std::fclose(fp);
    return lines;
}

static double extract_number(const std::string & s, const char * key) {
    std::string needle = std::string("\"") + key + "\":";
    auto p = s.find(needle);
    if (p == std::string::npos) {
        return -1.0;
    }
    return std::strtod(s.c_str() + p + needle.size(), nullptr);
}

static std::string extract_string(const std::string & s, const char * key) {
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

static int wait_child(pid_t pid) {
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

int main() {
    char tmpl[] = "/tmp/llama-trace-test-XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        std::perror("mkstemp");
        return 1;
    }
    close(fd);
    std::remove(tmpl);  // recreate empty per child

    // --- Child A: env unset, emit must be silent ---
    {
        pid_t pid = fork();
        if (pid == 0) {
            unsetenv("LLAMA_TRACE_NDJSON");
            CHECK(llama_trace_enabled() == false);
            llama_trace_emit(LLAMA_TRACE_EV_FORK_DRAFT, 0, 0, 0, 0, 0);
            llama_trace_flush();
            std::_Exit(0);
        }
        CHECK(wait_child(pid) == 0);
        // tmpl must not exist: emit was silent
        FILE * f = std::fopen(tmpl, "rb");
        if (f != nullptr) {
            std::fclose(f);
            std::fprintf(stderr, "child A leaked output to %s\n", tmpl);
            std::abort();
        }
    }

    // --- Child B: env set, emit must write 6 well-formed lines ---
    {
        pid_t pid = fork();
        if (pid == 0) {
            setenv("LLAMA_TRACE_NDJSON", tmpl, 1);
            CHECK(llama_trace_enabled() == true);
            llama_trace_emit(LLAMA_TRACE_EV_FORK_DRAFT,  3, 1,  100, -1, -1);
            llama_trace_emit(LLAMA_TRACE_EV_JOIN_DRAFT,  3, 1,  100,  3, -1);
            llama_trace_emit(LLAMA_TRACE_EV_FORK_VERIFY, 3, 1,  100, -1, -1);
            llama_trace_emit(LLAMA_TRACE_EV_JOIN_VERIFY, 3, 1,  100, -1, -1);
            llama_trace_emit(LLAMA_TRACE_EV_ACCEPT,      3, 1,  100,  3,  2);
            llama_trace_emit(LLAMA_TRACE_EV_REJECT,      3, 2,  102,  3,  0);
            llama_trace_flush();
            std::_Exit(0);
        }
        CHECK(wait_child(pid) == 0);
    }

    auto lines = read_lines(tmpl);
    CHECK(lines.size() == 6);

    const char * expected_names[] = {
        "FORK_DRAFT", "JOIN_DRAFT",
        "FORK_VERIFY", "JOIN_VERIFY",
        "ACCEPT", "REJECT",
    };
    double prev_t = -1.0;
    for (size_t i = 0; i < lines.size(); ++i) {
        const auto & ln = lines[i];
        CHECK(ln.front() == '{');
        CHECK(ln.back()  == '}');
        std::string ev = extract_string(ln, "ev");
        if (ev != expected_names[i]) {
            std::fprintf(stderr, "line %zu: ev='%s' expected '%s'\n", i, ev.c_str(), expected_names[i]);
            std::abort();
        }
        double t = extract_number(ln, "t");
        CHECK(t >= prev_t);
        prev_t = t;
        CHECK(extract_number(ln, "slot") == 3.0);
    }

    CHECK(extract_number(lines[4], "n_drafted") == 3.0);
    CHECK(extract_number(lines[4], "n_accepted") == 2.0);
    CHECK(extract_number(lines[5], "n_accepted") == 0.0);

    std::remove(tmpl);
    std::printf("test-trace-ndjson: PASS\n");
    return 0;
}
