// test-runtime-hardening-mlockall.cpp
//
// Binding test for common_apply_runtime_hardening() — specifically the
// --mlockall path added 2026-05-27 as part of PHASE_NP8_FLAKE code-level
// RT mitigations (C-1).
//
// What this test asserts:
//   T1. Calling common_apply_runtime_hardening() with use_mlockall=false
//       is a no-op (no crash, no mlockall side effect).
//   T2. Calling with use_mlockall=true on Linux either succeeds (process
//       pages locked, VmLck > 0) or fails gracefully with a warning
//       (does not crash, does not abort).
//   T3. Calling repeatedly is safe (idempotent).
//
// What this test does NOT assert:
//   - That mlockall actually succeeds in CI environments. Locking memory
//     requires CAP_IPC_LOCK or unlimited RLIMIT_MEMLOCK; we'd need root or
//     systemd config to guarantee success. The test treats EPERM as an
//     acceptable outcome.
//
// Exit code:
//   0  — all PASS
//   1  — any FAIL
//   77 — skipped (non-Linux build with use_mlockall=true is a documented warn)

#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#if defined(__linux__)
#include <unistd.h>
#include <sys/mman.h>  // munlockall (used in cleanup)
#endif

static long read_vmlck_kb() {
#if defined(__linux__)
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmLck:", 0) == 0) {
            long kb = 0;
            // Format: "VmLck:\t  <num> kB"
            if (sscanf(line.c_str(), "VmLck: %ld kB", &kb) == 1) {
                return kb;
            }
        }
    }
    return -1; // not found
#else
    return -1;
#endif
}

int main() {
    int fail_count = 0;
    auto report = [&](const char * name, bool ok, const char * detail = "") {
        printf("  %s: %s%s%s\n", name, ok ? "PASS" : "FAIL",
               detail[0] ? " — " : "", detail);
        if (!ok) ++fail_count;
    };

    printf("test-runtime-hardening-mlockall:\n");

    // T1 — no-op when use_mlockall=false
    {
        gpt_params params;
        params.use_mlockall = false;
        const long before = read_vmlck_kb();
        common_apply_runtime_hardening(params);
        const long after = read_vmlck_kb();
        // Allow either both -1 (non-Linux) or both 0 (Linux, unlocked).
        const bool ok = (before == after);
        char detail[128];
        snprintf(detail, sizeof(detail), "VmLck before=%ld after=%ld", before, after);
        report("T1 use_mlockall=false is a no-op", ok, detail);
    }

    // T2 — use_mlockall=true: either succeeds (VmLck > 0) or fails
    //      gracefully. Both outcomes are acceptable — we only check no
    //      crash and that the function returns.
    bool t2_locked = false;
    {
        gpt_params params;
        params.use_mlockall = true;
        common_apply_runtime_hardening(params);
        const long after = read_vmlck_kb();
        char detail[128];
        if (after > 0) {
            snprintf(detail, sizeof(detail), "VmLck=%ld kB (mlockall succeeded)", after);
            t2_locked = true;
        } else if (after == 0) {
            snprintf(detail, sizeof(detail),
                "VmLck=0 — mlockall presumably failed EPERM (run with CAP_IPC_LOCK to bind)");
        } else {
            snprintf(detail, sizeof(detail), "VmLck unreadable (after=%ld)", after);
        }
        report("T2 use_mlockall=true does not crash", true, detail);
    }

    // T3 — idempotent: a second call must also not crash.
    {
        gpt_params params;
        params.use_mlockall = true;
        common_apply_runtime_hardening(params);
        common_apply_runtime_hardening(params);  // second call
        report("T3 second call is idempotent", true, "no crash");
    }

    printf("test-runtime-hardening-mlockall: %s (%d FAIL)\n",
           fail_count == 0 ? "PASS" : "FAIL", fail_count);

#if defined(__linux__)
    if (t2_locked) {
        // unlock so subsequent tests in the same process see a clean state
        munlockall();
    }
#else
    (void) t2_locked;
#endif

    return fail_count == 0 ? 0 : 1;
}
