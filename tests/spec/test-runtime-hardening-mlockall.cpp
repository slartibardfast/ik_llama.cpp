// test-runtime-hardening-mlockall.cpp
//
// Binding test for common_apply_runtime_hardening() — covers the
// --mlockall (C-1), --rt-prio (C-2), and --cpu-mask (C-3) paths added
// 2026-05-27 as part of PHASE_NP8_FLAKE code-level RT mitigations.
//
// What this test asserts:
//   T1. Calling common_apply_runtime_hardening() with all knobs at
//       defaults is a no-op (no crash, no side effect).
//   T2. Calling with use_mlockall=true on Linux either succeeds (process
//       pages locked, VmLck > 0) or fails gracefully with a warning
//       (does not crash, does not abort).
//   T3. Calling repeatedly is safe (idempotent).
//   T4. Calling with rt_priority in [1, sched_get_priority_max(SCHED_FIFO)]
//       either succeeds (sched policy becomes SCHED_FIFO) or fails
//       gracefully with EPERM; either way, no crash.
//   T5. Out-of-range rt_priority is rejected with a warning (no crash,
//       no policy change).
//   T6. cpu_mask "0xF0" parses to {4,5,6,7}; applied as affinity if
//       the test runs on a host with ≥8 logical CPUs (affinity
//       set should succeed unconditionally — unlike mlockall/rt-prio,
//       setaffinity does NOT require privileges).
//   T7. cpu_mask "4-7" range syntax parses identically to "0xF0".
//   T8. Malformed cpu_mask is rejected with a warning (no crash, no
//       affinity change).
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
#include <pthread.h>
#include <sched.h>
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

    // T4 — rt_priority in valid range. Either succeeds (we now have
    // SCHED_FIFO) or fails EPERM (test runs unprivileged). Both PASS.
    {
        gpt_params params;
#if defined(__linux__)
        const int target_prio = sched_get_priority_min(SCHED_FIFO);
        params.rt_priority = target_prio > 0 ? target_prio : 1;
        common_apply_runtime_hardening(params);
        int policy = 0;
        struct sched_param sp = {};
        pthread_getschedparam(pthread_self(), &policy, &sp);
        char detail[128];
        if (policy == SCHED_FIFO) {
            snprintf(detail, sizeof(detail),
                "policy=SCHED_FIFO prio=%d (priority elevation succeeded)",
                sp.sched_priority);
            // Restore SCHED_OTHER so subsequent tests see default scheduling.
            struct sched_param zp = {};
            pthread_setschedparam(pthread_self(), SCHED_OTHER, &zp);
        } else {
            snprintf(detail, sizeof(detail),
                "policy=%d (unchanged; EPERM expected unprivileged)", policy);
        }
        report("T4 rt_priority in-range does not crash", true, detail);
#else
        params.rt_priority = 1;
        common_apply_runtime_hardening(params);
        report("T4 rt_priority on non-Linux is documented warn", true, "n/a");
#endif
    }

    // T5 — out-of-range rt_priority logs warning and leaves policy alone.
    {
        gpt_params params;
        params.rt_priority = 9999;  // clearly out of SCHED_FIFO range
        common_apply_runtime_hardening(params);
#if defined(__linux__)
        int policy = 0;
        struct sched_param sp = {};
        pthread_getschedparam(pthread_self(), &policy, &sp);
        const bool ok = (policy != SCHED_FIFO);
        char detail[128];
        snprintf(detail, sizeof(detail),
            "policy=%d (must NOT be SCHED_FIFO=%d after out-of-range request)",
            policy, SCHED_FIFO);
        report("T5 out-of-range rt_priority is rejected", ok, detail);
#else
        report("T5 out-of-range rt_priority on non-Linux", true, "n/a");
#endif
    }

#if defined(__linux__)
    // Save the test process's original affinity so we can restore between
    // T6/T7. setaffinity is unprivileged — no EPERM path to handle.
    cpu_set_t orig_mask;
    CPU_ZERO(&orig_mask);
    pthread_getaffinity_np(pthread_self(), sizeof(orig_mask), &orig_mask);
    const int n_online = sysconf(_SC_NPROCESSORS_ONLN);

    auto cpus_in_mask_as_string = [](const cpu_set_t & m) {
        std::string s;
        for (int i = 0; i < 32; ++i) {
            if (CPU_ISSET(i, &m)) {
                if (!s.empty()) s += ",";
                s += std::to_string(i);
            }
        }
        return s.empty() ? std::string("<empty>") : s;
    };

    // T6 — hex mask "0xF0" → CPUs {4,5,6,7}. Requires ≥8 logical CPUs
    // for the affinity call to succeed (otherwise the kernel returns
    // EINVAL because the mask contains no online CPU).
    if (n_online >= 8) {
        gpt_params params;
        params.cpu_mask = "0xF0";
        common_apply_runtime_hardening(params);
        cpu_set_t got;
        CPU_ZERO(&got);
        pthread_getaffinity_np(pthread_self(), sizeof(got), &got);
        const bool ok = CPU_ISSET(4, &got) && CPU_ISSET(5, &got) &&
                        CPU_ISSET(6, &got) && CPU_ISSET(7, &got) &&
                        !CPU_ISSET(0, &got) && !CPU_ISSET(3, &got);
        char detail[160];
        snprintf(detail, sizeof(detail),
            "actual affinity after '0xF0' = {%s}",
            cpus_in_mask_as_string(got).c_str());
        report("T6 cpu_mask='0xF0' pins to {4,5,6,7}", ok, detail);

        // restore for T7
        pthread_setaffinity_np(pthread_self(), sizeof(orig_mask), &orig_mask);
    } else {
        report("T6 cpu_mask='0xF0'", true, "skipped (<8 online CPUs)");
    }

    // T7 — range mask "4-7" → CPUs {4,5,6,7}. Must produce identical
    // affinity to T6.
    if (n_online >= 8) {
        gpt_params params;
        params.cpu_mask = "4-7";
        common_apply_runtime_hardening(params);
        cpu_set_t got;
        CPU_ZERO(&got);
        pthread_getaffinity_np(pthread_self(), sizeof(got), &got);
        const bool ok = CPU_ISSET(4, &got) && CPU_ISSET(5, &got) &&
                        CPU_ISSET(6, &got) && CPU_ISSET(7, &got) &&
                        !CPU_ISSET(0, &got) && !CPU_ISSET(3, &got);
        char detail[160];
        snprintf(detail, sizeof(detail),
            "actual affinity after '4-7' = {%s}",
            cpus_in_mask_as_string(got).c_str());
        report("T7 cpu_mask='4-7' matches '0xF0'", ok, detail);

        // restore
        pthread_setaffinity_np(pthread_self(), sizeof(orig_mask), &orig_mask);
    } else {
        report("T7 cpu_mask='4-7'", true, "skipped (<8 online CPUs)");
    }

    // T8 — malformed mask is rejected.
    {
        gpt_params params;
        params.cpu_mask = "garbage!!";
        cpu_set_t before;
        CPU_ZERO(&before);
        pthread_getaffinity_np(pthread_self(), sizeof(before), &before);
        common_apply_runtime_hardening(params);
        cpu_set_t after;
        CPU_ZERO(&after);
        pthread_getaffinity_np(pthread_self(), sizeof(after), &after);
        // CPU_EQUAL: 1 if equal, 0 otherwise
        const bool unchanged = CPU_EQUAL(&before, &after);
        report("T8 malformed cpu_mask is rejected", unchanged,
               "affinity unchanged");
    }

    // T9 — empirical: pthread_create(attr=NULL) inherits the creator's
    // CPU affinity mask. Critical for ggml workers (line 26910 of ggml.c
    // passes NULL attr) — proves they pick up whatever --cpu-mask sets
    // on the main thread BEFORE graph compute fires.
    if (n_online >= 8) {
        gpt_params params;
        params.cpu_mask = "0xF0";
        common_apply_runtime_hardening(params);

        // Spawn a child thread mimicking ggml's exact call pattern.
        struct child_ctx { cpu_set_t mask; };
        child_ctx ctx{};
        CPU_ZERO(&ctx.mask);
        auto child_fn = [](void * arg) -> void * {
            auto * c = static_cast<child_ctx *>(arg);
            pthread_getaffinity_np(pthread_self(), sizeof(c->mask), &c->mask);
            return nullptr;
        };
        pthread_t child = 0;
        const int rc = pthread_create(&child, /*attr=*/nullptr,
                                      child_fn, &ctx);
        if (rc == 0) {
            pthread_join(child, nullptr);
            const bool inherits = CPU_ISSET(4, &ctx.mask) &&
                                  CPU_ISSET(5, &ctx.mask) &&
                                  CPU_ISSET(6, &ctx.mask) &&
                                  CPU_ISSET(7, &ctx.mask) &&
                                  !CPU_ISSET(0, &ctx.mask) &&
                                  !CPU_ISSET(3, &ctx.mask);
            char detail[160];
            snprintf(detail, sizeof(detail),
                "child thread affinity = {%s} (parent set to {4,5,6,7})",
                cpus_in_mask_as_string(ctx.mask).c_str());
            report("T9 pthread_create(NULL) inherits cpu_mask", inherits, detail);
        } else {
            report("T9 pthread_create(NULL) inherits cpu_mask", false,
                   "could not create child thread");
        }

        // restore
        pthread_setaffinity_np(pthread_self(), sizeof(orig_mask), &orig_mask);
    } else {
        report("T9 pthread_create inheritance", true, "skipped (<8 online CPUs)");
    }
#else
    report("T6-T8 cpu_mask tests on non-Linux", true, "n/a");
#endif

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
