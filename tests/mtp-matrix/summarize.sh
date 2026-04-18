#!/usr/bin/env bash
# Summarize a completed run-all.sh output log into categorized failure/pass
# groupings. Useful for quickly triaging a ~500-row matrix.
set -u
LOG=${1:-/tmp/mtp-matrix-full.log}
if [ ! -f "$LOG" ]; then
    echo "usage: $0 <path to log>"
    exit 1
fi

echo "=== Summary of $LOG ==="
echo

# Per-category pass/fail count.
echo "By category:"
awk '
    / test-coherence.* PASS / { p["coherence"]++ }
    / test-coherence.* FAIL / { f["coherence"]++ }
    / test-coherence.* CRASH / { c["coherence"]++ }
    / test-shape-.* PASS / { p["shape"]++ }
    / test-shape-.* FAIL / { f["shape"]++ }
    / test-shape-.* CRASH / { c["shape"]++ }
    / test-(iter0|backend|per-iteration).* PASS / { p["semantics"]++ }
    / test-(iter0|backend|per-iteration).* FAIL / { f["semantics"]++ }
    / test-(iter0|backend|per-iteration).* CRASH / { c["semantics"]++ }
    / test-(reserve|same-shape|shape-flip|growing-shape).* PASS / { p["scheduler"]++ }
    / test-(reserve|same-shape|shape-flip|growing-shape).* FAIL / { f["scheduler"]++ }
    / test-(reserve|same-shape|shape-flip|growing-shape).* CRASH / { c["scheduler"]++ }
    / test-(startup|no-mtp).* PASS / { p["server"]++ }
    / test-(startup|no-mtp).* FAIL / { f["server"]++ }
    / test-(startup|no-mtp).* CRASH / { c["server"]++ }
    / test-(exact-paris|accept-floor|tps-floor|double-request).* PASS / { p["regressions"]++ }
    / test-(exact-paris|accept-floor|tps-floor|double-request).* FAIL / { f["regressions"]++ }
    / test-(exact-paris|accept-floor|tps-floor|double-request).* CRASH / { c["regressions"]++ }
    END {
        split("coherence semantics shape scheduler server regressions", cats, " ")
        for (i=1; i<=6; i++) {
            cat = cats[i]
            printf "  %-12s PASS=%-4d FAIL=%-4d CRASH=%-4d\n", cat, p[cat]+0, f[cat]+0, c[cat]+0
        }
    }
' "$LOG"
echo

# Per-build totals.
echo "By build (##header##):"
awk '
    /^### / { build = $0; sub(/^### /, "", build); sub(/ \(.*$/, "", build); next }
    / PASS / { pass[build]++ }
    / FAIL / { fail[build]++ }
    / CRASH / { crash[build]++ }
    / SKIP / { skip[build]++ }
    END {
        for (b in pass) printf "  %-35s PASS=%-4d FAIL=%-4d CRASH=%-4d SKIP=%-4d\n", b, pass[b]+0, fail[b]+0, crash[b]+0, skip[b]+0
    }
' "$LOG" | sort
echo

# Unique failures
echo "Unique failing tests (across all builds):"
grep -E "  \[.*\] test-.* FAIL" "$LOG" | awk '{print $3}' | sort -u
echo
echo "Unique crashing tests:"
grep -E "  \[.*\] test-.* CRASH" "$LOG" | awk '{print $3}' | sort -u
