#!/usr/bin/env python3
"""
PHASE46 B.5 — offline unit test for replay-trace's parse + summary + diff
logic. No server required.

Constructs synthetic NDJSON traces that exercise:
  * identical traces → diff returns empty (PASS)
  * mismatched FORK_DRAFT counts → diff names the event
  * accept-rate divergence > tolerance → diff names accept_rate
  * per-(slot, ev) count mismatch → diff names the slot/event pair
"""

from __future__ import annotations

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "tools"))

# Import the helpers from replay-trace.py without running its CLI.
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "replay_trace",
    os.path.join(ROOT, "..", "tools", "replay-trace.py"),
)
replay_trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay_trace)


def write_trace(lines: list[str]) -> str:
    fd, path = tempfile.mkstemp(suffix=".ndjson")
    with os.fdopen(fd, "w") as fp:
        for ln in lines:
            fp.write(ln)
            fp.write("\n")
    return path


# Trace A — single slot, three steps, 75% accept
A = [
    '{"t":0.001,"ev":"FORK_DRAFT","slot":0,"step":0,"pos":10,"n_drafted":-1,"n_accepted":-1}',
    '{"t":0.002,"ev":"JOIN_DRAFT","slot":0,"step":0,"pos":10,"n_drafted":4,"n_accepted":-1}',
    '{"t":0.003,"ev":"ACCEPT","slot":0,"step":0,"pos":10,"n_drafted":4,"n_accepted":3}',
    '{"t":0.004,"ev":"FORK_DRAFT","slot":0,"step":1,"pos":13,"n_drafted":-1,"n_accepted":-1}',
    '{"t":0.005,"ev":"JOIN_DRAFT","slot":0,"step":1,"pos":13,"n_drafted":4,"n_accepted":-1}',
    '{"t":0.006,"ev":"ACCEPT","slot":0,"step":1,"pos":13,"n_drafted":4,"n_accepted":3}',
]

# Trace B — same as A
B = list(A)

# Trace C — same shape but ACCEPT counts diverge (60% accept instead of 75%)
C = [
    '{"t":0.001,"ev":"FORK_DRAFT","slot":0,"step":0,"pos":10,"n_drafted":-1,"n_accepted":-1}',
    '{"t":0.002,"ev":"JOIN_DRAFT","slot":0,"step":0,"pos":10,"n_drafted":5,"n_accepted":-1}',
    '{"t":0.003,"ev":"ACCEPT","slot":0,"step":0,"pos":10,"n_drafted":5,"n_accepted":3}',
    '{"t":0.004,"ev":"FORK_DRAFT","slot":0,"step":1,"pos":13,"n_drafted":-1,"n_accepted":-1}',
    '{"t":0.005,"ev":"JOIN_DRAFT","slot":0,"step":1,"pos":13,"n_drafted":5,"n_accepted":-1}',
    '{"t":0.006,"ev":"ACCEPT","slot":0,"step":1,"pos":13,"n_drafted":5,"n_accepted":3}',
]

# Trace D — A but with one extra FORK_DRAFT
D = list(A) + [
    '{"t":0.007,"ev":"FORK_DRAFT","slot":0,"step":2,"pos":16,"n_drafted":-1,"n_accepted":-1}',
    '{"t":0.008,"ev":"JOIN_DRAFT","slot":0,"step":2,"pos":16,"n_drafted":4,"n_accepted":-1}',
]

# Trace E — A but extra slot
E = list(A) + [
    '{"t":0.007,"ev":"FORK_DRAFT","slot":1,"step":0,"pos":20,"n_drafted":-1,"n_accepted":-1}',
    '{"t":0.008,"ev":"JOIN_DRAFT","slot":1,"step":0,"pos":20,"n_drafted":4,"n_accepted":-1}',
    '{"t":0.009,"ev":"REJECT","slot":1,"step":0,"pos":20,"n_drafted":4,"n_accepted":0}',
]


def assert_eq(label: str, got, want):
    if got != want:
        raise SystemExit(f"FAIL {label}: got={got} want={want}")


def main() -> int:
    pa, pb, pc, pd, pe = (write_trace(t) for t in (A, B, C, D, E))
    try:
        sa = replay_trace.trace_summary(replay_trace.parse_trace(pa))
        sb = replay_trace.trace_summary(replay_trace.parse_trace(pb))
        sc = replay_trace.trace_summary(replay_trace.parse_trace(pc))
        sd = replay_trace.trace_summary(replay_trace.parse_trace(pd))
        se = replay_trace.trace_summary(replay_trace.parse_trace(pe))

        # 1. A vs B identical
        errs = replay_trace.diff_summaries(sa, sb, tolerance=0.0)
        assert_eq("A vs B (identical)", errs, [])

        # 2. A vs C accept-rate divergence (0.75 vs 0.60), tolerance=0
        errs = replay_trace.diff_summaries(sa, sc, tolerance=0.0)
        if not any("accept_rate" in e for e in errs):
            raise SystemExit(f"FAIL A vs C: expected accept_rate err, got {errs}")

        # 3. Same pair with tolerance=0.2 → tolerated
        errs = replay_trace.diff_summaries(sa, sc, tolerance=0.2)
        if any("accept_rate" in e for e in errs):
            raise SystemExit(f"FAIL A vs C (tol=0.2): expected no accept_rate err, got {errs}")

        # 4. A vs D extra FORK_DRAFT
        errs = replay_trace.diff_summaries(sa, sd, tolerance=0.0)
        if not any("FORK_DRAFT" in e for e in errs):
            raise SystemExit(f"FAIL A vs D: expected FORK_DRAFT err, got {errs}")

        # 5. A vs E extra slot 1 events
        errs = replay_trace.diff_summaries(sa, se, tolerance=0.0)
        if not any("slot/event (1," in e for e in errs):
            raise SystemExit(f"FAIL A vs E: expected slot 1 err, got {errs}")

        print("test-replay-trace-offline: PASS")
        return 0
    finally:
        for p in (pa, pb, pc, pd, pe):
            try:
                os.unlink(p)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
