#!/usr/bin/env python3
"""
PHASE46 B.5 — replay-trace.

Replays a recorded NDJSON trace by re-driving the server with the same
prompts/seeds and verifying that the *replay's* trace exhibits the same
structural shape: identical step count, identical FORK_DRAFT/JOIN_DRAFT
balance, identical per-slot ACCEPT/REJECT counts (within draft-rejection
noise), and accept rate within `--tolerance`.

If a reference output file is provided, the per-prompt completion text is
byte-equality-checked against the reference.

Usage:
    python3 replay-trace.py \\
        --trace /tmp/run0.ndjson \\
        --replay-trace /tmp/replay.ndjson \\
        --base-url http://127.0.0.1:8080 \\
        --prompts-file prompts.json \\
        [--reference-output /tmp/ref.json] [--tolerance 0.0]

prompts.json format: list of {"prompt": str, "seed": int, "n_predict": int}.

Exits 0 on PASS, 1 on FAIL. The structural check binds even without a
reference; the byte-equality check is opt-in.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request


def parse_trace(path: str) -> list[dict]:
    events: list[dict] = []
    with open(path, "rb") as fp:
        for ln_no, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{ln_no}: malformed JSON: {e}")
    return events


def trace_summary(events: list[dict]) -> dict:
    by_ev: collections.Counter = collections.Counter()
    by_slot_ev: dict[tuple[int, str], int] = collections.Counter()
    n_drafted = 0
    n_accepted = 0
    for e in events:
        ev = e.get("ev", "?")
        by_ev[ev] += 1
        slot = int(e.get("slot", -1))
        by_slot_ev[(slot, ev)] += 1
        if ev == "JOIN_DRAFT":
            d = int(e.get("n_drafted", 0))
            if d > 0:
                n_drafted += d
        elif ev == "ACCEPT":
            n_accepted += int(e.get("n_accepted", 0))
    accept_rate = (n_accepted / n_drafted) if n_drafted > 0 else 0.0
    return {
        "by_ev": dict(by_ev),
        "by_slot_ev": dict(by_slot_ev),
        "n_drafted": n_drafted,
        "n_accepted": n_accepted,
        "accept_rate": accept_rate,
    }


def post_completion(base_url: str, prompt: str, n_predict: int, seed: int) -> str:
    body = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "seed": seed,
        "cache_prompt": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/completion",
        data=body,
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        raw = r.read().decode("utf-8")
    try:
        resp = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"non-JSON response: {raw[:200]}") from e
    return resp.get("content") or resp.get("text") or ""


def drive(base_url: str, prompts: list[dict]) -> list[str]:
    out: list[str | None] = [None] * len(prompts)

    def go(i: int) -> tuple[int, str]:
        p = prompts[i]
        return i, post_completion(base_url, p["prompt"], int(p["n_predict"]), int(p["seed"]))

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        for f in concurrent.futures.as_completed([ex.submit(go, i) for i in range(len(prompts))]):
            i, text = f.result()
            out[i] = text
            time.sleep(0.005)
    return [s if s is not None else "" for s in out]


def diff_summaries(orig: dict, replay: dict, tolerance: float) -> list[str]:
    errs: list[str] = []
    # Event-class counts must match exactly.
    for ev in ("FORK_DRAFT", "JOIN_DRAFT", "FORK_VERIFY", "JOIN_VERIFY", "ACCEPT", "REJECT"):
        a = orig["by_ev"].get(ev, 0)
        b = replay["by_ev"].get(ev, 0)
        if a != b:
            errs.append(f"event {ev}: original={a} replay={b}")
    # Per-slot ACCEPT counts may legitimately differ if the underlying
    # decisions diverged; that's the failure we're hunting. Same with REJECT.
    seen = set(orig["by_slot_ev"]) | set(replay["by_slot_ev"])
    for key in sorted(seen):
        a = orig["by_slot_ev"].get(key, 0)
        b = replay["by_slot_ev"].get(key, 0)
        if a != b:
            errs.append(f"slot/event {key}: original={a} replay={b}")
    # Accept rate within tolerance
    da = abs(orig["accept_rate"] - replay["accept_rate"])
    if da > tolerance:
        errs.append(
            f"accept_rate: original={orig['accept_rate']:.4f} "
            f"replay={replay['accept_rate']:.4f} delta={da:.4f} > tol={tolerance}"
        )
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="original NDJSON trace from the recorded run")
    ap.add_argument("--replay-trace", required=True,
                    help="path the *replay* run will write its trace to "
                         "(set LLAMA_TRACE_NDJSON to this when starting the replay server)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--prompts-file", required=True,
                    help="JSON list of {prompt, seed, n_predict}")
    ap.add_argument("--reference-output", default=None,
                    help="optional JSON list of reference completion texts; "
                         "byte-equality check against replay output")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="accept-rate tolerance (default 0.0 = strict)")
    args = ap.parse_args()

    print(f"==> replay-trace")
    print(f"  original trace: {args.trace}")
    print(f"  replay trace:   {args.replay_trace}")
    print(f"  base URL:       {args.base_url}")

    orig_events = parse_trace(args.trace)
    if not orig_events:
        print("original trace has no events", file=sys.stderr)
        return 1
    orig = trace_summary(orig_events)
    print(f"  original: {orig['by_ev']}")
    print(f"  original n_drafted={orig['n_drafted']} n_accepted={orig['n_accepted']} "
          f"accept_rate={orig['accept_rate']:.4f}")

    with open(args.prompts_file, "rb") as fp:
        prompts = json.load(fp)
    if not isinstance(prompts, list) or not prompts:
        print(f"{args.prompts_file}: must be a non-empty JSON list", file=sys.stderr)
        return 1

    # Truncate the replay-trace path so we read fresh content.
    open(args.replay_trace, "wb").close()
    outputs = drive(args.base_url, prompts)

    # Wait briefly for trace flush, then re-read.
    time.sleep(0.5)
    replay_events = parse_trace(args.replay_trace)
    if not replay_events:
        print("replay trace has no events; was LLAMA_TRACE_NDJSON pointed "
              "at the right path on the replay server?", file=sys.stderr)
        return 1
    replay = trace_summary(replay_events)
    print(f"  replay:   {replay['by_ev']}")
    print(f"  replay   n_drafted={replay['n_drafted']} n_accepted={replay['n_accepted']} "
          f"accept_rate={replay['accept_rate']:.4f}")

    errs = diff_summaries(orig, replay, args.tolerance)

    if args.reference_output is not None:
        with open(args.reference_output, "rb") as fp:
            reference = json.load(fp)
        if not isinstance(reference, list):
            print(f"{args.reference_output}: must be a JSON list", file=sys.stderr)
            return 1
        if len(reference) != len(outputs):
            errs.append(f"reference length {len(reference)} != "
                        f"replay length {len(outputs)}")
        else:
            for i, (a, b) in enumerate(zip(reference, outputs)):
                if a != b:
                    ah = hashlib.sha256(a.encode("utf-8")).hexdigest()[:16]
                    bh = hashlib.sha256(b.encode("utf-8")).hexdigest()[:16]
                    errs.append(f"output[{i}]: reference={ah} replay={bh}")

    if not errs:
        print("\nreplay-trace: PASS")
        return 0
    print()
    for e in errs:
        print(f"  FAIL: {e}", file=sys.stderr)
    print(f"\nreplay-trace: FAIL ({len(errs)} divergences)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
