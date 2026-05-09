#!/usr/bin/env python3
"""
PHASE46 B.6 — slot-permutation determinism harness.

Submits a fixed set of prompts to a running llama-server in three orderings
that exercise different slot-assignment paths through the parallel/ scheduler.
For each prompt, captures the generated token stream across all orderings and
asserts they are identical — i.e. the user-observable output for a given
prompt does not depend on which slot processed it.

Usage:
    python3 permute-slots.py --base-url http://127.0.0.1:8080 \\
        --prompts "Write a haiku" "Translate hello to French" "What is 2+2?" \\
        --n-predict 64 --runs 3

Exits 0 on PASS (all per-prompt streams match across orderings),
1 on FAIL.

Pre-req: a running llama-server with -np >= len(prompts) slots, started
with deterministic settings (seed, temp=0). Production target:
    LLAMA_TRACE_NDJSON=/tmp/perm.ndjson \\
    build/bin/llama-server -m <model> -np 3 --draft 3 -c 262144 \\
        -ngl 999 --port 8080 --log-disable
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request


def post_completion(base_url: str, prompt: str, n_predict: int, seed: int) -> dict:
    body = json.dumps(
        {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.0,
            "seed": seed,
            "cache_prompt": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/completion",
        data=body,
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read().decode("utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # Server may stream partial chunks; for non-streaming /completion
        # this is a real error.
        raise RuntimeError(f"non-JSON response: {raw[:200]}") from e


def output_text(resp: dict) -> str:
    # llama.cpp /completion returns "content" with the generated text.
    return resp.get("content") or resp.get("text") or ""


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def run_ordered(base_url: str, prompts: list[str], order: tuple[int, ...],
                n_predict: int, seed: int) -> dict[int, str]:
    """Submit prompts in `order` (indices into `prompts`) concurrently;
    return {original_prompt_index: output_text}."""
    submitted = [(idx, prompts[idx]) for idx in order]
    results: dict[int, str] = {}

    def go(item: tuple[int, str]) -> tuple[int, str]:
        idx, prompt = item
        resp = post_completion(base_url, prompt, n_predict, seed)
        return idx, output_text(resp)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(submitted)) as ex:
        # Submit one-at-a-time with tiny stagger so the server's slot
        # scheduler sees them in `order`.
        futures = []
        for item in submitted:
            futures.append(ex.submit(go, item))
            time.sleep(0.005)
        for f in concurrent.futures.as_completed(futures):
            idx, out = f.result()
            results[idx] = out
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--prompts", nargs="+", required=True,
                    help="prompts (one per slot); count must be <= server -np")
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", type=int, default=3,
                    help="repeats per ordering (smoke for transient noise)")
    args = ap.parse_args()

    n = len(args.prompts)
    if n < 2:
        print("need at least 2 prompts to permute", file=sys.stderr)
        return 2

    # Pick the first three permutations (or all if n < 3): identity,
    # rotate-left, rotate-right. For n=3 that gives [0,1,2], [1,2,0], [2,0,1].
    base = tuple(range(n))
    perms = [base]
    if n >= 2:
        perms.append(tuple(base[1:] + base[:1]))   # rotate left
    if n >= 3:
        perms.append(tuple(base[-1:] + base[:-1])) # rotate right

    print(f"==> permute-slots: n_prompts={n} runs={args.runs} perms={perms}")
    # per_prompt[idx] accumulates one output per (perm, run) sample.
    per_prompt: dict[int, list[str]] = {idx: [] for idx in range(n)}
    for perm in perms:
        for run in range(args.runs):
            results = run_ordered(args.base_url, args.prompts, perm,
                                  args.n_predict, args.seed)
            for idx in range(n):
                per_prompt[idx].append(results[idx])
            print(f"  perm={perm} run={run}: " +
                  ", ".join(f"p{i}={fingerprint(results[i])}" for i in range(n)))

    # For each prompt, all collected outputs must be identical.
    failures = 0
    for idx in range(n):
        outs = per_prompt[idx]
        if not outs:
            continue
        ref = outs[0]
        for j, o in enumerate(outs):
            if o != ref:
                failures += 1
                print(f"  FAIL prompt={idx} sample={j}: "
                      f"hash {fingerprint(o)} != reference {fingerprint(ref)}",
                      file=sys.stderr)
                # show first diff position for debugging
                for k in range(min(len(o), len(ref))):
                    if o[k] != ref[k]:
                        print(f"    first divergence at char {k}: "
                              f"{ref[k:k+40]!r} vs {o[k:k+40]!r}",
                              file=sys.stderr)
                        break

    print()
    if failures == 0:
        print(f"permute-slots: PASS — all {n} prompts produced identical "
              f"output across {len(perms)*args.runs} samples each")
        return 0
    print(f"permute-slots: FAIL — {failures} divergences detected")
    return 1


if __name__ == "__main__":
    sys.exit(main())
