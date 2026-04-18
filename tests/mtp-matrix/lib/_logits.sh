#!/usr/bin/env bash
# Logit capture + diff helpers.
#
# The llama.cpp server's /completion endpoint doesn't expose raw logits.
# For logit-diff tests we use a lightweight approach: run inference,
# capture the embedding output (which the server DOES expose via
# /embedding when cparams.embeddings=true), and diff those.
#
# For tighter logit tests we rely on test-mtp-logits-ith.cpp (a dedicated
# test binary that reads raw MTP logits via llama_get_mtp_logits_ith).
#
# This lib provides:
#   logits_capture_to FILE  - after matrix_completion, dump top-10 tokens
#                              + logprobs to FILE as JSON
#   logits_diff A B [eps]    - compare two logit dumps, echo max_abs_diff,
#                              fail if > eps (default 5e-3)

logits_capture_to() {
    local out_file=$1
    # Use /completion with logprobs option
    local prompt=${2:-"The capital of France is"}
    local n_predict=${3:-1}

    curl -s -m 60 "http://127.0.0.1:$PORT/completion" \
        -d "{\"prompt\":\"$prompt\",\"n_predict\":$n_predict,\"temperature\":0,\"n_probs\":40}" \
        | python3 -c "import sys,json
d = json.load(sys.stdin)
probs = d.get('completion_probabilities', [])
out = []
for step in probs:
    out.append({
        'top': [(p.get('tok_str','?'), p.get('prob', 0.0)) for p in step.get('probs', [])],
    })
print(json.dumps(out, indent=2))" > "$out_file"
}

logits_diff() {
    local file_a=$1
    local file_b=$2
    local eps=${3:-5e-3}

    python3 - <<EOF
import json, sys, math
a = json.load(open("$file_a"))
b = json.load(open("$file_b"))
if len(a) != len(b):
    print(f"FAIL: step count differs: {len(a)} vs {len(b)}")
    sys.exit(1)
max_diff = 0.0
mismatched_tokens = 0
for i, (sa, sb) in enumerate(zip(a, b)):
    tops_a = {t: p for t, p in sa['top']}
    tops_b = {t: p for t, p in sb['top']}
    for t in set(tops_a) | set(tops_b):
        pa = tops_a.get(t, 0.0)
        pb = tops_b.get(t, 0.0)
        max_diff = max(max_diff, abs(pa - pb))
    top_a = sa['top'][0][0] if sa['top'] else None
    top_b = sb['top'][0][0] if sb['top'] else None
    if top_a != top_b:
        mismatched_tokens += 1
print(f"max_diff={max_diff:.6f} mismatched_top_tokens={mismatched_tokens}/{len(a)}")
if max_diff > $eps:
    sys.exit(1)
EOF
}
