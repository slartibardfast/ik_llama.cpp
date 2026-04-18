#!/usr/bin/env bash
# Semantic check: in chained rollout, iter=0's logits at the last position
# should byte-identical (or tiny-epsilon) match the single-pass rollout=1
# logits at the same position. Proves Fix A (polaris port) preserves iter=0
# semantics.
#
# Uses completion_probabilities (top-40 tokens) as proxy since the server
# doesn't expose raw logits via HTTP. Top-40 token set and their probs should
# match for correct iter=0.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
source "$SCRIPT_DIR/../lib/_logits.sh"

PROMPT="The capital of France is"

# Run 1: rollout=1
ROLLOUT=1
matrix_setup
if ! matrix_launch; then echo "CRASH: r1 server"; exit 3; fi
FILE_R1=/tmp/logits-r1-$$.json
logits_capture_to "$FILE_R1" "$PROMPT" 1
matrix_teardown

# Run 2: rollout=3
ROLLOUT=3
PORT=$((PORT + 1))
matrix_setup
if ! matrix_launch; then echo "CRASH: r3 server"; rm -f "$FILE_R1"; exit 3; fi
FILE_R3=/tmp/logits-r3-$$.json
logits_capture_to "$FILE_R3" "$PROMPT" 1
matrix_teardown

# Diff. Top-token must match (deterministic greedy). Probs should be close.
if ! logits_diff "$FILE_R1" "$FILE_R3" 5e-3; then
    echo "FAIL: iter0 probs diverge from single-pass"
    rm -f "$FILE_R1" "$FILE_R3"
    exit 1
fi
rm -f "$FILE_R1" "$FILE_R3"
echo "PASS: iter0 matches single-pass"
exit 0
