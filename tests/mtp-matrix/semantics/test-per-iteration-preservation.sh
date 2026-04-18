#!/usr/bin/env bash
# Semantic check: running rollout=N and rollout=N+1 for the same prompt,
# iter=k's logits (k < N) should match between the two. Proves that adding
# more rollout iterations doesn't change earlier iterations.
#
# This test captures iter=0 top-tokens at two different rollout counts and
# asserts they match. If they diverge, the extra iterations are contaminating
# earlier ones (scheduler buffer aliasing, KV cache writes in wrong order, etc).
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
source "$SCRIPT_DIR/../lib/_logits.sh"

PROMPT="The capital of France is"

# Run at rollout=2
ROLLOUT=2
matrix_setup
if ! matrix_launch; then echo "CRASH: r2"; exit 3; fi
FILE_2=/tmp/pres-r2-$$.json
logits_capture_to "$FILE_2" "$PROMPT" 1
matrix_teardown

# Run at rollout=3
ROLLOUT=3
PORT=$((PORT + 1))
matrix_setup
if ! matrix_launch; then echo "CRASH: r3"; rm -f "$FILE_2"; exit 3; fi
FILE_3=/tmp/pres-r3-$$.json
logits_capture_to "$FILE_3" "$PROMPT" 1
matrix_teardown

# iter=0 (the first draft row) should be identical-ish between r=2 and r=3.
if ! logits_diff "$FILE_2" "$FILE_3" 5e-3; then
    echo "FAIL: iter=0 changes between rollout=2 and rollout=3"
    rm -f "$FILE_2" "$FILE_3"
    exit 1
fi
rm -f "$FILE_2" "$FILE_3"
echo "PASS: iter=0 preserved across rollout count"
exit 0
