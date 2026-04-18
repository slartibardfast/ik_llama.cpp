#!/usr/bin/env bash
# Coherence @ rollout=1: baseline path must produce non-collapsed output.
#
# Uses quality_check (not substring match) to catch model-collapse regressions
# that "contains paris" hid: e.g., " Paris.\nThe\nThe\nThe..." repetition
# produced a 61% acceptance artifact before the DeltaNet-cont revert.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
source "$SCRIPT_DIR/../lib/_quality.sh"
ROLLOUT=1
matrix_setup

if ! matrix_launch; then
    echo "CRASH: server did not start"
    tail -10 "$LOG" 2>/dev/null
    exit 3
fi

# 30 tokens is the minimum for entropy-based collapse detection to be useful.
matrix_completion "The capital of France is" 30 0

if [ -z "$OUT" ]; then matrix_fail "empty response body"; fi
if [ -z "$CONTENT" ]; then matrix_fail "no content in response: $(echo $OUT | head -c 200)"; fi

if ! quality_check "$CONTENT" >&2; then
    matrix_fail "output failed quality_check: $CONTENT"
fi

accept=$(matrix_parse_accept)
echo "content: $CONTENT"
echo "accept: ${accept:-n/a}"
matrix_pass "healthy output, accept=${accept:-?}"
