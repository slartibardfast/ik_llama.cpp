#!/usr/bin/env bash
# Coherence @ rollout=3: chained rollout must produce non-collapsed output.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
source "$SCRIPT_DIR/../lib/_quality.sh"
ROLLOUT=3
matrix_setup
if ! matrix_launch; then echo "CRASH: server did not start"; tail -10 "$LOG" 2>/dev/null; exit 3; fi
matrix_completion "The capital of France is" 30 0
if [ -z "$OUT" ] || [ -z "$CONTENT" ]; then matrix_fail "empty/no content"; fi
if ! quality_check "$CONTENT" >&2; then matrix_fail "output failed quality_check: $CONTENT"; fi
accept=$(matrix_parse_accept)
matrix_pass "healthy output, accept=${accept:-?}"
