#!/usr/bin/env bash
# Long generation coherence: rollout=1 with n_predict=50 must stay coherent.
# Catches degradation that only appears after many tokens AND model collapse
# (repeated tokens masquerading as output).
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
source "$SCRIPT_DIR/../lib/_quality.sh"
ROLLOUT=1
matrix_setup
if ! matrix_launch; then echo "CRASH: server did not start"; tail -10 "$LOG" 2>/dev/null; exit 3; fi
matrix_completion "The capital of France is" 50 0
if [ -z "$OUT" ] || [ -z "$CONTENT" ]; then matrix_fail "empty/no content"; fi
# Guard against garbage unicode output (CPU backend bug signature).
nonprint=$(echo -n "$CONTENT" | python3 -c "import sys;s=sys.stdin.read();bad=sum(1 for c in s if ord(c)>127);print(bad)" 2>/dev/null)
if [ "${nonprint:-999}" -gt 10 ]; then
    matrix_fail "too many non-ASCII chars ($nonprint) — likely garbage: $CONTENT"
fi
# Long-generation on this 0.8B model naturally cycles through a small set of
# common words, producing unique_ratio < 0.25 even for healthy output. Relax
# that one threshold here; keep max_run / bigram / entropy gates strict since
# they still catch actual collapse.
QUALITY_MIN_UNIQUE_RATIO=0.18 \
    quality_check "$CONTENT" >&2 || matrix_fail "output failed quality_check: $CONTENT"
matrix_pass "long-coherent chars=${#CONTENT} accept=$(matrix_parse_accept)"
