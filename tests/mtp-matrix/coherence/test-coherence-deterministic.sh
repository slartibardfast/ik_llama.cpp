#!/usr/bin/env bash
# Deterministic repeat: same prompt, temp=0, must produce byte-identical
# output across two consecutive calls. Catches use-after-free, state leaks,
# and non-determinism.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
ROLLOUT=${ROLLOUT:-1}
matrix_setup
if ! matrix_launch; then echo "CRASH"; exit 3; fi

matrix_completion "The capital of France is" 10 0
FIRST="$CONTENT"
matrix_completion "The capital of France is" 10 0
SECOND="$CONTENT"

if [ "$FIRST" != "$SECOND" ]; then
    matrix_fail "non-deterministic: run1='$FIRST' run2='$SECOND'"
fi
matrix_pass "deterministic across 2 runs: $FIRST"
