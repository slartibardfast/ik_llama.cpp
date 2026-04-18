#!/usr/bin/env bash
# Regression anchor: two back-to-back identical requests produce byte-
# identical outputs. Locks in determinism guarantee.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
ROLLOUT=${ROLLOUT:-1}
matrix_setup
if ! matrix_launch; then echo "CRASH"; exit 3; fi
matrix_completion "The capital of France is" 10 0
A="$CONTENT"
matrix_completion "The capital of France is" 10 0
B="$CONTENT"
if [ "$A" != "$B" ]; then matrix_fail "A='$A' B='$B'"; fi
matrix_pass "A==B: '$A'"
