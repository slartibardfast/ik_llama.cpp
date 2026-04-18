#!/usr/bin/env bash
# Regression anchor: baseline rollout=1 produces EXACTLY " Paris." as the
# first 6 characters of output. Byte-identical lock-in.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
ROLLOUT=1
matrix_setup
if ! matrix_launch; then echo "CRASH"; exit 3; fi
matrix_completion "The capital of France is" 10 0
if [ -z "$CONTENT" ]; then matrix_fail "empty"; fi
# First 6 chars should be " Paris."
prefix=$(printf "%s" "$CONTENT" | head -c 7)
if [ "$prefix" != " Paris." ]; then
    matrix_fail "expected ' Paris.' prefix, got '$prefix'"
fi
matrix_pass "exact match ' Paris.'"
