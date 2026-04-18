#!/usr/bin/env bash
# Scheduler: alternate between two distinct shape decodes. Forces repeated
# reserve_n calls across the shape boundary.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
ROLLOUT=${ROLLOUT:-1}
matrix_setup
if ! matrix_launch; then echo "CRASH: init"; exit 3; fi

PROMPTS=("Hi" "The capital of France is Paris. Its population in the year 2000 was")
for i in 0 1 0 1 0; do
    matrix_completion "${PROMPTS[$i]}" 5 0
    if ! kill -0 $SRV_PID 2>/dev/null; then
        echo "CRASH: died on iteration with prompt='${PROMPTS[$i]}'"
        tail -10 "$LOG"
        exit 3
    fi
    if [ -z "$CONTENT" ]; then
        matrix_fail "empty on iter with prompt='${PROMPTS[$i]}'"
    fi
done
matrix_pass "alternating shapes survive"
