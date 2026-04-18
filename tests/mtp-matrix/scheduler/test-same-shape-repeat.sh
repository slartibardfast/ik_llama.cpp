#!/usr/bin/env bash
# Scheduler: send MANY requests of the same shape. Exercises the compute-
# after-compute path (no shape changes). Catches use-after-free or state
# corruption across decodes.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
ROLLOUT=${ROLLOUT:-1}
matrix_setup
if ! matrix_launch; then echo "CRASH: init"; exit 3; fi

# 5 identical requests
for i in 1 2 3 4 5; do
    matrix_completion "The capital of France is" 5 0
    if ! kill -0 $SRV_PID 2>/dev/null; then
        echo "CRASH: died on request $i"
        tail -10 "$LOG"
        exit 3
    fi
    if [ -z "$CONTENT" ]; then
        matrix_fail "empty content on request $i"
    fi
done
matrix_pass "5 identical requests"
