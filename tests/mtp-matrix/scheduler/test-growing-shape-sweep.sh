#!/usr/bin/env bash
# Scheduler: send prompts of monotonically GROWING size. Each request
# may force a reserve_n grow. Finds the specific shape that triggers the
# crash, if any.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
ROLLOUT=${ROLLOUT:-1}
matrix_setup
if ! matrix_launch; then echo "CRASH: init"; exit 3; fi

LAST_OK=0
for n in 1 2 3 5 8 13 21; do
    local_prompt=$(python3 -c "print(' '.join(['the']*$n))")
    matrix_completion "$local_prompt" 3 0
    if ! kill -0 $SRV_PID 2>/dev/null; then
        echo "CRASH: died at shape n=$n (last ok=$LAST_OK)"
        tail -10 "$LOG"
        exit 3
    fi
    if [ -n "$CONTENT" ]; then LAST_OK=$n; fi
done
matrix_pass "survived growing-shape sweep up to n=$LAST_OK"
