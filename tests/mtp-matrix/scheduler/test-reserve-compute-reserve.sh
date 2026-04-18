#!/usr/bin/env bash
# Scheduler: init reserve, THEN warmup (compute), THEN shape-change reserve.
# The classic failure pattern. If this PASSes, the bug isn't reserve-after-compute.
# If this FAILs where reserve-reserve-no-compute PASSes, we've localized.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
ROLLOUT=${ROLLOUT:-3}
matrix_setup
if ! matrix_launch; then echo "CRASH: init/warmup"; exit 3; fi
# Warmup already ran inside matrix_launch (implicit via llama_init_from_params).
# First decode triggers shape-change reserve_n.
matrix_completion "Hi" 3 0
if ! kill -0 $SRV_PID 2>/dev/null; then
    echo "CRASH: reserve-after-warmup-compute"
    tail -15 "$LOG"
    exit 3
fi
matrix_pass "reserve-compute-reserve survives"
