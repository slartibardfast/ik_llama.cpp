#!/usr/bin/env bash
# Scheduler: reserve at shape A, reserve at shape B, no compute between.
# Does the free+calloc in reserve_n crash WITHOUT prior compute?
#
# Strategy: start server with one config (doesn't run warmup via --no-warmup),
# that triggers init reserve only. Then send a completion request — this
# builds new graph for runtime (shape change) → triggers reserve_n.
# If server survives = driver bug requires compute between reserves.
# If server crashes = driver bug triggers on ANY second reserve_n.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
EXTRA_ARGS="${EXTRA_ARGS:-} --no-warmup"
ROLLOUT=${ROLLOUT:-3}
matrix_setup
if ! matrix_launch; then echo "CRASH: init (no warmup)"; exit 3; fi
# Try a tiny completion - this triggers the first runtime decode reserve_n.
matrix_completion "Hi" 3 0
if ! kill -0 $SRV_PID 2>/dev/null; then
    echo "CRASH: server died on first decode reserve (no prior compute)"
    tail -15 "$LOG"
    exit 3
fi
matrix_pass "first decode reserve_n survives"
