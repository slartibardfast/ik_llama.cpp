#!/usr/bin/env bash
# Server starts with --parallel 2 (2 concurrent slots). Catches init paths
# that don't scale beyond 1 slot.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
ROLLOUT=1
# Override the launch line's "--parallel 1" by adding late; llama-server uses the last one.
EXTRA_ARGS="${EXTRA_ARGS:-} --parallel 2"
matrix_setup
if ! matrix_launch; then echo "CRASH: startup parallel=2"; exit 3; fi
matrix_pass "parallel=2 server started"
