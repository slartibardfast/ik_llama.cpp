#!/usr/bin/env bash
# Server starts cleanly with rollout=3 (chained rollout).
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
ROLLOUT=3
matrix_setup
if ! matrix_launch; then echo "CRASH: startup at rollout=3"; tail -5 "$LOG"; exit 3; fi
matrix_pass "rollout=3 server reached ready"
