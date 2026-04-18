#!/usr/bin/env bash
# Server starts cleanly with rollout=1, reaches /health, then stops cleanly.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
ROLLOUT=1
matrix_setup
if ! matrix_launch; then echo "CRASH: startup"; exit 3; fi
matrix_pass "server reached ready"
