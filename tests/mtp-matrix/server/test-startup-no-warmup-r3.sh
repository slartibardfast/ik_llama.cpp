#!/usr/bin/env bash
# Server starts with rollout=3 AND --no-warmup. Tests that init succeeds
# without relying on warmup for buffer sizing.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
ROLLOUT=3
EXTRA_ARGS="${EXTRA_ARGS:-} --no-warmup"
matrix_setup
if ! matrix_launch; then echo "CRASH: startup"; exit 3; fi
matrix_pass "--no-warmup rollout=3 server started"
