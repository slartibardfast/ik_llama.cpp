#!/usr/bin/env bash
# Regression anchor: baseline rollout=1 achieves ≥ 0.25 acceptance rate
# (0.33333 is current, allow 8pp slack for minor drift).
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
ROLLOUT=1
matrix_setup
if ! matrix_launch; then echo "CRASH"; exit 3; fi
matrix_completion "The capital of France is" 30 0
accept=$(matrix_parse_accept)
if [ -z "$accept" ]; then matrix_fail "no acceptance rate in log"; fi
ok=$(python3 -c "import sys;a=float('$accept');print(1 if a>=0.25 else 0)")
if [ "$ok" != "1" ]; then matrix_fail "accept $accept < 0.25"; fi
matrix_pass "accept=$accept ≥ 0.25"
