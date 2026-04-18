#!/usr/bin/env bash
# Regression anchor: baseline rollout=1 achieves ≥ 8 t/s on eval
# (12 is current Vulkan, allow 30% slack).
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
ROLLOUT=1
matrix_setup
if ! matrix_launch; then echo "CRASH"; exit 3; fi
matrix_completion "The capital of France is" 30 0
tps=$(matrix_parse_tps)
if [ -z "$tps" ]; then matrix_fail "no t/s in log"; fi
ok=$(python3 -c "import sys;t=float('$tps');print(1 if t>=8.0 else 0)")
if [ "$ok" != "1" ]; then matrix_fail "t/s $tps < 8.0"; fi
matrix_pass "t/s=$tps ≥ 8.0"
