#!/usr/bin/env bash
# Graph-metrics regression anchor.
#
# Captures Vulkan scheduler efficiency invariants that any code change must
# not regress. Backgrounded by prior set_input fix
# (src/llama-build-context.cpp:4907, 4964) which dropped splits from
# 11 → 1 at r=5 and Vulkan_Host compute buffer from 30.47 MiB → 0.14 MiB.
#
# If a future edit brings splits back or inflates host-visible staging,
# this test fails fast instead of showing up as a silent perf regression.
#
# Usage:
#   EXTRA_ARGS="-ngl 999" EXTRA_ENV="GGML_VK_DISABLE_MMVQ=1" \
#       BIN=./build-vk/bin/llama-server \
#       bash test-graph-metrics.sh
#
# Invariants (Vulkan, NAVI21 + Vega10 2-GPU, MMVQ off):
#   r=1: splits == 1, nodes ≤ 1250, host_buf ≤ 0.5 MiB
#   r=5: splits == 1, nodes ≤ 1400, host_buf ≤ 0.5 MiB
#
# Note on `failed to allocate graph, reserving ...` log lines: these fire 3-4
# times during normal init+prompt-eval+first-decode transitions and do NOT
# indicate a problem. They counted the same in pre- and post-set_input-fix
# states, so they aren't a useful regression signal. The `graph splits`
# metric is what catches the real scheduler-churn regression.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"

# Max thresholds — set slightly above measured post-fix values to absorb
# normal jitter. Any regression beyond these is a real scheduler regression.
MAX_SPLITS=1
MAX_NODES_R1=1250
MAX_NODES_R5=1400
MAX_HOST_MIB="0.5"

any_fail=0

check_rollout() {
    local r=$1
    local max_nodes=$2
    local log=/tmp/graph-metrics-$$-r$r.log

    ROLLOUT=$r
    PORT=$((PORT + 1))
    LOG=$log
    MATRIX_KEEP_LOG=1
    matrix_setup
    if ! matrix_launch; then
        echo "FAIL: r=$r server did not start"
        any_fail=1
        return 1
    fi
    # One tiny request so the scheduler allocates its graph (reserve-on-first-decode).
    matrix_completion "hi" 3 0 > /dev/null

    local splits nodes host
    splits=$(awk -F'=' '/graph splits/ {gsub(/ /,"",$2); print $2; exit}' "$log")
    nodes=$(awk -F'=' '/graph nodes/ {gsub(/ /,"",$2); print $2; exit}' "$log")
    host=$(awk '/Vulkan_Host compute buffer size/ {for(i=1;i<=NF;i++) if($i=="MiB") {print $(i-1); exit}}' "$log")

    printf "  r=%d: splits=%s nodes=%s host_MiB=%s\n" \
        "$r" "$splits" "$nodes" "$host"

    if [ "${splits:-999}" -gt "$MAX_SPLITS" ]; then
        echo "  FAIL r=$r: splits=$splits > $MAX_SPLITS"; any_fail=1
    fi
    if [ "${nodes:-999999}" -gt "$max_nodes" ]; then
        echo "  FAIL r=$r: nodes=$nodes > $max_nodes"; any_fail=1
    fi
    if awk -v h="${host:-9999}" -v m="$MAX_HOST_MIB" 'BEGIN{exit !(h+0 > m+0)}'; then
        echo "  FAIL r=$r: host_MiB=$host > $MAX_HOST_MIB"; any_fail=1
    fi

    matrix_teardown
    rm -f "$log"
}

echo "=== graph-metrics regression ==="
check_rollout 1 "$MAX_NODES_R1"
check_rollout 5 "$MAX_NODES_R5"

if [ "$any_fail" -ne 0 ]; then
    echo "FAIL: graph-metrics regression detected"
    exit 1
fi
echo "PASS: graph metrics within thresholds"
exit 0
