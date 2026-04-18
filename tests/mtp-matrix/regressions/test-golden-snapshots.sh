#!/usr/bin/env bash
# Golden-snapshot regression anchor.
#
# Runs fixed prompts at rollout=1 against current binary and compares 30-token
# outputs byte-for-byte against committed snapshots. Catches ANY numerical
# drift in the decode path — including silent regressions where output stays
# "mostly coherent" but characters shift.
#
# Usage:
#   bash test-golden-snapshots.sh            # compare against snapshots
#   UPDATE_SNAPSHOTS=1 bash test-golden-snapshots.sh   # regenerate snapshots
#
# Snapshots live at tests/mtp-matrix/snapshots/<backend-tag>-r1-<slug>.txt.
# Use SNAP_TAG env to name snapshots per backend (defaults to "vulkan").
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
SNAPS_DIR="$SCRIPT_DIR/../snapshots"
SNAP_TAG=${SNAP_TAG:-vulkan}
ROLLOUT=1

PROMPTS=(
    "capital:The capital of France is"
    "moon:In 1969 humans first landed on"
    "lang:The best programming language for systems is"
    "sky:Explain why the sky appears blue during daytime"
)

matrix_setup
if ! matrix_launch; then
    echo "CRASH: server did not start"; tail -10 "$LOG" 2>/dev/null; exit 3
fi

mkdir -p "$SNAPS_DIR"
any_diff=0
for row in "${PROMPTS[@]}"; do
    slug=${row%%:*}
    prompt=${row#*:}
    matrix_completion "$prompt" 30 0
    if [ -z "$CONTENT" ]; then
        echo "FAIL: empty content for '$slug'"; any_diff=1; continue
    fi
    snap="$SNAPS_DIR/${SNAP_TAG}-r1-${slug}.txt"
    if [ "${UPDATE_SNAPSHOTS:-0}" = "1" ]; then
        printf '%s' "$CONTENT" > "$snap"
        echo "wrote $snap (${#CONTENT} chars)"
        continue
    fi
    if [ ! -f "$snap" ]; then
        echo "FAIL: no snapshot for '$slug' — run with UPDATE_SNAPSHOTS=1 first"
        any_diff=1; continue
    fi
    expected=$(cat "$snap")
    if [ "$CONTENT" != "$expected" ]; then
        echo "FAIL: '$slug' drift vs $snap"
        echo "  expected: $(printf '%s' "$expected" | head -c 80)"
        echo "  actual:   $(printf '%s' "$CONTENT" | head -c 80)"
        any_diff=1
    else
        echo "ok: $slug"
    fi
done

matrix_teardown
if [ "${UPDATE_SNAPSHOTS:-0}" = "1" ]; then
    echo "PASS: snapshots updated"
    exit 0
fi
if [ $any_diff -ne 0 ]; then
    echo "FAIL: one or more snapshots drifted"
    exit 1
fi
echo "PASS: all snapshots match"
exit 0
