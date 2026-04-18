#!/usr/bin/env bash
# Sweep diverse prompts at rollout=3 to characterize the bench outlier
# seen on the "sky" prompt (accept=0.032 at r=3 vs 0.55-0.68 at r=2/4/5).
#
# Not a pass/fail test — prints per-prompt acceptance for documentation.
# Diagnoses whether the low-accept is prompt-specific or systemic at r=3.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
ROLLOUT=${ROLLOUT:-3}
matrix_setup
if ! matrix_launch; then echo "CRASH"; exit 3; fi

PROMPTS=(
    "The capital of France is"
    "Explain why the sky appears blue during daytime"
    "The best programming language for systems is"
    "In 1969 humans first landed on"
    "Water is made of two elements:"
    "A quick brown fox jumps over the lazy"
)

printf "rollout=%d prompt-sweep\n" "$ROLLOUT"
printf "%-50s  %s\n" "prompt" "acceptance"
for p in "${PROMPTS[@]}"; do
    matrix_completion "$p" 30 0 > /dev/null
    a=$(matrix_parse_accept)
    printf "%-50s  %s\n" "$(printf '%s' "$p" | head -c 48)" "${a:-?}"
done
matrix_teardown
echo "done"
