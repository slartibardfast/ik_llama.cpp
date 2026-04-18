#!/usr/bin/env bash
# Throughput benchmark: does chained rollout win vs single-pass?
#
# Sweeps rollout ∈ {1, 2, 3, 4, 5} on a fixed prompt + n_predict.
# Records: prompt-eval t/s, generation t/s, acceptance rate, wall-clock.
# Effective t/s ≈ generation_tps × (1 + accept × (rollout − 1))
#
# Usage:
#   BIN=build-vk/bin/llama-server EXTRA_ARGS="-ngl 999" \
#       EXTRA_ENV="GGML_VK_DISABLE_MMVQ=1" \
#       bash bench-rollout-throughput.sh > results.txt
#
# Writes a plain-text table to stdout. ROLLOUTS env overrides defaults.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"

PROMPT=${PROMPT:-"Explain why the sky appears blue during daytime in two sentences."}
N_PREDICT=${N_PREDICT:-100}
ROLLOUTS=${ROLLOUTS:-"1 2 3 4 5"}
WARMUP=${WARMUP:-1}
TRIALS=${TRIALS:-3}

printf "rollout\tprompt_tps\tgen_tps\taccept\teff_tps\twall_ms\n"

for r in $ROLLOUTS; do
    ROLLOUT=$r
    matrix_setup
    if ! matrix_launch; then
        printf "%d\tCRASH\n" "$r"
        continue
    fi

    if [ "$WARMUP" = "1" ]; then
        matrix_completion "$PROMPT" 5 0 > /dev/null
    fi

    best_wall=0
    best_gen=0
    best_prompt=0
    best_accept=0
    for trial in $(seq 1 "$TRIALS"); do
        start_ns=$(date +%s%N)
        matrix_completion "$PROMPT" "$N_PREDICT" 0 > /dev/null
        end_ns=$(date +%s%N)
        wall_ms=$(( (end_ns - start_ns) / 1000000 ))

        # Lines look like:
        #   prompt eval time =  X ms / N tokens (  Y ms per token,    Z tokens per second)
        #          eval time =  X ms / N tokens (  Y ms per token,    Z tokens per second)
        # Need the number AFTER the comma, not before it.
        gen_tps=$(grep 'eval time' "$LOG" | grep -v 'prompt eval' | tail -1 | \
            awk -F'[(,]' '{print $3}' | awk '{print $1}')
        prompt_tps=$(grep 'prompt eval time' "$LOG" | tail -1 | \
            awk -F'[(,]' '{print $3}' | awk '{print $1}')
        accept=$(matrix_parse_accept)

        # Keep best (fastest) trial
        if [ "$trial" = "1" ] || [ "$wall_ms" -lt "$best_wall" ]; then
            best_wall=$wall_ms
            best_gen=${gen_tps:-0}
            best_prompt=${prompt_tps:-0}
            best_accept=${accept:-0}
        fi
    done

    eff_tps=$(python3 -c "
g=${best_gen:-0}; a=${best_accept:-0}; r=$r
print(f'{g*(1+a*(r-1)):.2f}')
" 2>/dev/null || echo "0")

    printf "%d\t%s\t%s\t%s\t%s\t%d\n" \
        "$r" "${best_prompt:-0}" "${best_gen:-0}" \
        "${best_accept:-0}" "$eff_tps" "$best_wall"

    matrix_teardown
done
