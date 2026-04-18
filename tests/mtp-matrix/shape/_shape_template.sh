#!/usr/bin/env bash
# Shape-boundary test template. The concrete test scripts source this with
# TOKENS and ROLLOUT set, then execute.
#
# A shape-test prompt forces the server to see EXACTLY n_tokens tokens in
# the prompt. We approximate this by repeating a single-token word.
matrix_shape_test() {
    local tokens=$1
    local rollout=$2
    # Build a prompt of ~$tokens short words (each tokenizes to ~1 token
    # for most Latin-alphabet words).
    local prompt
    prompt=$(python3 -c "print(' '.join(['the']*$tokens))")

    export ROLLOUT=$rollout
    matrix_setup
    if ! matrix_launch; then echo "CRASH: startup at tokens=$tokens rollout=$rollout"; exit 3; fi
    matrix_completion "$prompt" 5 0
    if [ -z "$OUT" ] || [ -z "$CONTENT" ]; then
        matrix_fail "empty response at tokens=$tokens rollout=$rollout"
    fi
    matrix_pass "tokens=$tokens rollout=$rollout content_len=${#CONTENT}"
}
