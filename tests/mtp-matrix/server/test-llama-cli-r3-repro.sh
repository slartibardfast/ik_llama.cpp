#!/usr/bin/env bash
# Server-free reproducer using llama-cli. This BYPASSES the HTTP server,
# speculative compat probe, and everything in examples/server/, confirming
# the bug is in llama.cpp's core decode path, not server-specific.
#
# CRASH (pre-fix): "corrupted double-linked list" in ggml_gallocr_reserve_n
# after first decode completes, before second decode.
# PASS (post-fix): llama-cli generates 10 tokens cleanly.
set -u
MODEL=${MODEL:-/opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0-MTP.gguf}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# Use the same binary as the matrix, but derive llama-cli path from BIN.
if [ -z "${BIN:-}" ]; then echo "SKIP: BIN not set"; exit 2; fi
LLAMA_CLI=$(dirname "$BIN")/llama-cli
if [ ! -x "$LLAMA_CLI" ]; then echo "SKIP: llama-cli not found at $LLAMA_CLI"; exit 2; fi
if [ ! -f "$MODEL" ]; then echo "SKIP: no model"; exit 2; fi

ROLLOUT=${ROLLOUT:-3}
LOG=/tmp/llama-cli-r$ROLLOUT-$$.log

# Match server config: -c 2048 -ub 32 -b 32, --no-warmup to avoid warmup crashes.
# shellcheck disable=SC2086
${EXTRA_ENV:-} LLAMA_MTP_ROLLOUT=$ROLLOUT timeout 60 "$LLAMA_CLI" \
    -m "$MODEL" -mtp -c 2048 -ub 32 -b 32 --no-warmup \
    -n 10 -p "The capital of France is" \
    $EXTRA_ARGS > "$LOG" 2>&1
rc=$?

if grep -qE "corrupted double-linked list|Aborted|Segmentation fault|SIGSEGV" "$LOG"; then
    echo "CRASH: llama-cli heap corruption at rollout=$ROLLOUT"
    tail -15 "$LOG"
    rm -f "$LOG"
    exit 3
fi
if [ $rc -eq 124 ]; then
    echo "CRASH: llama-cli timeout at rollout=$ROLLOUT"
    tail -15 "$LOG"
    rm -f "$LOG"
    exit 3
fi
if [ $rc -ne 0 ]; then
    echo "FAIL: llama-cli exit=$rc"
    tail -10 "$LOG"
    rm -f "$LOG"
    exit 1
fi
# Extract the generated text (rough heuristic): the "The capital of France is"
# line should be followed by generated tokens.
out=$(grep -A 1 "France is" "$LOG" 2>/dev/null | tail -1 | head -c 100)
echo "PASS: llama-cli completed, output: $out"
rm -f "$LOG"
exit 0
