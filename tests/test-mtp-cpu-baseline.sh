#!/usr/bin/env bash
# CPU baseline (rollout=1) — does CPU backend achieve 33% acceptance like Vulkan?
# Uses the CPU-ASAN build.
set -u
MODEL=${MODEL:-/opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0-MTP.gguf}
PORT=${PORT:-8250}
LOG=/tmp/mtp-cpu-baseline.log
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="${BIN:-$SCRIPT_DIR/../build-cpu-asan/bin/llama-server}"

ASAN_OPTIONS="new_delete_type_mismatch=0:detect_leaks=0:halt_on_error=0" \
    "$BIN" -m "$MODEL" --port "$PORT" --host 127.0.0.1 -c 2048 --parallel 1 -mtp -ub 32 -b 32 \
    > "$LOG" 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null; wait 2>/dev/null' EXIT

for i in $(seq 1 180); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
    if ! kill -0 $SRV_PID 2>/dev/null; then echo "died at ${i}s"; tail -50 "$LOG"; exit 1; fi
    sleep 1
done

OUT=$(curl -s -m 120 "http://127.0.0.1:$PORT/completion" \
    -d '{"prompt":"The capital of France is","n_predict":30,"temperature":0}')
kill $SRV_PID 2>/dev/null; wait 2>/dev/null

content=$(echo "$OUT" | python3 -c "import sys,json;print(json.load(sys.stdin).get('content',''))" 2>/dev/null)
echo "content: $content" | head -3
echo
grep -E 'acceptance rate' "$LOG" | tail -1
