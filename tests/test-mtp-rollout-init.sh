#!/usr/bin/env bash
# Reproduces the LLAMA_MTP_ROLLOUT=3 compute-buffer realloc crash
# (crash (b) per project_ik_llama_mtp_ir_port.md).
#
# Success (post-fix): rollout=3 produces non-empty output without segfault.
# Failure (pre-fix):  segfault or empty response — stderr log tail shown.
#
# Output coherence is NOT checked here — Track A (rollback) / Track B
# (pos-0 invariant) can legitimately degrade coherence. This test gates
# the server-doesn't-crash question only.
set -u

MODEL=${MODEL:-/opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0-MTP.gguf}
PORT=${PORT:-8197}
LOG=/tmp/mtp-rollout-test.log
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="${BIN:-$SCRIPT_DIR/../build-vk/bin/llama-server}"

if [ ! -x "$BIN" ]; then
    echo "FAIL: $BIN not found — build llama-server first"; exit 2
fi
if [ ! -f "$MODEL" ]; then
    echo "FAIL: model not found at $MODEL"; exit 2
fi

echo "=== test-mtp-rollout-init — LLAMA_MTP_ROLLOUT=3 compute buffer unblock ==="
echo "model: $MODEL"
echo "bin:   $BIN"

GGML_VK_DISABLE_MMVQ=1 LLAMA_MTP_ROLLOUT=3 \
    "$BIN" -m "$MODEL" -ngl 999 \
    --port "$PORT" --host 127.0.0.1 -c 2048 --parallel 1 -mtp -ub 32 -b 32 \
    > "$LOG" 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null; wait 2>/dev/null' EXIT

# Wait up to 60s for server ready or death
for i in $(seq 1 60); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "server ready after ${i}s"
        break
    fi
    if ! kill -0 $SRV_PID 2>/dev/null; then
        echo "FAIL: server died during init (after ${i}s)"
        echo "--- log tail ---"
        tail -25 "$LOG"
        exit 1
    fi
    sleep 1
done

if ! kill -0 $SRV_PID 2>/dev/null; then
    echo "FAIL: server never ready (60s timeout)"
    tail -25 "$LOG"; exit 1
fi

# Make a request — 10 tokens is enough to trigger the crash if present.
OUT=$(curl -s -m 30 "http://127.0.0.1:$PORT/completion" \
    -d '{"prompt":"The capital of France is","n_predict":10,"temperature":0}')

# Kill server now; results already captured in $OUT
kill $SRV_PID 2>/dev/null
wait 2>/dev/null

if [ -z "$OUT" ]; then
    echo "FAIL: empty response (likely segfault)"
    echo "--- log tail ---"
    tail -25 "$LOG"
    exit 1
fi

CONTENT=$(echo "$OUT" | python3 -c "import sys,json
try:
    d = json.load(sys.stdin)
    print(d.get('content',''), end='')
except Exception as e:
    print('PARSE_ERROR', file=sys.stderr)")

if [ -z "$CONTENT" ]; then
    echo "FAIL: no content in response"
    echo "response: $(echo "$OUT" | head -c 300)"
    echo "--- log tail ---"
    tail -25 "$LOG"
    exit 1
fi

echo "PASS: got $(printf "%s" "$CONTENT" | wc -c) chars"
echo "output: $CONTENT" | head -5
exit 0
