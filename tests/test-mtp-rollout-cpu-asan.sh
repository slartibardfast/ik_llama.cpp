#!/usr/bin/env bash
# CPU-ASAN variant of test-mtp-rollout-init.sh.
# Reproduces the LLAMA_MTP_ROLLOUT=3 crash using the CPU backend under
# AddressSanitizer. Requires build-cpu-asan/bin/llama-server.
#
# Purpose: localize the heap-corrupting write that causes the rollout>=2
# segfault. If the crash reproduces on CPU, ASAN pinpoints the write.
# If it doesn't, the corruption is Vulkan-backend-specific.
set -u

MODEL=${MODEL:-/opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0-MTP.gguf}
PORT=${PORT:-8227}
LOG=/tmp/mtp-rollout-asan.log
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="${BIN:-$SCRIPT_DIR/../build-cpu-asan/bin/llama-server}"

if [ ! -x "$BIN" ]; then echo "FAIL: $BIN not found"; exit 2; fi
if [ ! -f "$MODEL" ]; then echo "FAIL: model not found at $MODEL"; exit 2; fi

echo "=== test-mtp-rollout-cpu-asan — CPU ASAN reproduce ==="
echo "bin: $BIN"

# CPU-only: no -ngl. ASAN_OPTIONS: produce full stack, halt on error,
# detect heap overflow + use-after-free. LSAN_OPTIONS: suppress leaks
# we don't care about (glibc internals etc).
ASAN_OPTIONS="abort_on_error=1:halt_on_error=1:print_stacktrace=1:handle_segv=1:detect_stack_use_after_return=1" \
LSAN_OPTIONS="detect_leaks=0" \
LLAMA_MTP_ROLLOUT=3 \
    "$BIN" -m "$MODEL" --port "$PORT" --host 127.0.0.1 -c 2048 --parallel 1 -mtp -ub 32 -b 32 \
    > "$LOG" 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null; wait 2>/dev/null' EXIT

# CPU is slower — allow 180s for ready
for i in $(seq 1 180); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "ready after ${i}s"; break
    fi
    if ! kill -0 $SRV_PID 2>/dev/null; then
        echo "FAIL: server died during init (${i}s)"
        echo "--- ASAN output ---"
        tail -80 "$LOG"
        exit 1
    fi
    sleep 1
done

if ! kill -0 $SRV_PID 2>/dev/null; then
    echo "FAIL: server never ready"; tail -80 "$LOG"; exit 1
fi

OUT=$(curl -s -m 120 "http://127.0.0.1:$PORT/completion" \
    -d '{"prompt":"The capital of France is","n_predict":10,"temperature":0}')
kill $SRV_PID 2>/dev/null; wait 2>/dev/null

if [ -z "$OUT" ]; then
    echo "FAIL: empty response"; echo "--- ASAN output ---"
    tail -100 "$LOG"
    exit 1
fi

CONTENT=$(echo "$OUT" | python3 -c "import sys,json
try: print(json.load(sys.stdin).get('content',''), end='')
except: pass")

if [ -z "$CONTENT" ]; then
    echo "FAIL: no content"; tail -60 "$LOG"; exit 1
fi

echo "PASS: $CONTENT"
# If ASAN printed warnings even on pass, dump them
if grep -q "AddressSanitizer" "$LOG"; then
    echo "--- ASAN warnings ---"
    grep -A 30 "AddressSanitizer" "$LOG"
fi
exit 0
