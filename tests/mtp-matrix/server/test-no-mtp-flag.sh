#!/usr/bin/env bash
# Server starts WITHOUT -mtp flag on an MTP-capable model. Exercises the
# non-MTP path to ensure no regression from MTP-specific code.
# Uses custom launch (not matrix_launch's -mtp default).
set -u
MODEL=${MODEL:-/opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0-MTP.gguf}
PORT=${PORT:-$((8500 + $$ % 500))}
if [ -z "${BIN:-}" ]; then echo "SKIP: BIN not set"; exit 2; fi
if [ ! -x "$BIN" ]; then echo "SKIP: BIN not exec"; exit 2; fi
if [ ! -f "$MODEL" ]; then echo "SKIP: no model"; exit 2; fi

LOG=/tmp/no-mtp-$$.log
# shellcheck disable=SC2086
${EXTRA_ENV:-} "$BIN" -m "$MODEL" --port "$PORT" --host 127.0.0.1 \
    -c 2048 --parallel 1 -ub 32 -b 32 $EXTRA_ARGS > "$LOG" 2>&1 &
SRV=$!
for i in $(seq 1 120); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "PASS: non-MTP server started at ${i}s"
        kill $SRV 2>/dev/null
        wait 2>/dev/null
        rm -f "$LOG"
        exit 0
    fi
    if ! kill -0 $SRV 2>/dev/null; then
        echo "CRASH: non-MTP server died at ${i}s"
        tail -15 "$LOG"
        rm -f "$LOG"
        exit 3
    fi
    sleep 1
done
echo "CRASH: never ready (120s)"
kill $SRV 2>/dev/null; wait 2>/dev/null
tail -15 "$LOG"
rm -f "$LOG"
exit 3
