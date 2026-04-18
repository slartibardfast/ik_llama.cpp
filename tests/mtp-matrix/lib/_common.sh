#!/usr/bin/env bash
# Shared helpers for mtp-matrix tests.
#
# Environment:
#   MODEL   - model .gguf path (required)
#   BIN     - llama-server path (required)
#   PORT    - base port (optional; tests allocate based on PID/PORT)
#   ROLLOUT - LLAMA_MTP_ROLLOUT value (optional, default 1)
#   EXTRA_ARGS - extra argv to llama-server (optional)
#   EXTRA_ENV  - extra env vars prefix (e.g. "GGML_VK_DISABLE_MMVQ=1")
#
# Each test script sources this and calls:
#   matrix_setup        # parses env, exports paths, sets up log/port
#   matrix_launch       # starts server in background, waits for /health
#   matrix_completion   # sends a /completion request, returns JSON in $OUT
#   matrix_parse_accept # reads server log, echoes acceptance rate
#   matrix_teardown     # kills server, cleans up
#
# Exit codes:
#   0  = PASS
#   1  = FAIL (test assertion violated)
#   2  = SKIP (infrastructure not available — missing binary/model)
#   3  = CRASH (server died during test)

set -u

MODEL=${MODEL:-/opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0-MTP.gguf}
PORT=${PORT:-$(( 8300 + $$ % 1000 ))}
ROLLOUT=${ROLLOUT:-1}
EXTRA_ARGS=${EXTRA_ARGS:-}
EXTRA_ENV=${EXTRA_ENV:-}

# Required env: BIN
if [ -z "${BIN:-}" ]; then
    echo "SKIP: BIN not set (expected path to llama-server binary)"
    exit 2
fi
if [ ! -x "$BIN" ]; then
    echo "SKIP: BIN=$BIN not executable"
    exit 2
fi
if [ ! -f "$MODEL" ]; then
    echo "SKIP: MODEL=$MODEL not found"
    exit 2
fi

LOG=${LOG:-/tmp/mtp-matrix-$$-$ROLLOUT.log}
SRV_PID=
OUT=

matrix_setup() {
    # Clean any prior state
    rm -f "$LOG"
    # Wait for the port to become free — a prior test's TCP socket may be in
    # TIME_WAIT for a few seconds after the server exits. Previously this
    # returned SKIP immediately, causing chains of sequential test runs to
    # fail with cryptic exit-2 output.
    local wait_limit=${PORT_WAIT_TIMEOUT:-30}
    local waited=0
    while [ "$waited" -lt "$wait_limit" ]; do
        if ! ss -tln 2>/dev/null | grep -q ":$PORT "; then
            break
        fi
        if [ "$waited" -eq 0 ]; then
            # First occupied observation: show which process currently owns the port.
            local owner
            owner=$(ss -tlnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p {print $NF}' | head -1)
            echo "matrix_setup: port $PORT busy (owner: ${owner:-unknown}), waiting up to ${wait_limit}s..." >&2
        fi
        sleep 1
        waited=$((waited + 1))
    done
    if ss -tln 2>/dev/null | grep -q ":$PORT "; then
        echo "SKIP: port $PORT still in use after ${wait_limit}s"
        exit 2
    fi
}

# Launch the server. Returns 0 if ready, 3 if died during init.
matrix_launch() {
    local env_cmd=""
    if [ -n "$EXTRA_ENV" ]; then env_cmd="$EXTRA_ENV"; fi

    # shellcheck disable=SC2086
    eval "$env_cmd LLAMA_MTP_ROLLOUT=$ROLLOUT \"$BIN\" \
        -m \"$MODEL\" \
        --port $PORT --host 127.0.0.1 \
        -c 2048 --parallel 1 -mtp -ub 32 -b 32 \
        $EXTRA_ARGS \
        > \"$LOG\" 2>&1 &"
    SRV_PID=$!

    local ready_timeout=${READY_TIMEOUT:-120}
    for i in $(seq 1 $ready_timeout); do
        if curl -s -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 $SRV_PID 2>/dev/null; then
            echo "CRASH: server died during init (after ${i}s)"
            return 3
        fi
        sleep 1
    done
    echo "CRASH: server never ready (${ready_timeout}s)"
    return 3
}

# Send a completion. Args: prompt (string), n_predict (int, default 10).
# Sets OUT (raw JSON) and CONTENT (parsed content).
matrix_completion() {
    local prompt=${1:-"The capital of France is"}
    local n_predict=${2:-10}
    local temperature=${3:-0}

    OUT=$(curl -s -m 120 "http://127.0.0.1:$PORT/completion" \
        -d "{\"prompt\":\"$prompt\",\"n_predict\":$n_predict,\"temperature\":$temperature}")

    CONTENT=$(echo "$OUT" | python3 -c "import sys,json
try:
    d = json.load(sys.stdin)
    print(d.get('content', ''), end='')
except Exception:
    pass" 2>/dev/null)
}

# Read the acceptance rate from the server's log (set after completion).
# Echoes e.g. 0.33333 or empty.
matrix_parse_accept() {
    grep 'acceptance rate' "$LOG" | tail -1 | \
        sed -n 's/.*acceptance rate = \([0-9.]*\).*/\1/p'
}

matrix_parse_tps() {
    grep 'eval time' "$LOG" | tail -1 | \
        sed -n 's/.*(\([0-9.]*\) tokens per second.*/\1/p'
}

matrix_teardown() {
    if [ -n "${SRV_PID:-}" ]; then
        # SIGTERM first, wait briefly, then SIGKILL if still alive. Leaving
        # the server hung in a half-closed state keeps the port in TIME_WAIT
        # longer than the next test's matrix_setup can tolerate.
        kill $SRV_PID 2>/dev/null
        local max_wait=${TEARDOWN_WAIT:-8}
        local waited=0
        while [ "$waited" -lt "$max_wait" ] && kill -0 $SRV_PID 2>/dev/null; do
            sleep 1
            waited=$((waited + 1))
        done
        if kill -0 $SRV_PID 2>/dev/null; then
            kill -9 $SRV_PID 2>/dev/null
            wait 2>/dev/null
        fi
    fi
    if [ "${MATRIX_KEEP_LOG:-0}" != "1" ]; then
        rm -f "$LOG"
    fi
}

# Utility: fail with a message, dumping the last bit of log.
matrix_fail() {
    local msg=$1
    echo "FAIL: $msg"
    echo "--- log tail ---"
    tail -20 "$LOG" 2>/dev/null
    matrix_teardown
    exit 1
}

matrix_pass() {
    local msg=${1:-ok}
    echo "PASS: $msg"
    matrix_teardown
    exit 0
}
