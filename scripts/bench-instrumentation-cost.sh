#!/usr/bin/env bash
#
# PHASE46 B.7 — Determinism instrumentation cost bench.
#
# Measures the throughput delta between two runs of the same workload:
#   (a) baseline:     LLAMA_TRACE_NDJSON unset (zero-cost trace path)
#   (b) instrumented: LLAMA_TRACE_NDJSON set, full per-step trace emit
#
# Asserts the delta is within the documented ceiling (≤2% steady-state).
# Run config matches the production target: np=3, ctx=262144, --draft 3.
#
# Pre-reqs: a Qwen3.6 (or compatible) GGUF model + a built llama-server
# binary that links the post-PHASE46 B.4 libllama with the spec_loop
# trace hooks compiled in.
#
# Track-C addendum: once fork/join is wired into the bug-prone ops, the
# script will also measure the fork/join-mechanism delta vs single-slot
# reference (≤4% gate). Currently that bind is a TODO marker; it will be
# uncommented when C.1/C.2 land.

set -euo pipefail

# ------------------------------------------------------------------ args
SERVER_BIN="${SERVER_BIN:-build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-}"
N_PARALLEL="${N_PARALLEL:-3}"
CTX_SIZE="${CTX_SIZE:-262144}"
N_DRAFT="${N_DRAFT:-3}"
N_TOKENS_BENCH="${N_TOKENS_BENCH:-1024}"
N_RUNS="${N_RUNS:-5}"
GATE_PCT="${GATE_PCT:-2.0}"
PORT="${PORT:-18080}"

# ------------------------------------------------------------------ usage
usage() {
    cat <<USAGE
usage: MODEL_PATH=<path-to-gguf> [SERVER_BIN=...] $0

env knobs (with defaults):
  SERVER_BIN      $SERVER_BIN
  MODEL_PATH      (required)
  N_PARALLEL      $N_PARALLEL
  CTX_SIZE        $CTX_SIZE
  N_DRAFT         $N_DRAFT
  N_TOKENS_BENCH  $N_TOKENS_BENCH    tokens to generate per slot per run
  N_RUNS          $N_RUNS    runs per condition (mean reported)
  GATE_PCT        $GATE_PCT    fail if instrumented > baseline + this %
  PORT            $PORT  server listen port

emits two condition results (baseline / instrumented) and exits non-zero
if the instrumented run exceeds GATE_PCT throughput regression.
USAGE
}

if [[ -z "$MODEL_PATH" ]]; then
    usage >&2
    exit 2
fi
if [[ ! -x "$SERVER_BIN" ]]; then
    echo "error: SERVER_BIN '$SERVER_BIN' not executable" >&2
    exit 2
fi
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "error: MODEL_PATH '$MODEL_PATH' not found" >&2
    exit 2
fi

# ------------------------------------------------------------------ helpers
TMPDIR=$(mktemp -d -t llama-bench-XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT

server_pid=
start_server() {
    local trace_path="${1:-}"
    local env_args=()
    if [[ -n "$trace_path" ]]; then
        env_args+=("LLAMA_TRACE_NDJSON=$trace_path")
    fi
    env "${env_args[@]}" "$SERVER_BIN" \
        -m "$MODEL_PATH" \
        --port "$PORT" \
        --host 127.0.0.1 \
        -np "$N_PARALLEL" \
        --draft "$N_DRAFT" \
        -c "$CTX_SIZE" \
        -ngl 999 \
        --log-disable \
        > "$TMPDIR/server.log" 2>&1 &
    server_pid=$!
    # wait for /health
    local i
    for i in $(seq 1 60); do
        if curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "error: server failed to come up; tail of log:" >&2
    tail -50 "$TMPDIR/server.log" >&2 || true
    return 1
}

stop_server() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        server_pid=
    fi
}

# Single bench iteration: completion endpoint, fixed prompt, N tokens,
# N_PARALLEL concurrent requests. Returns aggregate t/s as plain number.
bench_one() {
    local tag="$1"
    local out="$TMPDIR/$tag.json"
    local start_ns end_ns elapsed_s tps
    start_ns=$(date +%s%N)
    for slot in $(seq 1 "$N_PARALLEL"); do
        curl -fsS \
            -H 'content-type: application/json' \
            -d '{"prompt":"Write a short haiku about deterministic computation.","n_predict":'"$N_TOKENS_BENCH"',"temperature":0,"seed":42}' \
            "http://127.0.0.1:$PORT/completion" \
            >> "$out" 2>/dev/null &
    done
    wait
    end_ns=$(date +%s%N)
    elapsed_s=$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN{printf "%.6f", (e-s)/1e9}')
    tps=$(awk -v t="$N_TOKENS_BENCH" -v p="$N_PARALLEL" -v s="$elapsed_s" 'BEGIN{printf "%.3f", (t*p)/s}')
    echo "$tps"
}

# Mean of N_RUNS bench iterations under one condition.
mean_tps() {
    local label="$1" trace_path="${2:-}" total=0 i
    start_server "$trace_path"
    # warmup
    bench_one "warmup-$label" >/dev/null
    for i in $(seq 1 "$N_RUNS"); do
        local r
        r=$(bench_one "$label-$i")
        total=$(awk -v a="$total" -v b="$r" 'BEGIN{printf "%.6f", a+b}')
        echo "  $label run $i: $r t/s aggregate" >&2
    done
    stop_server
    awk -v t="$total" -v n="$N_RUNS" 'BEGIN{printf "%.3f", t/n}'
}

# ------------------------------------------------------------------ runs
echo "==> Track B.7 instrumentation-cost bench"
echo "  np=$N_PARALLEL ctx=$CTX_SIZE draft=$N_DRAFT tokens=$N_TOKENS_BENCH runs=$N_RUNS"

mean_baseline=$(mean_tps baseline)
mean_traced=$(mean_tps traced "$TMPDIR/trace.ndjson")
delta_pct=$(awk -v b="$mean_baseline" -v t="$mean_traced" 'BEGIN{printf "%.3f", 100.0*(b-t)/b}')

echo
echo "  baseline:     $mean_baseline t/s aggregate (mean of $N_RUNS)"
echo "  instrumented: $mean_traced t/s aggregate (mean of $N_RUNS)"
echo "  delta:        $delta_pct % (regression — positive means slower)"
echo "  gate:         <= $GATE_PCT %"

# ------------------------------------------------------------------ gate
exceeded=$(awk -v d="$delta_pct" -v g="$GATE_PCT" 'BEGIN{print (d>g)?1:0}')
if [[ "$exceeded" == "1" ]]; then
    echo "FAIL: delta $delta_pct% exceeds gate $GATE_PCT%" >&2
    exit 1
fi

echo "PASS"
# Track-C TODO: once fork/join is wired into DeltaNet + FA mma_f16, add
# a third condition (fork/join enabled vs reference single-slot run) and
# gate at <= 4% per PLAN.md B.7.b.
