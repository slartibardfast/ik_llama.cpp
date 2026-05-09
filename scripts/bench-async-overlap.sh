#!/usr/bin/env bash
#
# PHASE46 E.2 — conditional async-overlap bench (Track E).
#
# Detects the CUDA compute capability and skips the async-overlap
# measurement on sm_75 (Phase 38 E ground truth: async overlap measures
# negative on Quadro RTX 6000 because decode is memory-bandwidth-bound).
#
# On sm_80+ where SM slack exists, runs the production workload twice
# (LLAMA_DRAFT_OVERLAP=0 vs =1) and reports the overlap delta. Default
# baseline (overlap=0) must always meet PLAN B.7 instrumentation gate.
#
# Pre-reqs: a Qwen3.6-style GGUF + the post-PHASE46 server binary.

set -euo pipefail

SERVER_BIN="${SERVER_BIN:-build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-}"
N_PARALLEL="${N_PARALLEL:-3}"
CTX_SIZE="${CTX_SIZE:-262144}"
N_DRAFT="${N_DRAFT:-3}"
N_TOKENS_BENCH="${N_TOKENS_BENCH:-1024}"
N_RUNS="${N_RUNS:-5}"
PORT="${PORT:-18080}"

usage() {
    cat <<USAGE
usage: MODEL_PATH=<path-to-gguf> $0

env knobs:
  SERVER_BIN, MODEL_PATH (required), N_PARALLEL, CTX_SIZE, N_DRAFT,
  N_TOKENS_BENCH, N_RUNS, PORT

behavior:
  * detects the lowest CUDA compute capability across visible devices
  * sm_75 → emit "sm_75: skipped, Phase 38 E baseline" and exit 0
  * sm_80+ → run two conditions (overlap=0, overlap=1), print delta
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

# ---- compute capability detection -----------------------------------
detect_min_cc() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "error: nvidia-smi not found; cannot detect cc" >&2
        return 1
    fi
    local raw min
    raw=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' .')
    if [[ -z "$raw" ]]; then
        echo "error: nvidia-smi returned no compute_cap rows" >&2
        return 1
    fi
    min=$(printf '%s\n' "$raw" | sort -n | head -n1)
    if [[ -z "$min" ]]; then
        return 1
    fi
    echo "$min"
}

cc=$(detect_min_cc)
echo "==> Track E.2 async-overlap bench"
echo "  detected min compute capability: $cc"

if [[ "$cc" -lt 80 ]] 2>/dev/null; then
    echo "sm_$cc: skipped, Phase 38 E baseline (memory-bw-bound on Turing-class)"
    exit 0
fi

echo "  proceeding with async-overlap bench on sm_$cc"

# ---- bench helpers (mirror bench-instrumentation-cost.sh) -----------
TMPDIR=$(mktemp -d -t llama-overlap-XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT

server_pid=
start_server() {
    local overlap="$1"
    LLAMA_DRAFT_OVERLAP="$overlap" "$SERVER_BIN" \
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

bench_one() {
    local out="$TMPDIR/run.out"
    : > "$out"
    local start_ns end_ns elapsed_s
    start_ns=$(date +%s%N)
    local slot
    for slot in $(seq 1 "$N_PARALLEL"); do
        curl -fsS \
            -H 'content-type: application/json' \
            -d '{"prompt":"Determinism implies","n_predict":'"$N_TOKENS_BENCH"',"temperature":0,"seed":42}' \
            "http://127.0.0.1:$PORT/completion" >> "$out" 2>/dev/null &
    done
    wait
    end_ns=$(date +%s%N)
    elapsed_s=$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN{printf "%.6f", (e-s)/1e9}')
    awk -v t="$N_TOKENS_BENCH" -v p="$N_PARALLEL" -v s="$elapsed_s" 'BEGIN{printf "%.3f", (t*p)/s}'
}

mean_tps() {
    local label="$1" overlap="$2" total=0 i r
    start_server "$overlap"
    bench_one >/dev/null
    for i in $(seq 1 "$N_RUNS"); do
        r=$(bench_one)
        total=$(awk -v a="$total" -v b="$r" 'BEGIN{printf "%.6f", a+b}')
        echo "  $label run $i: $r t/s aggregate" >&2
    done
    stop_server
    awk -v t="$total" -v n="$N_RUNS" 'BEGIN{printf "%.3f", t/n}'
}

# ---- runs ------------------------------------------------------------
mean_off=$(mean_tps "overlap=0" 0)
mean_on=$(mean_tps "overlap=1" 1)
delta_pct=$(awk -v a="$mean_off" -v b="$mean_on" 'BEGIN{printf "%.3f", 100*(b-a)/a}')

echo
echo "  overlap=0 (baseline):  $mean_off t/s aggregate"
echo "  overlap=1 (async):     $mean_on t/s aggregate"
echo "  lift:                  $delta_pct % (positive = faster with overlap)"
