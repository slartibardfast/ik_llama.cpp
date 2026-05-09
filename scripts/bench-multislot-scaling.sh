#!/usr/bin/env bash
#
# PHASE46 D.1 + D.2 — multi-slot scaling exploration.
#
# Sweeps over np values at the documented production config and reports
# aggregate throughput, per-slot throughput, scaling factor vs np=1, and
# % of theoretical bandwidth ceiling. Two passes:
#   D.1: np = 1, 2, 3, 4 at ctx=262144 (production prompt budget)
#   D.2: np = 1, 2, 4, 8 at ctx=65536  (deeper-fan-out budget)
#
# Theoretical bandwidth ceiling is approximated as
#   (aggregate_mem_bw_GBps / weight_size_GB)  *  (1 + accept_rate * draft_depth)
# default values match 2× Quadro RTX 6000 + Qwen3.6 27B IQ4 + draft 3 +
# 0.7 accept; override with WEIGHT_GB / MEM_BW_GBPS / ACCEPT_RATE.
#
# Pre-reqs: same as bench-instrumentation-cost.sh.

set -euo pipefail

SERVER_BIN="${SERVER_BIN:-build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-}"
N_DRAFT="${N_DRAFT:-3}"
N_TOKENS_BENCH="${N_TOKENS_BENCH:-1024}"
N_RUNS="${N_RUNS:-3}"
PORT="${PORT:-18080}"
WEIGHT_GB="${WEIGHT_GB:-14.0}"
MEM_BW_GBPS="${MEM_BW_GBPS:-1248.0}"
ACCEPT_RATE="${ACCEPT_RATE:-0.7}"
OUT_TSV="${OUT_TSV:-/tmp/multislot-scaling.tsv}"

usage() {
    cat <<USAGE
usage: MODEL_PATH=<path-to-gguf> $0

env knobs:
  SERVER_BIN, MODEL_PATH (required), N_DRAFT, N_TOKENS_BENCH, N_RUNS, PORT
  WEIGHT_GB ($WEIGHT_GB)         model weight bytes for ceiling math
  MEM_BW_GBPS ($MEM_BW_GBPS)     aggregate device memory bandwidth
  ACCEPT_RATE ($ACCEPT_RATE)     speculative-decode accept rate
  OUT_TSV ($OUT_TSV)             scaling table output path

writes a TSV with columns:
  ctx  np  mean_aggregate_tps  per_slot_tps  scaling_x  pct_ceiling

emits this table to stdout as well, in human-readable form.
USAGE
}

if [[ -z "$MODEL_PATH" ]]; then
    usage >&2
    exit 2
fi

TMPDIR=$(mktemp -d -t llama-scaling-XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT

server_pid=
start_server() {
    local np="$1" ctx="$2"
    "$SERVER_BIN" \
        -m "$MODEL_PATH" \
        --port "$PORT" \
        --host 127.0.0.1 \
        -np "$np" \
        --draft "$N_DRAFT" \
        -c "$ctx" \
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
    echo "error: server failed to come up at np=$np ctx=$ctx" >&2
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
    local np="$1"
    local out="$TMPDIR/run.json"
    : > "$out"
    local start_ns end_ns elapsed_s
    start_ns=$(date +%s%N)
    for slot in $(seq 1 "$np"); do
        curl -fsS \
            -H 'content-type: application/json' \
            -d '{"prompt":"Write a short paragraph about determinism.","n_predict":'"$N_TOKENS_BENCH"',"temperature":0,"seed":42}' \
            "http://127.0.0.1:$PORT/completion" \
            >> "$out" 2>/dev/null &
    done
    wait
    end_ns=$(date +%s%N)
    elapsed_s=$(awk -v s="$start_ns" -v e="$end_ns" 'BEGIN{printf "%.6f", (e-s)/1e9}')
    awk -v t="$N_TOKENS_BENCH" -v p="$np" -v s="$elapsed_s" 'BEGIN{printf "%.3f", (t*p)/s}'
}

# Theoretical ceiling at np=N: pure decode reads ALL weights once per token;
# aggregate mem bw ÷ weight size sets the upper bound. MTP bonus is
# (1 + accept_rate * draft_depth).
compute_ceiling() {
    awk -v bw="$MEM_BW_GBPS" -v w="$WEIGHT_GB" -v a="$ACCEPT_RATE" -v d="$N_DRAFT" \
        'BEGIN{printf "%.3f", (bw/w) * (1 + a*d)}'
}

run_sweep() {
    local ctx="$1" ; shift
    local nps=("$@")
    local mean_np1=
    echo
    echo "===== sweep ctx=$ctx, np in {${nps[*]}} ====="
    printf '%-6s %-12s %-14s %-14s %-12s %-14s\n' "np" "agg_tps" "per_slot_tps" "scaling_x" "ceiling" "%_ceiling"
    for np in "${nps[@]}"; do
        start_server "$np" "$ctx"
        bench_one "$np" >/dev/null  # warmup
        local total=0 r i
        for i in $(seq 1 "$N_RUNS"); do
            r=$(bench_one "$np")
            total=$(awk -v a="$total" -v b="$r" 'BEGIN{printf "%.6f", a+b}')
        done
        stop_server
        local mean per_slot scaling pct ceiling
        mean=$(awk -v t="$total" -v n="$N_RUNS" 'BEGIN{printf "%.3f", t/n}')
        per_slot=$(awk -v m="$mean" -v p="$np" 'BEGIN{printf "%.3f", m/p}')
        ceiling=$(compute_ceiling)
        if [[ "$np" == "1" ]]; then
            mean_np1="$mean"
        fi
        scaling=$(awk -v m="$mean" -v one="${mean_np1:-1}" 'BEGIN{printf "%.2f", m/one}')
        pct=$(awk -v m="$mean" -v c="$ceiling" 'BEGIN{printf "%.1f", 100*m/c}')
        printf '%-6s %-12s %-14s %-14s %-12s %-14s\n' "$np" "$mean" "$per_slot" "${scaling}x" "$ceiling" "${pct}%"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$ctx" "$np" "$mean" "$per_slot" "$scaling" "$pct" >> "$OUT_TSV"
    done
}

# ------------------------------------------------------------------ run
: > "$OUT_TSV"
echo "# ctx  np  mean_agg_tps  per_slot_tps  scaling_x  pct_ceiling" >> "$OUT_TSV"
run_sweep 262144 1 2 3 4
run_sweep 65536  1 2 4 8

echo
echo "scaling table written to $OUT_TSV"
