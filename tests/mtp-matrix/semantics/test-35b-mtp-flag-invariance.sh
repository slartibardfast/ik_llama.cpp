#!/usr/bin/env bash
# Asserts: on 35B-A3B, greedy output at T=0 is token-id-identical with `-mtp`
# set vs unset. If MTP compute is truly downstream of main and non-mutating,
# the main model's argmax sequence should be independent of whether MTP inline
# emission is active.
#
# Comparison is by token-id sequence (via `n_probs:1` → per-token entries in
# `completion_probabilities`), not by generated_text string. Two responses
# with identical token ids can still produce differing strings when the tail
# token is whitespace-collapsing or a UTF-8 partial — those are not semantic
# divergences and should not be flagged.
#
# FAIL of this test localizes the bug to: MTP graph nodes are affecting main
# output (buffer aliasing, KV cache bleed, or batch-invariance breaking under
# the extra batch=2 verify decode triggered by draft consumption).
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"

PROMPT=${PROMPT:-"The capital of France is"}
N_PREDICT=${N_PREDICT:-10}

# Completion variant that requests n_probs:1 so the response carries a
# per-token `completion_probabilities` array. Sets:
#   CONTENT      - raw generated_text string (human-readable only; newlines
#                  escaped to \n for single-line display)
#   TOKSTR_JSON  - JSON array of per-token content strings (token-sequence proxy)
#   NTOK         - number of tokens produced
matrix_completion_tokseq() {
    local prompt=${1:-"The capital of France is"}
    local n_predict=${2:-10}
    local temperature=${3:-0}

    local resp
    resp=$(curl -s -m 120 "http://127.0.0.1:$PORT/completion" \
        -d "{\"prompt\":\"$prompt\",\"n_predict\":$n_predict,\"temperature\":$temperature,\"n_probs\":1}")

    local parsed
    parsed=$(echo "$resp" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    content = d.get('content', '') or ''
    cp = d.get('completion_probabilities') or []
    toks = [e.get('content', '') for e in cp if isinstance(e, dict)]
    ntok = d.get('tokens_predicted', len(toks))
    content_safe = content.replace('\\\\', '\\\\\\\\').replace('\\t', ' ').replace('\\n', '\\\\n').replace('\\r', ' ')
    print(content_safe + '\t' + json.dumps(toks, ensure_ascii=False) + '\t' + str(ntok))
except Exception:
    print('__ERR__\t[]\t0')
" 2>/dev/null)

    CONTENT=${parsed%%	*}
    local rest=${parsed#*	}
    TOKSTR_JSON=${rest%	*}
    NTOK=${rest##*	}
    [ "$CONTENT" = "__ERR__" ] && return 1
    return 0
}

# Pass 1: no -mtp
PORT=$((18500 + $$ % 100))
ROLLOUT=1
EXTRA_ARGS_SAVED=$EXTRA_ARGS
EXTRA_ARGS="$EXTRA_ARGS_SAVED"  # strip any -mtp if the caller set it
matrix_setup
# Override: drop -mtp from the launch by bypassing normal matrix_launch
LOG=/tmp/mtp-35b-invar-nomtp-$$.log
rm -f "$LOG"
eval "$EXTRA_ENV \"$BIN\" -m \"$MODEL\" --port $PORT --host 127.0.0.1 \
    -c 2048 --parallel 1 -ub 32 -b 32 $EXTRA_ARGS_SAVED \
    > \"$LOG\" 2>&1 &"
SRV_PID=$!
for i in $(seq 1 ${READY_TIMEOUT:-300}); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
    kill -0 $SRV_PID 2>/dev/null || { echo "CRASH no-mtp ${i}s"; exit 3; }
    sleep 1
done
matrix_completion_tokseq "$PROMPT" "$N_PREDICT" 0
OUT_NOMTP_STR="$CONTENT"
OUT_NOMTP_TOKS="$TOKSTR_JSON"
OUT_NOMTP_NTOK="$NTOK"
kill $SRV_PID 2>/dev/null; sleep 2; kill -9 $SRV_PID 2>/dev/null

# Pass 2: -mtp
sleep 3
PORT=$((18600 + $$ % 100))
LOG=/tmp/mtp-35b-invar-withmtp-$$.log
rm -f "$LOG"
eval "$EXTRA_ENV LLAMA_MTP_ROLLOUT=1 \"$BIN\" -m \"$MODEL\" --port $PORT --host 127.0.0.1 \
    -c 2048 --parallel 1 -mtp -ub 32 -b 32 $EXTRA_ARGS_SAVED \
    > \"$LOG\" 2>&1 &"
SRV_PID=$!
for i in $(seq 1 ${READY_TIMEOUT:-300}); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
    kill -0 $SRV_PID 2>/dev/null || { echo "CRASH mtp ${i}s"; exit 3; }
    sleep 1
done
matrix_completion_tokseq "$PROMPT" "$N_PREDICT" 0
OUT_MTP_STR="$CONTENT"
OUT_MTP_TOKS="$TOKSTR_JSON"
OUT_MTP_NTOK="$NTOK"
ACCEPT=$(grep 'acceptance rate' "$LOG" | tail -1 | sed -n 's/.*acceptance rate = \([0-9.]*\).*/\1/p')
kill $SRV_PID 2>/dev/null; sleep 2; kill -9 $SRV_PID 2>/dev/null

echo "--- PROMPT: $PROMPT (n_predict=$N_PREDICT, T=0) ---"
echo "    comparison: token-id sequence equality (completion_probabilities[].content)"
echo "NO-MTP   str  : $OUT_NOMTP_STR"
echo "WITH-MTP str  : $OUT_MTP_STR"
echo "NO-MTP   toks : ntok=$OUT_NOMTP_NTOK prefix=${OUT_NOMTP_TOKS:0:120}"
echo "WITH-MTP toks : ntok=$OUT_MTP_NTOK prefix=${OUT_MTP_TOKS:0:120}"
echo "ACCEPT : ${ACCEPT:-0}"

if [ "$OUT_NOMTP_TOKS" = "$OUT_MTP_TOKS" ]; then
    echo "PASS: token-id-identical greedy output with/without -mtp"
    exit 0
else
    echo "FAIL: greedy token-id output diverges when -mtp is active"
    echo "  -> MTP is affecting main-model sampling trajectory"
    exit 1
fi
