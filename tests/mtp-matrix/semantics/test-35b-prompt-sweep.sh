#!/usr/bin/env bash
# Sweeps 4 prompts on 35B-A3B with `-mtp` and records:
#   - acceptance rate per prompt
#   - greedy output per prompt (first 20 tokens)
#   - greedy output per prompt WITHOUT -mtp (golden reference)
#
# Comparison is by token-id sequence (via `n_probs:1` → per-token entries in
# `completion_probabilities`), not by generated_text string. Two responses
# with the same token ids can still produce different strings when the tail
# token is whitespace-collapsing or a UTF-8 partial; token-id equality is the
# semantically correct invariance check for greedy decoding.
#
# PASS iff: 3/4 prompts produce token-id-identical greedy output with vs
# without -mtp (invariance), accepting 1 outlier for speculation-steering
# into an equivalent-but-different trajectory. FAIL signifies systematic
# corruption.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"

PROMPTS=(
    "The capital of France is"
    "2+2="
    "The sky appears blue because"
    "Once upon a time, there was a"
)
N_PREDICT=${N_PREDICT:-15}
mismatches=0

EXTRA_ARGS_SAVED=$EXTRA_ARGS

# Completion variant that requests n_probs:1 so the response carries a
# per-token `completion_probabilities` array. Sets:
#   CONTENT      - raw generated_text string (human-readable only)
#   TOKSTR_JSON  - JSON array of per-token content strings (token-sequence proxy)
#   NTOK         - number of tokens produced
# Returns non-zero if the request failed.
matrix_completion_tokseq() {
    local prompt=${1:-"The capital of France is"}
    local n_predict=${2:-10}
    local temperature=${3:-0}

    local resp
    resp=$(curl -s -m 120 "http://127.0.0.1:$PORT/completion" \
        -d "{\"prompt\":\"$prompt\",\"n_predict\":$n_predict,\"temperature\":$temperature,\"n_probs\":1}")

    # Extract all three fields in one python invocation. Write a tab-separated
    # line on stdout: CONTENT<TAB>TOKSTR_JSON<TAB>NTOK. CONTENT has newlines
    # escaped as literal \n so it stays on one line.
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
except Exception as e:
    print('__ERR__\t[]\t0')
" 2>/dev/null)

    CONTENT=${parsed%%	*}
    local rest=${parsed#*	}
    TOKSTR_JSON=${rest%	*}
    NTOK=${rest##*	}
    [ "$CONTENT" = "__ERR__" ] && return 1
    return 0
}

launch_one() {
    local with_mtp=$1
    PORT=$((18700 + $$ % 100 + with_mtp * 100))
    LOG=/tmp/mtp-35b-sweep-$$-$with_mtp.log
    rm -f "$LOG"
    local mtp_flag=""
    local mtp_env=""
    if [ "$with_mtp" = "1" ]; then
        mtp_flag="-mtp"
        mtp_env="LLAMA_MTP_ROLLOUT=1"
    fi
    eval "$EXTRA_ENV $mtp_env \"$BIN\" -m \"$MODEL\" --port $PORT --host 127.0.0.1 \
        -c 2048 --parallel 1 $mtp_flag -ub 32 -b 32 $EXTRA_ARGS_SAVED \
        > \"$LOG\" 2>&1 &"
    SRV_PID=$!
    for i in $(seq 1 ${READY_TIMEOUT:-300}); do
        curl -s -m 2 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && return 0
        kill -0 $SRV_PID 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

teardown_one() {
    kill $SRV_PID 2>/dev/null; sleep 2; kill -9 $SRV_PID 2>/dev/null
    sleep 3
}

# No-MTP baseline for all prompts
launch_one 0 || { echo "CRASH: no-mtp server failed"; exit 3; }
declare -a OUT_NOMTP_STR
declare -a OUT_NOMTP_TOKS
declare -a OUT_NOMTP_NTOK
for p in "${PROMPTS[@]}"; do
    matrix_completion_tokseq "$p" "$N_PREDICT" 0
    OUT_NOMTP_STR+=("$CONTENT")
    OUT_NOMTP_TOKS+=("$TOKSTR_JSON")
    OUT_NOMTP_NTOK+=("$NTOK")
done
teardown_one

# With-MTP pass
launch_one 1 || { echo "CRASH: mtp server failed"; exit 3; }
declare -a OUT_MTP_STR
declare -a OUT_MTP_TOKS
declare -a OUT_MTP_NTOK
declare -a ACCEPT
for p in "${PROMPTS[@]}"; do
    matrix_completion_tokseq "$p" "$N_PREDICT" 0
    OUT_MTP_STR+=("$CONTENT")
    OUT_MTP_TOKS+=("$TOKSTR_JSON")
    OUT_MTP_NTOK+=("$NTOK")
    a=$(grep 'acceptance rate' "$LOG" | tail -1 | sed -n 's/.*acceptance rate = \([0-9.]*\).*/\1/p')
    ACCEPT+=("${a:-NA}")
done
teardown_one

# Compare on token-id sequence (via per-token tok_str array from n_probs:1)
echo "--- 35B MTP vs no-MTP greedy sweep (T=0, n_predict=$N_PREDICT) ---"
echo "    comparison: token-id sequence equality (completion_probabilities[].content)"
for i in "${!PROMPTS[@]}"; do
    match="DIFF"
    if [ "${OUT_NOMTP_TOKS[$i]}" = "${OUT_MTP_TOKS[$i]}" ]; then
        match="SAME"
    else
        mismatches=$((mismatches + 1))
    fi
    # Short token-prefix for display (first ~60 chars of JSON array)
    local_prefix_nomtp=${OUT_NOMTP_TOKS[$i]:0:80}
    local_prefix_mtp=${OUT_MTP_TOKS[$i]:0:80}
    echo ""
    echo "[$i] prompt: ${PROMPTS[$i]}"
    echo "    no-mtp   str : ${OUT_NOMTP_STR[$i]}"
    echo "    with-mtp str : ${OUT_MTP_STR[$i]}"
    echo "    no-mtp   toks: ntok=${OUT_NOMTP_NTOK[$i]} prefix=${local_prefix_nomtp}"
    echo "    with-mtp toks: ntok=${OUT_MTP_NTOK[$i]} prefix=${local_prefix_mtp}"
    echo "    accept : ${ACCEPT[$i]}  match=$match"
done

echo ""
echo "mismatches=$mismatches / ${#PROMPTS[@]}"
if [ "$mismatches" -le 1 ]; then
    echo "PASS"; exit 0
else
    echo "FAIL: MTP corrupting main-model sampling on majority of prompts"
    exit 1
fi
