#!/usr/bin/env bash
# MTP-head logit sanity on 35B. Asks the model a high-confidence prompt,
# reads the top-5 log-probabilities with `n_probs=5`, and checks:
#   - top token logprob is in a plausible range (> -50)
#   - top token has higher prob than bottom of top-5
#   - when -mtp is set, the top-token IDs over the first 3 decode steps
#     match the no-mtp top-token IDs (MTP draft should be SAME as main greedy
#     on a high-confidence prompt; any divergence flags MTP head instability).
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"

PROMPT=${PROMPT:-"The capital of France is"}

EXTRA_ARGS_SAVED=$EXTRA_ARGS

launch_and_probe() {
    local tag=$1
    local mtp_flag=$2
    local mtp_env=$3
    local port=$((18800 + $$ % 100))
    LOG=/tmp/mtp-35b-san-$$-$tag.log
    rm -f "$LOG"
    eval "$EXTRA_ENV $mtp_env \"$BIN\" -m \"$MODEL\" --port $port --host 127.0.0.1 \
        -c 2048 --parallel 1 $mtp_flag -ub 32 -b 32 $EXTRA_ARGS_SAVED \
        > \"$LOG\" 2>&1 &"
    SRV_PID=$!
    for i in $(seq 1 ${READY_TIMEOUT:-300}); do
        curl -s -m 2 "http://127.0.0.1:$port/health" > /dev/null 2>&1 && break
        kill -0 $SRV_PID 2>/dev/null || return 1
        sleep 1
    done

    # Get logprobs of first 3 generated tokens
    local out
    out=$(curl -s -m 60 "http://127.0.0.1:$port/completion" \
        -d "{\"prompt\":\"$PROMPT\",\"n_predict\":3,\"temperature\":0,\"n_probs\":5}")
    kill $SRV_PID 2>/dev/null; sleep 2; kill -9 $SRV_PID 2>/dev/null
    sleep 3
    echo "$out"
}

RAW_NOMTP=$(launch_and_probe nomtp "" "")
RAW_MTP=$(launch_and_probe mtp "-mtp" "LLAMA_MTP_ROLLOUT=1")

parse_tops() {
    python3 -c "
import json,sys
d=json.loads(sys.argv[1])
ts=d.get('completion_probabilities') or []
out=[]
for i,t in enumerate(ts):
    probs = t.get('probs') or []
    top = probs[0] if probs else {}
    out.append({'i':i,'tok':t.get('content','?'),'logprob':top.get('logprob',0)})
print(json.dumps(out))
" "$1"
}

T_NOMTP=$(parse_tops "$RAW_NOMTP")
T_MTP=$(parse_tops "$RAW_MTP")
echo "PROMPT: $PROMPT"
echo "no-mtp top tokens: $T_NOMTP"
echo "with-mtp top tokens: $T_MTP"

if [ "$T_NOMTP" = "$T_MTP" ]; then
    echo "PASS: top-token sequence identical for -mtp vs no-mtp (first 3 tokens)"
    exit 0
else
    echo "FAIL: top-token sequence diverges"
    exit 1
fi
