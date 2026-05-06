#!/usr/bin/env bash
# test-accept-bonus-becomes-seed.sh
#
# Drives:
#   mtp_verify_accept.allium BonusBecomesNextSeed (system-wide)
#
# Property: the bonus_token of round t IS the seed_token of round
# t+1, at position bonus_pos. This is the contract that makes the
# speculative loop progress.
#
# Server-level integration test. Captures two consecutive verify
# rounds and asserts the bonus of round t feeds round t+1's seed.
#
# Prerequisites:
#   - llama-server built with the verify-accept implementation
#   - debug endpoints:
#       GET /debug/recent_accept_decisions?seq_id=0&n=2
#         Return the last n AcceptDecisions for this seq, in
#         chronological order (oldest first), each including
#         {n_accepted, bonus_token, bonus_pos, pos_seed,
#          seed_token, n_drafts}.
#
# RED today: endpoint missing → exit 2.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../mtp-rollout-investigation/_common.sh"

PORT=${PORT:-9091}

assert_bin_model
launch_server "$PORT" -mtp || { echo "FAIL: server launch (rc=$?)" >&2; exit 2; }

# Run a longer completion that produces multiple verify rounds.
curl -s -m 30 "http://127.0.0.1:$PORT/completion" \
    -d '{"prompt":"Continue the sentence: the quick brown fox","n_predict":32,"temperature":0,"_force_spec":true}' > /dev/null

DECISIONS=$(curl -sf -m 5 "http://127.0.0.1:$PORT/debug/recent_accept_decisions?seq_id=0&n=2" || echo "")
if [ -z "$DECISIONS" ]; then
    echo "FAIL: /debug/recent_accept_decisions endpoint missing" >&2
    echo "      Implementation must expose this introspection point." >&2
    exit 2
fi

PREV_BONUS_TOKEN=$(echo "$DECISIONS" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[0]['bonus_token'])")
PREV_BONUS_POS=$(echo "$DECISIONS"   | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[0]['bonus_pos'])")
NEXT_SEED_TOKEN=$(echo "$DECISIONS"  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[1]['seed_token'])")
NEXT_POS_SEED=$(echo "$DECISIONS"    | python3 -c "import sys,json;d=json.load(sys.stdin);print(d[1]['pos_seed'])")

if [ "$PREV_BONUS_TOKEN" -ne "$NEXT_SEED_TOKEN" ] || [ "$PREV_BONUS_POS" -ne "$NEXT_POS_SEED" ]; then
    echo "FAIL: bonus(round t) != seed(round t+1)" >&2
    echo "      prev: token=$PREV_BONUS_TOKEN pos=$PREV_BONUS_POS" >&2
    echo "      next: token=$NEXT_SEED_TOKEN pos=$NEXT_POS_SEED" >&2
    echo "      BonusBecomesNextSeed invariant violated." >&2
    exit 1
fi

echo "PASS: bonus(round t) → seed(round t+1) at token=$PREV_BONUS_TOKEN pos=$PREV_BONUS_POS"
exit 0
