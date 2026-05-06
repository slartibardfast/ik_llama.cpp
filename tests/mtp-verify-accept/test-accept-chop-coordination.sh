#!/usr/bin/env bash
# test-accept-chop-coordination.sh
#
# Drives:
#   mtp_verify_accept.allium ChopRangeFollowsAcceptDecision
#   (cross-spec property → mtp_ubatch_hook.allium KvRangeChop)
#
# Property: the MTP-KV chop range is exactly
#   [pos_seed + n_accepted + 1, pos_seed + n_drafts]
# The bonus position (pos_seed + n_accepted + 1) IS the first chopped
# position because the bonus's KV has not yet been written — it will
# be written by the NEXT verify forward, where the bonus serves as
# that round's seed.
#
# Server-level integration test. Wires AcceptVerify -> KvRangeChop
# end-to-end and asserts the chop range matches the accept-decision.
#
# Prerequisites:
#   - llama-server built with the verify-accept implementation
#   - debug endpoints (must be exposed by the implementation):
#       POST /debug/force_n_accepted     {"n":<k>}
#         Force the next verify cycle to accept exactly k drafts
#         regardless of model output.
#       GET  /debug/last_chop_range?seq_id=0
#         Return the [first, last] chop range applied by the most
#         recent KvRangeChop, or null if no chop fired this round.
#       GET  /debug/last_accept_decision?seq_id=0
#         Return {n_accepted, bonus_token, bonus_pos, pos_seed,
#         n_drafts} from the most recent AcceptVerify.
#
# RED behaviour today: debug endpoints don't exist; this test exits
# 2 (infra not ready) until the implementation lands.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../mtp-rollout-investigation/_common.sh"

PORT=${PORT:-9090}

assert_bin_model
launch_server "$PORT" -mtp || { echo "FAIL: server launch (rc=$?)" >&2; exit 2; }

# 1. Pre-fill prompt to advance the seq.
curl -s -m 30 "http://127.0.0.1:$PORT/completion" \
    -d '{"prompt":"hello world","n_predict":1,"temperature":0}' > /dev/null

# 2. Force n_accepted = 1 (mid-range partial-accept regime).
if ! curl -sf -m 5 -X POST "http://127.0.0.1:$PORT/debug/force_n_accepted" \
        -d '{"n":1}' > /dev/null; then
    echo "FAIL: /debug/force_n_accepted endpoint missing" >&2
    echo "      Implementation must expose this control point." >&2
    exit 2
fi

# 3. Trigger a speculative round.
curl -s -m 30 "http://127.0.0.1:$PORT/completion" \
    -d '{"prompt":"hello world","n_predict":1,"temperature":0,"_force_spec":true}' > /dev/null

# 4. Read the decision and the chop range.
DECISION=$(curl -sf -m 5 "http://127.0.0.1:$PORT/debug/last_accept_decision?seq_id=0" || echo "")
CHOP=$(curl -sf -m 5 "http://127.0.0.1:$PORT/debug/last_chop_range?seq_id=0" || echo "")
if [ -z "$DECISION" ] || [ -z "$CHOP" ]; then
    echo "FAIL: debug endpoints did not return values" >&2
    echo "      Need /debug/last_accept_decision and /debug/last_chop_range." >&2
    exit 2
fi

POS_SEED=$(echo "$DECISION"  | python3 -c "import sys,json;print(json.load(sys.stdin)['pos_seed'])")
N_ACC=$(echo "$DECISION"     | python3 -c "import sys,json;print(json.load(sys.stdin)['n_accepted'])")
N_DRAFTS=$(echo "$DECISION"  | python3 -c "import sys,json;print(json.load(sys.stdin)['n_drafts'])")
CHOP_FIRST=$(echo "$CHOP"    | python3 -c "import sys,json;print(json.load(sys.stdin)['first'])")
CHOP_LAST=$(echo "$CHOP"     | python3 -c "import sys,json;print(json.load(sys.stdin)['last'])")

EXPECT_FIRST=$((POS_SEED + N_ACC + 1))
EXPECT_LAST=$((POS_SEED + N_DRAFTS))

if [ "$CHOP_FIRST" -ne "$EXPECT_FIRST" ] || [ "$CHOP_LAST" -ne "$EXPECT_LAST" ]; then
    echo "FAIL: chop range mismatch" >&2
    echo "      n_accepted=$N_ACC pos_seed=$POS_SEED n_drafts=$N_DRAFTS" >&2
    echo "      expected [$EXPECT_FIRST,$EXPECT_LAST] got [$CHOP_FIRST,$CHOP_LAST]" >&2
    echo "      ChopRangeFollowsAcceptDecision invariant violated." >&2
    exit 1
fi

echo "PASS: chop range [$CHOP_FIRST,$CHOP_LAST] matches n_accepted=$N_ACC at pos_seed=$POS_SEED"
exit 0
