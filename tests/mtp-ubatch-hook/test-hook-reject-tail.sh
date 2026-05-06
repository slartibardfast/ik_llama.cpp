#!/usr/bin/env bash
# test-hook-reject-tail.sh
#
# Drives:
#   mtp_ubatch_hook.allium MtpKvHasAcceptedPrefix
#   mtp_ubatch_hook.allium MtpKvLacksRejectedSuffix
#   mtp_ubatch_hook.allium ChopRangeIsRejectedSuffix (KvRangeChop contract)
#
# Server-level integration test. Simulates a verify cycle with
# n_accepted < n_drafts and asserts post-chop MTP-KV state matches
# the accepted prefix exactly.
#
# Strategy:
#   1. Pre-fill prompt to position p.
#   2. Issue verify batch [last_accepted, D1, D2, D3] (positions
#      p, p+1, p+2, p+3).
#   3. Inject n_accepted via a debug control point (must be added to
#      the implementation).
#   4. Query MTP-KV state at positions p..p+3.
#   5. Assert positions [p..p+n_accepted-1] are present and
#      [p+n_accepted..p+3] are absent.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../mtp-rollout-investigation/_common.sh"

PORT=${PORT:-9088}

assert_bin_model
launch_server "$PORT" -mtp || { echo "FAIL: server launch (rc=$?)" >&2; exit 2; }

# 1. Pre-fill known prompt to advance to a known position.
curl -s -m 30 "http://127.0.0.1:$PORT/completion" \
    -d '{"prompt":"hello world","n_predict":1,"temperature":0}' > /dev/null

# 2. Set forced n_accepted = 2 via debug control. Implementation must
# expose:
#   POST /debug/force_n_accepted -d '{"n":2}'
# This forces the next verify cycle to accept exactly 2 drafts
# regardless of model output. Without this, the test cannot
# deterministically position itself in the partial-accept regime.

if ! curl -sf -m 5 -X POST "http://127.0.0.1:$PORT/debug/force_n_accepted" \
        -d '{"n":2}' > /dev/null; then
    echo "FAIL: /debug/force_n_accepted endpoint missing or rejected" >&2
    echo "      Implementation must expose this control point." >&2
    exit 2
fi

# 3. Trigger one round of speculative decode that runs verify with the
# forced accept count.
curl -s -m 30 "http://127.0.0.1:$PORT/completion" \
    -d '{"prompt":"hello world","n_predict":1,"temperature":0,"_force_spec":true}' > /dev/null

# 4. Inspect MTP-KV. Implementation exposes:
#   GET /debug/mtp_kv_has_pos?seq_id=0&pos=N -> "true" or "false"

POS_BASE=$(curl -s "http://127.0.0.1:$PORT/debug/main_kv_pos_max?seq_id=0")
P0=$((POS_BASE - 3))
P1=$((POS_BASE - 2))
P2=$((POS_BASE - 1))
P3=$((POS_BASE))

# Expected: P0 (last accepted) and P1 (first draft, accepted) present;
#           P2 (second draft, rejected per force=2) and P3 absent.
# The boundary depends on whether n_accepted counts the last-accepted
# token. Per contract, KvRangeChop removes [pos_first + n_accepted,
# pos_last]. With pos_first=P0 and n_accepted=2 (counting drafts), the
# kept range is [P0, P0+2-1] = [P0, P1]; chopped is [P2, P3].

H0=$(curl -s "http://127.0.0.1:$PORT/debug/mtp_kv_has_pos?seq_id=0&pos=$P0")
H1=$(curl -s "http://127.0.0.1:$PORT/debug/mtp_kv_has_pos?seq_id=0&pos=$P1")
H2=$(curl -s "http://127.0.0.1:$PORT/debug/mtp_kv_has_pos?seq_id=0&pos=$P2")
H3=$(curl -s "http://127.0.0.1:$PORT/debug/mtp_kv_has_pos?seq_id=0&pos=$P3")

fail=0
if [ "$H0" != "true" ]; then echo "FAIL: pos $P0 expected present, got $H0" >&2; fail=1; fi
if [ "$H1" != "true" ]; then echo "FAIL: pos $P1 expected present, got $H1" >&2; fail=1; fi
if [ "$H2" != "false" ]; then echo "FAIL: pos $P2 expected absent, got $H2" >&2; fail=1; fi
if [ "$H3" != "false" ]; then echo "FAIL: pos $P3 expected absent, got $H3" >&2; fail=1; fi

if [ "$fail" -ne 0 ]; then
    echo "FAIL: chop semantics violated; MtpKvHasAcceptedPrefix or" >&2
    echo "      MtpKvLacksRejectedSuffix invariant tripped." >&2
    exit 1
fi

echo "PASS: accepted prefix [$P0..$P1] present, rejected suffix [$P2..$P3] absent"
exit 0
