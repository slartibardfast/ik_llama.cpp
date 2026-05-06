#!/usr/bin/env bash
# test-hook-sole-populator.sh
#
# Drives:
#   mtp_ubatch_hook.allium HookIsSoleMtpKvPopulator (system-wide invariant)
#
# Soak test that traces every MTP-layer KV write and asserts each
# write's source attribution is in {hook, fused_draft}. No legacy
# code path (MTP_OP_WARMUP, MTP_OP_UPDATE_ACCEPTED post-accept
# decode, or any other) writes MTP KV.
#
# This test requires an instrumented build. The test-harness extension
# needed is:
#   - Per-write source tagging in the MTP-layer KV write path
#   - A counter exposed via /debug/mtp_kv_writes_by_source returning a
#     JSON object: {"hook": N1, "fused_draft": N2, "other": N3}
#   - "other" must be 0 in the ideal state
#
# This is BEYOND what test-backend-ops can express today and is
# called out as a separate test-harness extension task.
#
# RED behaviour today: the debug endpoint does not exist; this test
# fails with exit 2 (infra not ready) until the extension lands.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../mtp-rollout-investigation/_common.sh"

PORT=${PORT:-9089}

assert_bin_model
launch_server "$PORT" -mtp || { echo "FAIL: server launch (rc=$?)" >&2; exit 2; }

# Drive a moderate workload: a few completion rounds covering both
# prefill (hook fires per ubatch) and gen (fused draft + hook on
# verify forwards).
for i in 1 2 3; do
    curl -s -m 30 "http://127.0.0.1:$PORT/completion" \
        -d "{\"prompt\":\"Round $i — quick test prompt.\",\"n_predict\":16,\"temperature\":0}" \
        > /dev/null
done

# Read source attribution.
RESPONSE=$(curl -s "http://127.0.0.1:$PORT/debug/mtp_kv_writes_by_source" || echo "")
if [ -z "$RESPONSE" ]; then
    echo "FAIL: /debug/mtp_kv_writes_by_source endpoint missing" >&2
    echo "      Implementation must expose per-write source attribution." >&2
    echo "      This is the test-harness extension flagged in the spec README." >&2
    exit 2
fi

# Parse: expect JSON like {"hook": N, "fused_draft": M, "other": K}
HOOK=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('hook', 0))")
FUSED=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('fused_draft', 0))")
OTHER=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('other', -1))")

if [ "$OTHER" -ne 0 ]; then
    echo "FAIL: $OTHER MTP-KV writes from non-{hook,fused_draft} source" >&2
    echo "      HookIsSoleMtpKvPopulator invariant violated." >&2
    echo "      Source breakdown: hook=$HOOK fused_draft=$FUSED other=$OTHER" >&2
    exit 1
fi

echo "PASS: all MTP-KV writes attributed to {hook,fused_draft}: hook=$HOOK fused_draft=$FUSED"
exit 0
