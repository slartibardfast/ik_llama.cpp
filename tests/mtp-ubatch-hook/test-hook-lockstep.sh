#!/usr/bin/env bash
# test-hook-lockstep.sh
#
# Drives:
#   mtp_ubatch_hook.allium MtpKvLockstepWithMainKv (system-wide invariant)
#
# After every ubatch processed by the main model, the MTP-layer KV
# is populated for exactly the same position range as the main-layer
# KV for that ubatch's sequence. No drift between the two KVs in the
# forward direction.
#
# Server-level integration test, follows the
# tests/mtp-rollout-investigation/_common.sh pattern.
#
# Prerequisites:
#   - llama-server built with the hook-enabled implementation
#   - MODEL=... (default: 0.8B Q8_0 MTP fixture)
#   - llama-server exposes a /debug/mtp_kv_pos_max endpoint or similar
#     (must be added by implementation alongside the hook)
#
# Outcome:
#   exit 0 — main and MTP KV agree at every position (GREEN)
#   exit 1 — drift detected (RED, hook missing or buggy)
#   exit 2 — infrastructure not ready (binary, model, or debug endpoint)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../mtp-rollout-investigation/_common.sh"

PORT=${PORT:-9087}

assert_bin_model

# Launch with MTP enabled
launch_server "$PORT" -mtp || {
    echo "FAIL: server failed to launch (rc=$?)" >&2
    exit 2
}

# Submit a known prompt that fills multiple ubatches.
# 64-token prompt with ub=32 produces 2 ubatches.
PROMPT="The quick brown fox jumps over the lazy dog. $(yes "tokens here" | head -n 50 | tr -d '\n')"

curl -s -m 30 "http://127.0.0.1:$PORT/completion" \
    -d "{\"prompt\":\"$PROMPT\",\"n_predict\":1,\"temperature\":0}" > /dev/null

# Probe the KV state. Implementation must expose:
#   GET /debug/main_kv_pos_max?seq_id=0 -> integer
#   GET /debug/mtp_kv_pos_max?seq_id=0  -> integer
# Both must agree post-prefill.

MAIN_MAX=$(curl -s "http://127.0.0.1:$PORT/debug/main_kv_pos_max?seq_id=0" || echo "")
MTP_MAX=$(curl -s "http://127.0.0.1:$PORT/debug/mtp_kv_pos_max?seq_id=0" || echo "")

if [ -z "$MAIN_MAX" ] || [ -z "$MTP_MAX" ]; then
    echo "FAIL: debug endpoints did not return values (got main='$MAIN_MAX' mtp='$MTP_MAX')" >&2
    echo "      Implementation must expose /debug/main_kv_pos_max and /debug/mtp_kv_pos_max." >&2
    exit 2
fi

if [ "$MAIN_MAX" -ne "$MTP_MAX" ]; then
    echo "FAIL: KV drift — main pos_max=$MAIN_MAX, MTP pos_max=$MTP_MAX" >&2
    echo "      MtpKvLockstepWithMainKv invariant violated." >&2
    exit 1
fi

echo "PASS: main_kv_pos_max=$MAIN_MAX matches mtp_kv_pos_max=$MTP_MAX (GREEN)"
exit 0
