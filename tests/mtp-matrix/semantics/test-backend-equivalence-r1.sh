#!/usr/bin/env bash
# Cross-backend semantics: rollout=1, same prompt, same seed, on TWO different
# backends must produce the same top-token and close probs (Q8_0 quantization
# allows some drift; set epsilon to 5e-2).
#
# Requires TWO binaries: BIN_A (the one under test) and BIN_B (the oracle).
# If BIN_B is not set, test SKIPs.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
source "$SCRIPT_DIR/../lib/_logits.sh"

if [ -z "${BIN_B:-}" ]; then
    echo "SKIP: BIN_B not set (oracle backend binary)"
    exit 2
fi
if [ ! -x "$BIN_B" ]; then
    echo "SKIP: BIN_B not executable: $BIN_B"
    exit 2
fi

PROMPT="The capital of France is"
ROLLOUT=1

# Run A (BIN under test)
matrix_setup
if ! matrix_launch; then echo "CRASH: A"; exit 3; fi
FILE_A=/tmp/eqv-a-$$.json
logits_capture_to "$FILE_A" "$PROMPT" 3
matrix_teardown

# Run B (oracle). Save current state, swap BIN, rerun.
ORIG_BIN=$BIN
BIN=$BIN_B
PORT=$((PORT + 1))
matrix_setup
if ! matrix_launch; then echo "CRASH: B"; rm -f "$FILE_A"; exit 3; fi
FILE_B=/tmp/eqv-b-$$.json
logits_capture_to "$FILE_B" "$PROMPT" 3
matrix_teardown
BIN=$ORIG_BIN

# Compare. Tolerance 5e-2 for across-backend Q8_0 drift.
if ! logits_diff "$FILE_A" "$FILE_B" 5e-2; then
    echo "FAIL: backends disagree"
    rm -f "$FILE_A" "$FILE_B"
    exit 1
fi
rm -f "$FILE_A" "$FILE_B"
echo "PASS: backends agree within tolerance"
exit 0
