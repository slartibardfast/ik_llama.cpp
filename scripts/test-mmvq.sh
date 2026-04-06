#!/bin/bash
# Comprehensive MUL_MAT test matrix for MMVQ debugging
# Tests all quant types × sizes × batch counts to identify exactly which
# configurations fail and whether the issue is MMVQ, batch_n, or per-type.
#
# Usage: GGML_VK_VISIBLE_DEVICES=1 ./scripts/test-mmvq.sh [gpu_id]
# Output: tab-separated results for easy analysis
set -euo pipefail

cd "$(dirname "$0")/.."

GPU=${1:-1}
TEST=build/bin/test-backend-ops

if [ ! -x "$TEST" ]; then
    echo "ERROR: $TEST not found" >&2
    exit 1
fi

echo "=== MMVQ Test Matrix ==="
echo "GPU: Vulkan${GPU}"
echo "Date: $(date -Iseconds)"
echo ""

# Run full MUL_MAT suite and capture all results
LOG=/tmp/mmvq-test-$(date +%s).log
echo "Running full MUL_MAT test suite..."
GGML_VK_VISIBLE_DEVICES=$GPU "$TEST" test -o MUL_MAT > "$LOG" 2>&1 || true

echo ""
echo "=== Results by type × size ==="
echo ""
printf "%-8s %-8s %-8s %-8s %-6s %-8s %s\n" "TYPE" "M" "N" "K" "BS" "RESULT" "DETAIL"
echo "-------- -------- -------- -------- ------ -------- --------"

# Parse results
while IFS= read -r line; do
    if echo "$line" | grep -qE "MUL_MAT\("; then
        # Extract fields
        type_a=$(echo "$line" | grep -oP 'type_a=\K[^,]+')
        m=$(echo "$line" | grep -oP 'm=\K[^,]+')
        n=$(echo "$line" | grep -oP 'n=\K[^,]+')
        k=$(echo "$line" | grep -oP 'k=\K[^,]+')
        bs=$(echo "$line" | grep -oP 'bs=\K[^,\]]+\]' | tr -d '[]')

        if echo "$line" | grep -q "FAIL"; then
            detail=$(echo "$line" | grep -oP '\[MUL_MAT\] \K.*(?= FAIL)')
            printf "%-8s %-8s %-8s %-8s %-6s %-8s %s\n" "$type_a" "$m" "$n" "$k" "$bs" "FAIL" "$detail"
        elif echo "$line" | grep -q "OK"; then
            printf "%-8s %-8s %-8s %-8s %-6s %-8s\n" "$type_a" "$m" "$n" "$k" "$bs" "OK"
        fi
    fi
done < "$LOG"

echo ""
echo "=== Summary ==="
TOTAL=$(grep -cE "OK|FAIL" "$LOG" 2>/dev/null || echo 0)
PASS=$(grep -c "OK" "$LOG" 2>/dev/null || echo 0)
FAIL=$(grep -c "FAIL" "$LOG" 2>/dev/null || echo 0)
echo "Total: $TOTAL  Pass: $PASS  Fail: $FAIL"

echo ""
echo "=== Failure Analysis ==="

# Group failures by type
echo "By quant type:"
grep "FAIL" "$LOG" | grep -oP 'type_a=\K[^,]+' | sort | uniq -c | sort -rn

echo ""
echo "By n (batch size):"
grep "FAIL" "$LOG" | grep -oP 'n=\K[^,]+' | sort | uniq -c | sort -rn

echo ""
echo "By k (reduction dim):"
grep "FAIL" "$LOG" | grep -oP 'k=\K[^,]+' | sort | uniq -c | sort -rn

echo ""
echo "By m (output rows):"
grep "FAIL" "$LOG" | grep -oP 'm=\K[^,]+' | sort | uniq -c | sort -rn

echo ""
echo "MMVQ-eligible failures (k>=2048 or n>1, quantized types):"
grep "FAIL" "$LOG" | grep -vE "type_a=f32|type_a=f16|type_a=bf16" | head -20

echo ""
echo "Non-MMVQ failures (k<2048 and n=1, or float types):"
grep "FAIL" "$LOG" | grep -E "k=256.*n=1|type_a=f32|type_a=f16|type_a=bf16" | head -20 || echo "(none)"

echo ""
echo "Full log: $LOG"
