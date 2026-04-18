#!/usr/bin/env bash
# Output-quality guards. Detect model collapse that substring-matching tests
# miss (e.g., "The\nThe\nThe..." still contains a known word).
#
# The cause of collapse (what these detect):
#   - Draft/target numerically "ties", sampler keeps picking same token
#   - Numerical precision regression (our DeltaNet fix did this on Vulkan)
#   - Bad quantization quirk
#   - Improper KV-cache state carry-over
#
# API:
#   quality_check "<content string>"      → PASS/FAIL with metrics printed
#   quality_metrics "<content string>"    → prints metrics only (for logging)

# Tunables (override via env). Calibrated on Qwen3.5 greedy decoding:
# healthy: unique_ratio ≈ 0.30-0.60, max_bigram_run ≤ 2, Shannon H > 2.8.
# degenerate: unique_ratio < 0.25, max_bigram_run ≥ 3, Shannon H < 2.5.
QUALITY_MIN_UNIQUE_RATIO=${QUALITY_MIN_UNIQUE_RATIO:-0.25}
QUALITY_MAX_SAME_TOKEN_RUN=${QUALITY_MAX_SAME_TOKEN_RUN:-4}
QUALITY_MAX_BIGRAM_LOOP=${QUALITY_MAX_BIGRAM_LOOP:-3}
QUALITY_MIN_SHANNON_H=${QUALITY_MIN_SHANNON_H:-2.5}

quality_metrics() {
    local content=$1
    python3 - <<EOF
import sys, re, math
from collections import Counter
content = """$content"""
# Content-token tokenization: words and punct only, whitespace (inc. \n)
# is a separator. Without this the newline-interleaved token runs like
# "The\nThe\nThe" would look like "The, \n, The, \n" which hides the run.
toks = re.findall(r"[A-Za-z0-9_']+|[^\w\s]", content)
n = len(toks)
if n == 0:
    print("n_tokens=0 unique_ratio=0.00 max_run=0 max_bigram_run=0 shannon_H=0.00")
    sys.exit(0)

unique_ratio = len(set(toks)) / n

# Longest consecutive-same-token run
max_run = 1; cur = 1
for i in range(1, n):
    if toks[i] == toks[i-1]:
        cur += 1; max_run = max(max_run, cur)
    else:
        cur = 1

# Longest AB-AB-AB pattern run (bigram loop length in repeats)
max_big = 1
for span in range(2, min(6, n // 2 + 1)):
    for start in range(n - 2*span):
        k = 1
        while start + (k+1) * span <= n and \
              toks[start + k*span : start + (k+1)*span] == toks[start : start + span]:
            k += 1
        max_big = max(max_big, k)

# Shannon entropy (bits) over token distribution
c = Counter(toks)
H = -sum((v/n) * math.log2(v/n) for v in c.values() if v > 0)

print(f"n_tokens={n} unique_ratio={unique_ratio:.2f} max_run={max_run} max_bigram_run={max_big} shannon_H={H:.2f}")
EOF
}

quality_check() {
    local content=$1
    local metrics=$(quality_metrics "$content")
    echo "$metrics"

    local n_tokens=$(echo "$metrics" | grep -oP 'n_tokens=\K[0-9]+')
    local unique_ratio=$(echo "$metrics" | grep -oP 'unique_ratio=\K[0-9.]+')
    local max_run=$(echo "$metrics" | grep -oP 'max_run=\K[0-9]+')
    local max_big=$(echo "$metrics" | grep -oP 'max_bigram_run=\K[0-9]+')
    local H=$(echo "$metrics" | grep -oP 'shannon_H=\K[0-9.]+')

    if [ "${n_tokens:-0}" -lt 5 ]; then
        echo "QUALITY FAIL: output too short (n_tokens=$n_tokens)"
        return 1
    fi

    python3 -c "
import sys
ur = float('$unique_ratio')
run = int('$max_run')
big = int('$max_big')
H = float('$H')
fails = []
if ur < $QUALITY_MIN_UNIQUE_RATIO: fails.append(f'unique_ratio {ur:.2f} < $QUALITY_MIN_UNIQUE_RATIO')
if run > $QUALITY_MAX_SAME_TOKEN_RUN: fails.append(f'max_run {run} > $QUALITY_MAX_SAME_TOKEN_RUN')
if big > $QUALITY_MAX_BIGRAM_LOOP: fails.append(f'max_bigram_run {big} > $QUALITY_MAX_BIGRAM_LOOP')
if H < $QUALITY_MIN_SHANNON_H: fails.append(f'shannon_H {H:.2f} < $QUALITY_MIN_SHANNON_H')
if fails:
    print('QUALITY FAIL: ' + '; '.join(fails))
    sys.exit(1)
print('QUALITY PASS')
sys.exit(0)
"
    return $?
}
