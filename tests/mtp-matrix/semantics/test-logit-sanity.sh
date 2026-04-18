#!/usr/bin/env bash
# Logit-sanity probe: verifies the LM head never produces pathological
# log-probs across a diverse prompt battery. Serves as the safety net
# before removing the per-iteration `ggml_clamp(-1e4, 1e4)` in the
# chained rollout loop at src/llama-build-context.cpp:4966.
#
# The clamp was defensive; the hypothesis is that a trained Qwen3.5
# LM head never produces logits of magnitude near 1e4. If so, the
# clamp is a dead kernel dispatch per iteration and can be removed.
#
# We proxy "logit magnitude" via log_softmax = logprob. The top-40
# log-probs from /completion should stay in a sane band:
#   - top-1 logprob >= MIN_TOP1 (-5.0 allows a low-confidence peak)
#   - All top-40 logprobs >= MIN_ANY (-20.0 covers normal-range noise)
#   - No NaN / -inf
#
# Threshold choice: Qwen3.5 vocab = 248320. log(1/248320) = -12.4, so
# a uniform distribution puts every token at -12.4. A well-trained
# model gives top-1 around -0.5 to -3.0 and top-40 in [-0.5, -12.0].
# -20.0 is well outside that band but still finite.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/../lib/_common.sh"
source "$SCRIPT_DIR/../lib/_logits.sh"
ROLLOUT=${ROLLOUT:-1}

MIN_TOP1=${MIN_TOP1:--5.0}
MIN_ANY=${MIN_ANY:--20.0}

PROMPTS=(
    "The capital of France is"
    "The best programming language for systems is"
    "Explain why the sky appears blue during daytime"
    "In 1969 humans first landed on"
    "Water is made of two elements:"
)

matrix_setup
if ! matrix_launch; then echo "CRASH"; exit 3; fi

any_fail=0
for prompt in "${PROMPTS[@]}"; do
    file=/tmp/logit-sanity-$$-$(echo "$prompt" | md5sum | cut -c1-8).json
    logits_capture_to "$file" "$prompt" 3

    res=$(python3 - <<PY
import json, math, sys
d = json.load(open("$file"))
top1_min = float('inf')
any_min = float('inf')
nan_or_inf = False
for step in d:
    top = step.get('top', [])
    if not top: continue
    # top[0] is highest prob token
    p0 = top[0][1]
    if p0 <= 0 or math.isnan(p0) or math.isinf(p0):
        nan_or_inf = True
        continue
    lp0 = math.log(p0)
    if lp0 < top1_min: top1_min = lp0
    for tok, prob in top:
        if prob <= 0 or math.isnan(prob) or math.isinf(prob):
            nan_or_inf = True
            continue
        lp = math.log(prob)
        if lp < any_min: any_min = lp
if nan_or_inf:
    print("FAIL_NAN_INF")
    sys.exit(0)
print(f"{top1_min:.3f} {any_min:.3f}")
PY
)
    rm -f "$file"
    case "$res" in
        FAIL_NAN_INF)
            echo "  FAIL: '$prompt' produced NaN or inf logits"
            any_fail=1
            continue ;;
    esac
    top1_min=$(echo "$res" | awk '{print $1}')
    any_min=$(echo "$res" | awk '{print $2}')
    printf "  prompt='%s' top1_min=%s any_min=%s\n" "$prompt" "$top1_min" "$any_min"
    if awk -v v="$top1_min" -v t="$MIN_TOP1" 'BEGIN{exit !(v+0 < t+0)}'; then
        echo "  FAIL: top1_min $top1_min < $MIN_TOP1"
        any_fail=1
    fi
    if awk -v v="$any_min" -v t="$MIN_ANY" 'BEGIN{exit !(v+0 < t+0)}'; then
        echo "  FAIL: any_min $any_min < $MIN_ANY"
        any_fail=1
    fi
done

matrix_teardown
if [ "$any_fail" -ne 0 ]; then
    echo "FAIL: logit-sanity probe"; exit 1
fi
echo "PASS: logit magnitudes sane across $(echo "${#PROMPTS[@]}") prompts"
exit 0
