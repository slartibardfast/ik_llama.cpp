#!/usr/bin/env bash
# MTP test matrix runner.
# Runs every test script under tests/mtp-matrix/{coherence,semantics,...}
# against every available build in _build_matrix.sh. Emits a Markdown
# table of results.
#
# Usage:
#   bash tests/mtp-matrix/run-all.sh              # all builds, all categories
#   bash tests/mtp-matrix/run-all.sh coherence    # only the coherence category
#   BUILD=vk-mmvq-off bash tests/mtp-matrix/run-all.sh   # only one build
#
# Exit code: 0 if no NEW failures vs regressions/baseline.json, non-zero otherwise.

set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$SCRIPT_DIR/lib/_build_matrix.sh"

CATEGORIES=${1:-coherence semantics shape ops scheduler server regressions}

# Result aggregation:
# results[build_id][test_name] = {PASS,FAIL,SKIP,CRASH}
declare -A results
declare -A durations
declare -A messages

timestamp() { date '+%H:%M:%S'; }

run_one() {
    local build_id=$1 bin=$2 env_vars=$3 extra=$4 test_script=$5
    local test_name=$(basename "$test_script" .sh)
    local key="${build_id}::${test_name}"

    # Each test has its own port so they can (in principle) run in parallel.
    # We run serially to avoid contention on the single GPU.
    local t0=$(date +%s)
    local out
    out=$(BIN="$bin" EXTRA_ENV="$env_vars" EXTRA_ARGS="$extra" \
          bash "$test_script" 2>&1)
    local rc=$?
    local t1=$(date +%s)
    local dur=$((t1 - t0))

    local status
    case $rc in
        0) status=PASS ;;
        2) status=SKIP ;;
        3) status=CRASH ;;
        *) status=FAIL ;;
    esac
    results[$key]=$status
    durations[$key]=$dur
    # Grab the last PASS:/FAIL:/SKIP:/CRASH: line as the summary message.
    messages[$key]=$(echo "$out" | grep -E '^(PASS|FAIL|SKIP|CRASH):' | tail -1 | cut -c1-120)
    printf "  [%s] %-35s %-6s %3ds\n" "$(timestamp)" "$test_name" "$status" "$dur"
}

# Main loop
build_ids=()
build_labels=()
test_names=()

echo "# MTP test matrix"
echo
echo "Generated: $(date)"
echo
echo "## Progress"
echo

while IFS='|' read -r id label bin env_vars extra; do
    if [ -n "${BUILD:-}" ] && [ "$id" != "$BUILD" ]; then continue; fi
    build_ids+=("$id")
    build_labels+=("$label")
    echo "### $label ($id) — $bin"
    for cat in $CATEGORIES; do
        cat_dir="$SCRIPT_DIR/$cat"
        [ -d "$cat_dir" ] || continue
        for t in "$cat_dir"/*.sh; do
            [ -f "$t" ] || continue
            local_name=$(basename "$t" .sh)
            if [[ ! " ${test_names[*]} " =~ " $local_name " ]]; then
                test_names+=("$local_name")
            fi
            run_one "$id" "$bin" "$env_vars" "$extra" "$t"
        done
    done
    echo
done < <(matrix_available_builds)

echo "## Results"
echo
# Table header
printf "| %-40s |" "Test"
for lbl in "${build_labels[@]}"; do printf " %-28s |" "$lbl"; done
echo
printf "|"
printf -- "------------------------------------------|"
for _ in "${build_labels[@]}"; do printf -- "------------------------------|"; done
echo
for t in "${test_names[@]}"; do
    printf "| %-40s |" "$t"
    for id in "${build_ids[@]}"; do
        s=${results["${id}::${t}"]:-"-"}
        printf " %-28s |" "$s"
    done
    echo
done

# Summary counts
echo
echo "## Summary"
echo
declare -A count
for k in "${!results[@]}"; do
    s=${results[$k]}
    count[$s]=$(( ${count[$s]:-0} + 1 ))
done
for s in PASS FAIL SKIP CRASH; do
    printf -- "- %s: %d\n" "$s" "${count[$s]:-0}"
done

# Exit non-zero if any FAIL or CRASH
if [ "${count[FAIL]:-0}" -gt 0 ] || [ "${count[CRASH]:-0}" -gt 0 ]; then
    exit 1
fi
exit 0
