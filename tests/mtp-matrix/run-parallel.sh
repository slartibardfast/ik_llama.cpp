#!/usr/bin/env bash
# Parallelized matrix runner.
#
# Key constraint: Vulkan builds contend for the single GPU, so they MUST
# run sequentially. CPU builds are disjoint and run in parallel.
#
# Strategy:
#   phase 1: launch all cpu-* builds concurrently (one subprocess each),
#            each writing results to a per-build log
#   phase 2: wait for phase 1, then run Vulkan builds sequentially
#   phase 3: aggregate all per-build logs into one Markdown table
#
# Usage:
#   bash tests/mtp-matrix/run-parallel.sh [categories...]
#
# Expected speedup: ~4x for the CPU phase (4 parallel builds), no change for
# Vulkan phase. Total time drops from ~60min to ~20-25min for the default
# matrix.
set -u
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/lib/_build_matrix.sh"

CATEGORIES=${*:-coherence semantics shape ops scheduler server regressions}

RESULTS_DIR=/tmp/mtp-matrix-parallel-$$
mkdir -p "$RESULTS_DIR"

run_single_build() {
    local id=$1 label=$2 bin=$3 env_vars=$4 extra=$5
    local out="$RESULTS_DIR/$id.log"
    echo "[$(date +%H:%M:%S)] START build=$id" >> "$out"
    for cat in $CATEGORIES; do
        cat_dir="$SCRIPT_DIR/$cat"
        [ -d "$cat_dir" ] || continue
        for t in "$cat_dir"/*.sh; do
            [ -f "$t" ] || continue
            test_name=$(basename "$t" .sh)
            t0=$(date +%s)
            # Each test gets its own port derived from $$ + build ordinal.
            local test_out
            test_out=$(BIN="$bin" EXTRA_ENV="$env_vars" EXTRA_ARGS="$extra" \
                bash "$t" 2>&1)
            local rc=$?
            t1=$(date +%s)
            local dur=$((t1 - t0))
            local status
            case $rc in
                0) status=PASS ;;
                2) status=SKIP ;;
                3) status=CRASH ;;
                *) status=FAIL ;;
            esac
            printf "%s|%s|%s|%d\n" "$id" "$test_name" "$status" "$dur" >> "$out.results"
            echo "[$(date +%H:%M:%S)] $id $test_name $status ${dur}s" >> "$out"
        done
    done
    echo "[$(date +%H:%M:%S)] DONE build=$id" >> "$out"
}

cpu_ids=()
vk_ids=()
cpu_rows=()
vk_rows=()

while IFS='|' read -r id label bin env_vars extra; do
    row="$id|$label|$bin|$env_vars|$extra"
    if [[ "$id" == cpu-* ]]; then
        cpu_ids+=("$id")
        cpu_rows+=("$row")
    else
        vk_ids+=("$id")
        vk_rows+=("$row")
    fi
done < <(matrix_available_builds)

echo "=== Phase 1: CPU builds (${#cpu_rows[@]} in parallel) ==="
declare -A pids
for row in "${cpu_rows[@]}"; do
    IFS='|' read -r id label bin env_vars extra <<< "$row"
    run_single_build "$id" "$label" "$bin" "$env_vars" "$extra" &
    pids[$id]=$!
    echo "  launched $id (pid ${pids[$id]})"
done
for id in "${!pids[@]}"; do
    wait "${pids[$id]}"
    echo "  $id done"
done

echo
echo "=== Phase 2: Vulkan builds (${#vk_rows[@]} sequential) ==="
for row in "${vk_rows[@]}"; do
    IFS='|' read -r id label bin env_vars extra <<< "$row"
    echo "  running $id"
    run_single_build "$id" "$label" "$bin" "$env_vars" "$extra"
done

echo
echo "=== Phase 3: Aggregation ==="
echo
echo "# MTP matrix (parallel run)"
echo
echo "Generated: $(date)"
echo

# Collect all unique test names and build ids
all_build_ids=()
all_test_names=()
declare -A result_map
declare -A dur_map

for row in "${cpu_rows[@]}" "${vk_rows[@]}"; do
    IFS='|' read -r id label bin env_vars extra <<< "$row"
    all_build_ids+=("$id")
    results_file="$RESULTS_DIR/$id.log.results"
    [ -f "$results_file" ] || continue
    while IFS='|' read -r rid tname status dur; do
        if [[ ! " ${all_test_names[*]} " =~ " $tname " ]]; then
            all_test_names+=("$tname")
        fi
        result_map["${rid}::${tname}"]=$status
        dur_map["${rid}::${tname}"]=$dur
    done < "$results_file"
done

# Build labels lookup
declare -A label_of
for row in "${cpu_rows[@]}" "${vk_rows[@]}"; do
    IFS='|' read -r id label _rest <<< "$row"
    label_of[$id]=$label
done

# Markdown table
printf "| %-42s |" "Test"
for id in "${all_build_ids[@]}"; do printf " %-22s |" "${label_of[$id]}"; done
echo
printf "|"
printf -- "--------------------------------------------|"
for _ in "${all_build_ids[@]}"; do printf -- "------------------------|"; done
echo
# Sort test names alphabetically
IFS=$'\n' sorted_tests=($(printf '%s\n' "${all_test_names[@]}" | sort))
for t in "${sorted_tests[@]}"; do
    printf "| %-42s |" "$t"
    for id in "${all_build_ids[@]}"; do
        s=${result_map["${id}::${t}"]:-"-"}
        printf " %-22s |" "$s"
    done
    echo
done

# Summary
echo
echo "## Summary"
echo
declare -A count
for k in "${!result_map[@]}"; do
    s=${result_map[$k]}
    count[$s]=$(( ${count[$s]:-0} + 1 ))
done
for s in PASS FAIL SKIP CRASH; do
    printf -- "- %s: %d\n" "$s" "${count[$s]:-0}"
done

echo
echo "Per-build log files: $RESULTS_DIR/"

# Exit code
if [ "${count[FAIL]:-0}" -gt 0 ] || [ "${count[CRASH]:-0}" -gt 0 ]; then
    exit 1
fi
exit 0
