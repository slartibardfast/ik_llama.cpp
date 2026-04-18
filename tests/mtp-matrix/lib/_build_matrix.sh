#!/usr/bin/env bash
# Canonical list of available binaries / configurations. Used by run-all.sh
# to iterate the backend matrix.
#
# Format: one entry per line:
#   id|label|bin_path|env_vars|extra_args
# where fields can be empty. id must be shell-safe (alnum + '-').

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)

matrix_builds() {
    cat <<EOF
cpu-release|CPU Release|$REPO_ROOT/build-cpu-release/bin/llama-server||
cpu-debug|CPU Debug|$REPO_ROOT/build-cpu-debug/bin/llama-server||
cpu-asan|CPU ASAN|$REPO_ROOT/build-cpu-asan/bin/llama-server|ASAN_OPTIONS=new_delete_type_mismatch=0:halt_on_error=0:detect_leaks=0|
cpu-ubsan|CPU UBSAN|$REPO_ROOT/build-cpu-ubsan/bin/llama-server|UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0|
vk-mmvq-off|Vulkan 2GPU mmvq-off|$REPO_ROOT/build-vk/bin/llama-server|GGML_VK_DISABLE_MMVQ=1|-ngl 999
vk-mmvq-on|Vulkan 2GPU mmvq-on|$REPO_ROOT/build-vk/bin/llama-server||-ngl 999
vk-navi-only|Vulkan NAVI21 only|$REPO_ROOT/build-vk/bin/llama-server|GGML_VK_DISABLE_MMVQ=1|-ngl 999 -ts 1,0 -mg 0
vk-vega-only|Vulkan Vega only|$REPO_ROOT/build-vk/bin/llama-server|GGML_VK_DISABLE_MMVQ=1|-ngl 999 -ts 0,1 -mg 1
vk-no-pp|Vulkan no pipeline-parallel|$REPO_ROOT/build-vk/bin/llama-server|GGML_VK_DISABLE_MMVQ=1|-ngl 999 -sm none
EOF
}

# Emit only builds whose binary actually exists (for partial runs).
matrix_available_builds() {
    matrix_builds | while IFS='|' read -r id label bin env_vars extra; do
        if [ -x "$bin" ]; then
            echo "$id|$label|$bin|$env_vars|$extra"
        fi
    done
}
