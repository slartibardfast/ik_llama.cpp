#!/bin/bash
# Test Q8_1 quantization correctness on GPU
# Validates both plain (block_q8_1) and x4 packed (block_q8_1_x4) output formats
# by comparing GPU-quantized data against CPU reference.
#
# Usage: GGML_VK_VISIBLE_DEVICES=1 ./scripts/test-quantize-q8_1.sh
set -euo pipefail

cd "$(dirname "$0")/.."

TEST=build/bin/test-quantize-q8_1

if [ ! -x "$TEST" ]; then
    echo "Building test-quantize-q8_1..."
    clang++ -std=c++17 -O2 \
        -I ggml/include -I include \
        tests/test-quantize-q8_1.cpp \
        -L build/ggml/src -lggml -lvulkan -lpthread -lm -ldl \
        -Wl,-rpath,build/ggml/src \
        -o "$TEST" 2>&1
    echo "Built."
fi

"$TEST"
