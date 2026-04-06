#!/bin/bash
# Clean build of ik_llama.cpp with Vulkan backend
# Usage: ./scripts/clean-build.sh
set -euo pipefail

cd "$(dirname "$0")/.."
echo "Working directory: $(pwd)"

# Step 1: Kill any leftover shader gen processes
echo "=== Step 1: Clean slate ==="
pkill -9 -f vulkan-shaders-gen 2>/dev/null || true
sleep 1

# Step 2: Clean and configure
echo "=== Step 2: Configure ==="
rm -rf build
cmake -B build \
  -DGGML_VULKAN=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  2>&1 | tail -3

# Step 3: Build shader generator
echo "=== Step 3: Build shader generator ==="
cmake --build build --target vulkan-shaders-gen -j$(nproc) 2>&1 | tail -3

# Step 4: Generate SPIR-V shaders
echo "=== Step 4: Generate shaders ==="
./scripts/build-shaders.sh

# Step 5: Build ggml and llama-cli
echo "=== Step 5: Build ggml ==="
cmake --build build --target ggml -j$(nproc) 2>&1 | tail -3

echo "=== Step 6: Build llama-cli ==="
cmake --build build --target llama-cli -j$(nproc) 2>&1 | tail -3

echo "=== Step 7: Build test-backend-ops ==="
cmake --build build --target test-backend-ops -j$(nproc) 2>&1 | tail -3

echo ""
echo "BUILD COMPLETE"
echo "  llama-cli:        build/bin/llama-cli"
echo "  test-backend-ops: build/bin/test-backend-ops"
