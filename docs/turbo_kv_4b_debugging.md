# TURBO_KV_4B Vulkan Flash Attention — Debugging Notes

## The Bug

The TURBO_KV_4B FA shader produced "London" instead of "Paris" for `The capital of France is` on the 0.8B Qwen3.5 model, while F16 and Q4_0 produced correct output.

**Root cause**: Multi-head KV cache stride bug. The shader indexed K/V blocks using `blocks_per_row = HSK / 128 = 2` as the inter-token stride, but with `n_head_kv=2`, the actual stride between same-head tokens in the interleaved cache is `k_stride = nb1/type_size = 4` blocks.

```
KV cache layout (n_head_kv=2, blocks_per_head=2):
  Token 0: [head0_blk0, head0_blk1, head1_blk0, head1_blk1]
  Token 1: [head0_blk0, head0_blk1, head1_blk0, head1_blk1]

Wrong (stride=2): token[1].head0 → blocks 2,3 = token[0].head1 data!
Right (stride=4): token[1].head0 → blocks 4,5 = token[1].head0 data ✓
```

**Fix**: One line change per K/V path:
```glsl
// Before (wrong):
block_idx = offset + kv_token * blocks_per_row + blk;
// After (correct):
block_idx = offset + kv_token * k_blk_stride + blk;
```

## Why This Was Hard to Find

This bug survived weeks of intensive debugging because:

### 1. The error was coherent, not garbled

Reading wrong-head data produces plausible but wrong attention scores. The model generates fluent English ("London" is a capital city), which makes it look like a quality/precision issue rather than a data addressing bug. Garbled output from a stride bug would be immediately suspicious; coherent wrong output looks like quantization noise.

### 2. Synthetic tests passed perfectly

All standalone tests used `nh=1` (single head) or `nh` where `n_head_kv = nh` (no GQA), so `blocks_per_row == k_stride`. The bug only manifests when `n_head_kv > 1` AND the model uses GQA (grouped query attention), which the 0.8B Qwen3.5 does (`n_head=8, n_head_kv=2, gqa_ratio=4`).

The test with KV=32 and nh=1 gave ratio=1.0000 — exact match. This "proved" the shader was correct when it was actually untested for the multi-head case.

### 3. The 9B model "worked"

The 9B model also uses GQA (`n_head=16, n_head_kv=4`) and has the same bug. It produced "Paris" anyway because its larger parameter count makes attention more robust to corrupted KV data. This made us dismiss the bug as "codebook quality" rather than a real error.

### 4. Red herrings consumed enormous effort

We investigated dozens of false leads:
- **Mesa NIR miscompilation**: We traced through 72K lines of ISA dump, found zero `v_sub_f32` in what we thought was the V section, filed a Mesa bug report, built a custom Mesa, patched `nir_opt_algebraic` — all based on checking the wrong line range in the dump file.
- **Shared memory FWHT precision**: We proved the FWHT is bit-identical to CPU (max_diff=0), yet kept searching for shmem precision issues.
- **FP accumulation order**: We analyzed CPU vs GPU dot product patterns (`(a0+a1)+(a2+a3)` vs `vec4 dot + subgroup tree`), tried pre/post scaling, all based on the assumption that the FWHT introduced the error.
- **RADV subgroupShuffleXor bug**: Real bug (garbled output in multi-subgroup WG), but unrelated to the "London" issue.

### 5. The V dump test CONFIRMED the shader was correct

The V value dump showed GPU kv_sh matched CPU to 1.19e-07. The Q·K score dump showed 2.24e-08 match. These results made us conclude the shader was correct and the error was "inherent codebook quality." But the dumps used the SAME synthetic test data (nh=1) that doesn't trigger the bug.

## How to Debug This Class of Error

### Multi-head stride bugs in quantized KV cache

1. **Always test with the model's actual head configuration**. If the model uses GQA (`n_head_kv != n_head`), your test must too. Single-head tests miss stride bugs.

2. **Print the actual tensor strides and compare with your shader's indexing**. The critical values are:
   ```
   k_stride = nbk1 / ggml_type_size(k->type)  // blocks between same-head tokens
   blocks_per_row = HSK / block_size            // blocks within one head of one token
   ```
   If `k_stride != blocks_per_row`, you have multi-head interleaving.

3. **The first generated token tells you everything**. If the very first token is wrong (before any autoregressive accumulation), the bug is in the FA itself, not accumulated error.

4. **Compare with F16 FA on the same model**. The standard F16 FA reads K/V directly from the buffer using `k_stride` from the push constants. If F16 works and your custom FA doesn't, look for stride/offset differences.

5. **Don't trust bit-identical FWHT tests**. An FWHT that's correct on isolated blocks can still produce wrong results if it reads from the wrong blocks due to stride bugs.

### General Vulkan compute shader debugging

1. **Dump intermediate values to the output buffer**. The simplest debugging technique: write kv_sh values, Q·K scores, or V dequant results to `data_o[]` instead of the normal FA output. Then read them back in the test.

2. **Use RADV_DEBUG=shaders for ISA analysis, but verify your line ranges**. The shader dump for a 40K-instruction FA shader is 72K+ lines. The pre-RA section, post-RA section, and assembly are all in the same file. Use `grep -n "ACO shader stage"` to find section boundaries.

3. **Test infrastructure bugs are common**. Our tests had:
   - Missing `ggml_init()` → fp16 lookup table uninitialized → CPU dequant returned zeros
   - KV < Bc without proper mask padding → softmax denominator diluted by `exp(0)` contributions
   - Single-head tests didn't exercise multi-head stride

4. **The CPU fallback path may use a different algorithm**. Our CPU FA uses `ggml_vec_dot_turbo_kv_4b_f32` which rotates Q and dots against codebook K in the RHT domain — a fundamentally different computation than the GPU's Q_rot → kv_sh → vec4 dot path. "CPU works, GPU doesn't" doesn't mean the GPU is doing the same computation wrong — it might be doing a different computation that's only correct under certain assumptions (like single-head).
