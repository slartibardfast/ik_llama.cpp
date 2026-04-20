# Vulkan batch-invariance test matrix

Run: `./build-vk/bin/test-backend-ops -o BATCH_INVARIANCE -b Vulkan0`
(no env flags — `GGML_VK_DISABLE_MMVQ` no longer needed; MMVQ routing off in source pending Phase 3.)

Hardware: AMD Radeon RX 6800 XT (RADV NAVI21), RDNA2.

## Current state — 101/101 PASS, Backend Vulkan0 OK, no env flags

| Op | Variants tested | PASS | FAIL |
|---|---|---:|---:|
| **MUL_MAT** | F16 / Q4_K / Q8_0 × K∈{256,512,1024,2048,2048,2048,2048} × M × N∈{2,4} | **42** | 0 |
| **SOFT_MAX** | K∈{64,512,4096} × N∈{2,4} | 6 | 0 |
| **RMS_NORM** | K∈{64,128,256} × N∈{2,4} | 6 | 0 |
| **UNARY (SILU/GELU/RELU)** | K=1024 × N=4 | 3 | 0 |
| **FUSED_UP_GATE** (dense) | Q4_K / Q8_0 × K × M × N∈{2,4} | 8 | 0 |
| **MOE_FUSED_UP_GATE** | Q4_K / Q8_0 × K × M × n_experts=16 × n_used=2 × N∈{2,4} | 8 | 0 |
| **MUL_MAT_ID** | Q4_K / Q8_0 × K × M × n_experts=16 × n_used=2 × N∈{2,4} | **16** | 0 |
| **FLASH_ATTN** | hs=128 × nh_q=16 × nh_kv∈{8,2} × kv=512 × N∈{2,4,8} × {f16acc,f32acc} × mask=1 | **12** | 0 |
| **TOTAL** | | **101** | **0** |

## Completion notes (2026-04-20)

All six routing-split classes are uniform now:

| Op | Split that broke BI | Uniform choice |
|---|---|---|
| MUL_MAT | n=1 → MMVQ shader, n=N → dequant-mmv | MMVQ disabled in source; both hit dequant-mmv |
| MUL_MAT | n=1..8 → separate NUM_COLS spec-const pipelines | NUM_COLS=max spec-const everywhere, runtime `p.num_cols` loop bound → one ISA for all n |
| MUL_MAT_ID | n=1 → mat-vec-id shader, n=N → mat-mat-id | mat-vec-id routing removed; all n → mat-mat-id |
| FLASH_ATTN | n=1 → split-K reduce, small-rows tile, gqa_ratio remap; n=N → large-rows single-pass | split_k forced off, `small_rows=false`, GQA n=1 remap disabled → uniform large-rows single-pass |
| FUSED_MOE_UP_GATE | n_tokens > 1 → fused `ggml_moe_up_gate` path, n_tokens == 1 → unfused mul_mat_id + mul_unary | fused branch disabled at build-context level; all n → unfused path |
| FUSED_UP_GATE (dense) | cur->ne[1] > 1 → fused `ggml_fused_up_gate` path, ne[1] == 1 → unfused | fused branch disabled at build-context level; all n → unfused path |

`GGML_VK_FORCE_BATCH_INVARIANT` env flag and its helper function deleted. `GGML_VK_DISABLE_MMVQ` no longer needed. `-no-fmoe` / `-no-fug` no longer needed for invariance (the gate is off in source).

## 35B-A3B validation (2026-04-20, MTP-Dynamic.gguf, --fit, Vulkan0, no env flags)

| Test | Result |
|---|---|
| `test-35b-pos-i-sequential-equivalence` pairs=4 batch=4 | 0/16 mismatches |
| `test-35b-pos-i-sequential-equivalence` pairs=8 batch=8 | 0/64 mismatches |
| `test-35b-batch-invariance-sweep` | 0/5 mismatches |
| `test-35b-trajectory-drift` 20 steps | byte-identical |
| `test-35b-full-accept-drift` 20 steps | byte-identical (historic FAIL step 4) |
| `test-35b-server-flow-drift` 20 steps | byte-identical (historic FAIL step 10) |

## Perf posture

- 0.8B tg128 Q4_K_M: 20.88 ± 0.02 t/s (matches the 20.50 baseline within noise)
- 35B-A3B tg64 (CPU-spilled on 16GB GPU): 1.75 t/s
- 35B PP throughput expected to be lower due to disabled fused-MoE / fused-up-gate pp kernels; not measured.

Perf restoration (rewire fused kernels to produce byte-identical output to the unfused path at every batch size) is future work. Correctness is non-negotiable; these paths stay off until they can be proven BI.

## Phase 1 landing notes (2026-04-20)

MUL_MAT 42/42 byte-identical — the `K≥2048 × N≥2` cliff is gone. Summary of what was actually needed vs what the investigation initially guessed:

1. **Pipeline collapse for `mul_mat_vec` family** — `layout (constant_id = 2) const uint NUM_COLS = 1;` stayed in place as the compile-time array allocator, but the `for (uint j = 0; j < NUM_COLS; ++j)` loops were rewritten to `for (uint j = 0; j < p.num_cols; ++j)` (runtime push constant). All 8 pipeline entries per quant type now specialize NUM_COLS to `mul_mat_vec_max_cols` (=8), so `ggml_vk_create_pipeline(..., {subgroup_size, NUM_ROWS, mul_mat_vec_max_cols}, ...)` produces one ISA; the pipeline cache sees identical (SPV + spec-const-value) tuples across the 8 entries. Phase 1 per plan.

2. **`temp` array zero-init stayed at NUM_COLS bound**, not p.num_cols, so any compiler-vectorized access across the j axis can't pick up garbage from uninitialized slots. (Empirically: the collapse alone did not close K=2048 at this step — the MMVQ divergence in §3 had to be fixed before the full invariance showed.)

3. **MMVQ disabled in `ggml_vk_should_use_mmvq`** — the actual K=2048 divergence cause was NOT the NUM_COLS pipeline specialization at all. It was `n=1` routing through the integer-dot MMVQ shader (gated `ne11==1 && ne10>=2048 && AMD`) while `n=N` fell back to the dequant `mul_mat_vec`. Two different algorithms, ~5 % magnitude delta (one whole q8_0 block mis-accumulated). Fix: `return false;` at the top of `ggml_vk_should_use_mmvq` with a comment pointing at Phase 3 (MMVQ batch-variance fix — extend MMVQ to handle ne11>1 with matching dequant-mmv semantics).

4. **NoContraction injection path fix in `vulkan-shaders-gen.cpp`** — the `../ggml/src/vulkan-shaders/scripts/inject_no_contraction.py` relative path was resolving against the CMake build cwd (`build-vk/ggml/src`) and silently missing — the injection had never run, so every prior "ACO ignores NoContraction" conclusion was actually a test of no-op code. Changed to `join_paths(input_dir, "scripts/...")` so the path is absolute-relative-to-source. Verified: `spirv-dis mul_mat_vec_q8_0_f32_f32.spv | grep -c NoContraction` now reports 20. Separately confirmed NoContraction decorations in the embedded SPV do NOT close the K=2048 cliff — it took the MMVQ fix (§3) to land invariance. Kept the injection as defensive posture: it prevents a class of mul+add→FMA drift that would otherwise re-emerge if ACO scheduler semantics change.

### Perf posture

MMVQ was the fast path at ne11=1 for AMD+K≥2048. Disabling it regresses tg throughput at those shapes. Phase 3 owns the fix — extend the MMVQ shader / dispatch to ne11>1 and land matching-semantics batching, then re-enable.

### Removed workaround knobs

`GGML_VK_DISABLE_MMVQ` is no longer part of the BI reproducer command. `GGML_VK_FORCE_BATCH_INVARIANT` / `-no-fmoe` / `-no-fug` remain alive in the source but are no longer needed to pass MUL_MAT invariance; they become relevant again only for MUL_MAT_ID / FLASH_ATTN failures that Phases 2–4 own.

## MUL_MAT_ID detail

Non-fused MoE mat-mat (`-no-fmoe` path). Output is `[M, n_used=2, n_tokens]`; we compare token-0's `M*n_used` float slice.

| type_a | K | M | N | max\|Δ\| |
|---|---:|---:|---:|---:|
| q4_K | 256 | 256 | 2,4 | 0.0149 |
| q4_K | 256 | 512 | 2,4 | 0.0184 |
| q4_K | 2048 | 256 | 2,4 | 0.0992 |
| q4_K | 2048 | 512 | 2,4 | 0.111 |
| q8_0 | 256 | 256 | 2,4 | 0.0155 |
| q8_0 | 256 | 512 | 2,4 | 0.0155 |
| q8_0 | 2048 | 256 | 2,4 | 0.121 |
| q8_0 | 2048 | 512 | 2,4 | 0.166 |

Same fingerprint as MUL_MAT: K=2048 is worst (0.1 scale), K=256 is smaller but still non-invariant. Every output element differs (`diff=M*n_used`). Consistent with the Qwen3.5 finding that `-no-fmoe` alone does not restore determinism: the non-fused MUL_MAT_ID path also routes through the same batch-variant mul_mat_vec pipeline that MUL_MAT uses. Neither N value converges to OK, and the delta is invariant across N — matches the MUL_MAT pattern.

## FLASH_ATTN detail

Qwen3.5 shapes: hs=128, nh_q=16, nh_kv∈{8,2} (GQA), kv=512, N∈{2,4,8}, f32acc ∈ {off, on}, mask=1. Uniform zero mask (all keys attended).

| nh_kv | N | f32acc | max\|Δ\| | diff/total |
|---:|---:|:---:|---:|:---:|
| 8 | 2,4,8 | 0 | 9.31e-09 | 1671/2048 |
| 2 | 2,4,8 | 0 | 7.45e-09 | 1637/2048 |
| 8 | 2,4,8 | 1 | 9.31e-09 | 1671/2048 |
| 2 | 2,4,8 | 1 | 7.45e-09 | 1637/2048 |

**All 12 fail, but the delta is near-ULP (~1e-8).** Distinctive findings:
1. **`f32acc=0` and `f32acc=1` produce identical deltas** — toggling the accumulator precision does not change the BI fingerprint on Vulkan0, suggesting the variance source is not the inner-loop accumulator but the split_k / workgroup-scheduling layer outside the small-matmul.
2. **N is irrelevant** — N=2/4/8 all give identical max\|Δ\| and diff-count. On Vulkan0/NAVI21 the path taken is the same scalar variant for these nb values (cm2/cm1 are not supported; see `ggml_vk_flash_attn_scalar_shmem_support` and the `path == FA_COOPMAT2 && N == 1 → FA_SCALAR` fallback). Small_rows is triggered when `N <= get_fa_num_small_rows(FA_SCALAR)`; this matrix sits inside the scalar path across all three N.
3. **nh_kv changes the GQA ratio** (16/8=2 vs 16/2=8) which changes split_k allocation. Different diff-counts (1671 vs 1637) and slightly different max\|Δ\| confirm split_k reduction is in the chain, but both still fail.
4. Not tested: `use_mask=0`, cm1/cm2 paths (not available on NAVI21), `kv` variations that exercise the aligned vs unaligned shaders.

The tiny magnitude suggests the root cause is reduction-order variability in the split_k_reduce pass or the intra-workgroup accumulation, not a structurally wrong kernel.

## Cross-check: Vulkan1 (Vega / VEGA10, warp=64) — reference

Running the same matrix on Vega produced the identical PASS/FAIL shape: MUL_MAT_ID 16/16 FAIL, FLASH_ATTN 12/12 FAIL. Magnitudes: MUL_MAT_ID is within ~5% of RDNA2; FA deltas were 1.12e-08 (slightly larger than 9.31e-09 on NAVI21). The BI bugs are not RDNA2-specific — they are in the shared Vulkan pipelines.

## Key finding (updated)

- MUL_MAT_ID joins MUL_MAT as a batch-variant op. Both rely on mul_mat_vec when N≤8, consistent with Phase 1's diagnosis: fixing mul_mat_vec should close MUL_MAT_ID too.
- FlashAttention is technically batch-variant but at ULP magnitude — likely a reduction-order issue in split_k_reduce, not a logic bug. Order of priority: MUL_MAT/MUL_MAT_ID first (0.05–0.17 deltas propagate), FA second (sub-ULP).

## Phase gating (updated after Phase 1 attempt)

- Phase 1 (`mul_mat_vec` single-pipeline dispatch fix) — **attempted and reverted.** Changed `ggml_vk_mul_mat_vec_q_f16` to force `dispatch_num_cols = 1` and dispatch `groups_y = ne11` when `batch_n`. Verified via stderr debug print that dispatch was actually using the NUM_COLS=1 pipeline with total_y=N. BI still showed 16 MUL_MAT failures with identical 0.05 magnitude. **The patch worked as designed but the bug is deeper.** With the NUM_COLS=1 pipeline dispatched at grid Y=1 vs Y=N, column-0 output still differs. So the divergence is NOT in the NUM_COLS spec const per se — it's at the dispatch-grid-size level, meaning workgroup scheduling or memory-access patterns themselves produce different f32 output for the same pipeline + same inputs at different grid shapes. Patch reverted because it adds risk surface without closing the bug; the existing `GGML_VK_FORCE_BATCH_INVARIANT` flag remains the working correctness switch.
- Verified state (2026-04-19): without flag → 57/101 PASS; with `GGML_VK_FORCE_BATCH_INVARIANT=1` → **89/101 PASS** (closes all 16 MUL_MAT + 16 MUL_MAT_ID = 32 cases). Remaining 12 failures are FlashAttention at ULP (~1e-08) magnitude; do not propagate to argmax flips at T=0.
- Phase 2 (`fused_up_gate` / `moe_fused_up_gate` tile collapse) — **NO-OP**. Op-level BI already passes.
- Phase 3 (MMVQ) — pending, may share the real root cause.
- Phase 4 (FlashAttention split_k_reduce reduction-order) — **open**. 12 failures, all ≤1e-08.
- **Phase 1b bisection in progress.** Already ruled out:
  - **4→2→1 outer unroll cascade** (Track B hypothesis #3): replaced with flat `for (i = 0; i < num_iters; ++i)` loop. BI still fails at identical 0.05 magnitude. Not the cause.
  - **`subgroupAdd` reduction ordering** (additional hypothesis): forced `has_sg = false` so all pipelines use the deterministic tree-reduction path instead. BI still fails identically. Not the cause.
  - q8_0 doesn't use `repack()` (that's q4_K/q4_0-specific) yet q8_0 shows the same 0.05 failure magnitude as q4_K. So repack alone is not sufficient to explain the bug.
- **Determinism probe landed** (extended `test_batch_invariance_mul_mat` with Run A + Run B both at n=1, plus Run C at n=N). Result on Vulkan0 q8_0 K=2048: **`det=OK` on all 16 failing cases** — the shader is fully deterministic, same inputs → same output. The 0.05 divergence is NOT GPU nondeterminism; it's a reliable function of the batch-size context (either pipeline spec const OR dispatch grid size).
- **Exhausted GLSL/SPIR-V source-level controls** (2026-04-19 session):
  - `precise` on accumulator (global and function scope), on `rowtmp`, on `temp[][]`: **no effect**.
  - Explicit scalar FMA chain (`fma(a,b,c)` for 8 lanes instead of `dot(vec4,vec4)`): **no effect**.
  - `[[unroll]]` removed from the j loop (runtime-bounded): **no effect**.
  - Single compiled pipeline with runtime `p.num_cols` push const (NUM_COLS=8 always): **no effect**.
  - Removed 4→2→1 outer unroll cascade in favor of flat `for (i = 0..num_iters)`: **no effect**.
  - Forced `has_sg = false` to bypass subgroupAdd reduction in favor of deterministic tree-reduce: **no effect**.
  - `spirv_execution_mode(31)` (ContractionOff): **blocked — OpenCL-only execution mode, rejected by Vulkan shaderc as requiring `Kernel` capability**.
  - `spirv_execution_mode(SignedZeroInfNanPreserve=4461) + RoundingModeRTE=4462` via `SPV_KHR_float_controls`: compiles on Vulkan, **no effect** (ACO doesn't honor for this pattern).
  - Every attempt produced IDENTICAL 0.0501 max-abs-delta — confirmed by sanity multiplier test (`temp *= 1000` scaled to 1.19e+04 proportionally, so shader edits DO land).
- **Conclusion:** the bug is at the ACO compiler level, below GLSL/SPIR-V source semantic control. Requires either (a) a MESA/RADV upstream fix (via `gitlab.freedesktop.org/mesa/mesa` issue + MR), or (b) `spirv-opt` post-processing that injects decorations ACO respects, or (c) dispatching through an entirely different kernel (the `GGML_VK_FORCE_BATCH_INVARIANT=1` workaround path, which routes to `mul_mat_q_f16` at ~34% tg perf cost).
- **Still to probe in Phase 1b:**
  - `block_q8_1_x4` struct-shaped SSBO access / per-block scale read pattern — shared between q8_0 and q4_K which both fail identically.
  - Per-thread partial-sum dump via auxiliary SSBO binding (requires C++ dispatch-side changes to add a debug buffer). Purpose: localize whether divergence enters during the FMA/dequant body or during reduction.
  - Phase 1 patch reinstated as A/B control: if we force NUM_COLS=1 pipeline + grid Y=N, do per-thread partials still differ from the default NUM_COLS=N pipeline + grid Y=1? That separates the two hypotheses.

## Ops not yet BI-tested

- `MMVQ` directly (currently covered implicitly via MUL_MAT without DISABLE_MMVQ).
- FA variants with `use_mask=0` (mask-free path).
- FA with cm1/cm2 pipelines (N/A on NAVI21).
- FA `kv % align == 0` vs unaligned probes (the aligned shader dispatches only when strides ≡ 0 mod 8; currently contiguous F16 strides are always aligned at hs=128).
