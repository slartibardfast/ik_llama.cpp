# 35B-A3B MTP inline-on-MoE tests

Tests that bring up and guard speculative decoding correctness on QWEN35MOE.

## Test matrix (current state)

### Shell tests (under `tests/mtp-matrix/semantics/`)

| File | Asserts | Vulkan+fit (default) | Vulkan `-no-fmoe -no-fug` |
|---|---|---|---|
| `test-35b-mtp-flag-invariance.sh` | greedy output byte-identical with/without `-mtp` | FAIL | (prompt-dependent) |
| `test-35b-prompt-sweep.sh` | ≥3/4 prompts match no-MTP | FAIL 1/4 | FAIL 2/4 (but different prompt noise) |
| `test-35b-mtp-logits-sanity.sh` | first 3 top-1 tokens match (n_probs=5) | PASS | PASS |

### C++ tests (under `tests/`)

| Binary | Asserts | CPU | Vulkan default | Vulkan `-no-fmoe -no-fug` |
|---|---|---|---|---|
| `test-35b-batch-invariance-sweep.cpp` | 8-pair single-step pos-0 stability | PASS | PASS | PASS |
| `test-35b-pos-i-sequential-equivalence.cpp` | **batch=N pos-i == sequential batch=1 through i** | PASS 16/16 | **FAIL 10/16** | FAIL 0-5/16 (prompt-dependent) |
| `test-35b-rollback-sweep-correctness.cpp` | rollback(token_idx=k) == sequential through k+1 | PASS 4/4 | PASS 4/4 | PASS 4/4 |
| `test-35b-trajectory-drift.cpp` | 50-step batch=1 vs batch=2+rollback (always-reject) | PASS 4/4 | PASS | PASS |
| `test-35b-full-accept-drift.cpp` | always-accept-both traj == batch=1 truth | PASS | FAIL step 4 | prompt-dependent |
| `test-35b-server-flow-drift.cpp` | faithful server simulation | PASS 4/4 | FAIL step 10 | PASS 4/4 on longer prompts, FAIL on short |
| `test-35b-mtp-draft-token-at.cpp` | `llama_get_mtp_draft_token_at(i)` correctness | PASS | PASS | PASS |
| `test-delta-net-emit-path-invariance.cpp` | `delta_net_ext(emit=0)` == `(emit=1)` byte-identical | **FAIL (max\|Δ\|≈0.03)** — IQK fast path bug | PASS | PASS |

## Root causes uncovered + tested

1. Missing inline MTP branch in `build_qwen35moe` — fix: inline MTP ported.
2. `ggml_clamp` buffer aliasing on main logits — fix: removed dead clamp.
3. Duplicate `build_inp_KQ_mask()` producing unfilled input — fix: reuse outer-scope KQ_mask.
4. Server didn't call `llama_rollback_delta_net_state` on partial reject — fix: rollback wire-up gated on `is_hybrid \|\| is_recurrent`.
5. Vulkan `ErrorDeviceLost` after many rollbacks — fix: `ggml_backend_sched_synchronize` at rollback tail.
6. Position-addressed MTP draft read — fix: new `llama_get_mtp_draft_token_at` API + server wire-up.
7. **Delta-net IQK fast path vs slow path diverge by ~3% on CPU** (test-delta-net-emit-path-invariance documents this). Fix: force `emit_intermediates=true` uniformly at `llama-delta-net.cpp:394`.

## Vulkan residual — OP-LEVEL ROOT CAUSE IDENTIFIED

**`test-backend-ops -o BATCH_INVARIANCE` (new, added this session) pinpoints the culprit:**

`MUL_MAT` with quantized weights (q4_K, q8_0) at **K≥2048** FAILS batch-invariance between `n_tokens=1` and `n_tokens=N` by ~5% max absolute delta — ALL 512 output elements differ. F16 weights PASS. K≤1024 PASSES. Other ops (SOFT_MAX, RMS_NORM, SILU/GELU/RELU) all PASS.

Mechanism: `ggml_vk_get_dequantize_mul_mat_vec(ctx, ..., num_cols, ...)` at `ggml-vulkan.cpp:5086` picks different `pipeline_dequant_mul_mat_vec_f32_f32[type][num_cols-1]` variants per batch size. The shader `mul_mat_vec.comp` has a compile-time `NUM_COLS` specialization (layout `constant_id = 2`) and per-variant pipelines. On RDNA2 at K≥2048 the compiled SPIR-V for `NUM_COLS>1` vs `NUM_COLS=1` produces ~5% different pos-0 output — likely due to ACO compiler auto-vectorizing the column-accumulation loop with packed f16 ops for `NUM_COLS>1` while keeping scalar f32 FMA for `NUM_COLS=1`.

**Fused MoE + fused up-gate kernels are the SECONDARY source** (disabled via `-no-fmoe -no-fug`):

Severity dropped from 10/16 → 0-5/16 by passing `-no-fmoe -no-fug` at runtime. On the "capital of France is" prompt the drift is eliminated entirely (0/16). On longer Once-upon prompts it's also eliminated. On the short "Once upon a time, there was a" and "2+2=" prompts, a smaller residual (3-5/16) remains from other ops.

Secondary contributors identified:
- `-no-fa` (disable flash-attn) drops 5→4 mismatches when combined with `-no-fmoe -no-fug`.
- `-no-mmad` — no help (and slightly worse in some combos).

### Fix delivered: `GGML_VK_FORCE_BATCH_INVARIANT=1` runtime flag

Added at `ggml/src/ggml-vulkan.cpp` around `ggml_vk_mul_mat()` and `ggml_vk_mul_mat_id()`. When the env flag is set, BOTH paths bypass the mat-vec specialization entirely and dispatch through the mat-mat shader for all batch sizes. The mat-mat shader uses ONE pipeline variant regardless of N, so pos-0 output is byte-identical whether we decode 1 token or N tokens.

**Verified results (Vulkan + `GGML_VK_DISABLE_MMVQ=1 GGML_VK_FORCE_BATCH_INVARIANT=1 -no-fmoe -no-fug`):**

| Test | Result |
|---|---|
| `test-backend-ops -o BATCH_INVARIANCE` | **57/57 PASS** (was 41/57) |
| `test-35b-pos-i-sequential-equivalence` (4 prompts) | **0/16 mismatches each** (was 3-10/16) |
| `test-35b-server-flow-drift` (4 prompts) | **4/4 PASS byte-identical** (was 1/4) |

Default path unchanged — no perf impact unless the flag is explicitly set. Shader-level tree-reduce / `precise` / explicit-FMA patches (Layers 1–3) were tried and did not close the gap; the ACO-level vectorization across `NUM_COLS` specializations is deeper than GLSL hints can neutralize.

### Bisection methodology (for the next session)

The `-no-fmoe -no-fug` finding came from running `test-35b-pos-i-sequential-equivalence` with individual Vulkan fusion flags toggled:

| Flags | Mismatches (Once upon, 35B Vulkan) |
|---|---|
| (default) | 10/16 |
| `-no-fmoe` | 1/16 |
| `-no-fug` | 10/16 |
| `-no-fmoe -no-fug` | 0/16 (on "capital of France"), 3-5/16 (other prompts) |
| `-no-fmoe -no-fug -no-fa` | 4/16 |
| `-no-fmoe -no-fug -no-fa -no-mmad` | 6/16 (worse) |

The residual 3-5 mismatches after `-no-fmoe -no-fug` must come from more-deeply-embedded kernels. Candidates for future bisection:
- MoE non-fused path (still used when -no-fmoe): router matmul, topk, expert gather, accumulate.
- Attention k/v projection and output projection (mul_mm shader).
- SoftMax / SiLU / RMSNorm fused variants.
- GQA KV write pattern.

To hunt further: instrument Vulkan ops with a trace dump and diff per-layer outputs between batch=1 and batch=N trajectories.

## Running

```bash
cd /home/llm/yarn-agentic/ik_llama.cpp
MODEL=/opt/models/qwen3.5-35b-a3b/Qwen3.5-35B-A3B-MTP-Dynamic.gguf

# Shell suite (Vulkan, recommended settings)
BIN=./build-vk/bin/llama-server \
    MODEL=$MODEL \
    EXTRA_ARGS="--device Vulkan0 --fit -no-fmoe -no-fug" \
    EXTRA_ENV="GGML_VK_DISABLE_MMVQ=1" \
    READY_TIMEOUT=300 \
    bash tests/mtp-matrix/semantics/test-35b-prompt-sweep.sh

# C++ suite (CPU) — all should PASS except delta-net-emit-path-invariance (documents bug)
for t in test-35b-pos-i-sequential-equivalence test-35b-full-accept-drift \
         test-35b-rollback-sweep-correctness test-35b-server-flow-drift \
         test-35b-trajectory-drift test-35b-mtp-draft-token-at; do
  ./build-cpu-release/bin/$t -m $MODEL -c 2048 -b 32 -ub 32 -mtp
done

# C++ suite (Vulkan, with runtime workaround)
for t in test-35b-pos-i-sequential-equivalence test-35b-full-accept-drift \
         test-35b-rollback-sweep-correctness test-35b-server-flow-drift \
         test-35b-trajectory-drift; do
  GGML_VK_DISABLE_MMVQ=1 ./build-vk/bin/$t -m $MODEL -c 2048 -b 32 -ub 32 -mtp \
    -no-fmoe -no-fug --device Vulkan0 --fit
done
```
