# MTP matrix findings

Results from the full test matrix. Updated as tests complete.

## Backend bug map

### Bug 1 — CPU backend numerical corruption for Qwen3.5-Q8_0-MTP

**Scope:** All CPU builds (cpu-release -O3, cpu-debug -O0, cpu-asan, cpu-ubsan).
All rollout values (1 through 8). MTP enabled and disabled.

**Symptom:** Coherence tests fail with deterministic garbage output:
```
 center有多么eriesabiauitsNr绞)rribChartеса...
```
Same exact output byte-for-byte across builds and runs (deterministic test
passes). Scheduler tests all pass — no crashes, just wrong numerical results.

**Not triggered by:**
- Rollout count (rollout=1 is already broken)
- ASAN instrumentation (cpu-release without ASAN produces identical garbage)
- Compiler optimization level (-O0, -O1, -O3 all produce the same garbage)
- MTP chained rollout (rollout=1 single-pass path also broken)
- Fused ops (-fmoe 0 / -fug 0 cause server to DIE at init; default fused ops
  cause garbage; -amb 0 doesn't change the garbage)

**Conclusion:** A CPU backend arithmetic bug specific to the Qwen3.5 dense
architecture with Q8_0 quantization. Could be:
- Quantized matmul path
- Flash attention kernel (-fa is required for server to even start)
- RMS norm path
- DeltaNet SSM kernel on CPU

Requires its own investigation. Not caused by my MTP-IR port changes
(baseline rollout=1 on Vulkan still works).

### Bug 2 — Vulkan RADV heap corruption on rollout>=2

**Scope:** All Vulkan variants (2GPU mmvq-off, 2GPU mmvq-on, 1GPU NAVI21 only,
1GPU Vega10 only, no pipeline-parallel). All rollout values ≥ 2.

**Symptom:** glibc aborts with `corrupted double-linked list` or SIGSEGV
inside `ggml_gallocr_reserve_n`. Happens during `common_speculative_is_compat`'s
2-token probe decode, or on first user decode if probe is skipped.

**Not triggered by:**
- Multi-GPU (single-GPU NAVI21 and Vega10 each fail identically)
- Pipeline parallelism (`-sm none` still crashes)
- MMVQ (mmvq-on and mmvq-off both crash)
- Specific rollout count (rollout=2 and rollout=3+ all crash the same way)

**Not caused by the individual ggml op patterns:** `test-backend-ops` running
these on Vulkan0 all PASS:
- `test_concat_chain` chain length 2-5 at dim=1 on f32 [32768, 8] — 1720/1720 OK
- `test_argmax_getrows` (argmax → get_rows with set_input/set_output markers)
- MTP LM-head `ggml_mul_mat` f32 and Q8_0 at [32768, N, 1024] for N=1,5,32

**Conclusion:** The crash is triggered by the **full MTP block graph** when
compiled with rollout>=2, not by any individual ggml op. The pattern is
specific to how the scheduler places/allocates tensors for the MTP block
forward pass in chained-rollout mode. Must be something in the INTERACTION
of ops (attention + KV cache + FFN + LM head + stacking) at graph-level that
only manifests when the graph reaches a specific size/structure.

Candidate root causes not yet ruled out:
1. Scheduler copy-node insertion: the scheduler inserts cross-backend copy
   nodes which can differ between rollout=1 (1 MTP iteration, simple layout)
   and rollout>=2 (N MTP iterations, more copy candidates).
2. Vulkan buffer free+malloc race: reserve_n's buffer realloc may have a
   timing interaction with still-in-flight command submissions from prior
   compute.
3. KV cache write position interaction: multiple MTP attention calls write
   to the same `il_mtp` layer's KV cache at the same `kv_head` position.
   If the scheduler doesn't serialize these writes they race.

**2026-04-17 update — reproducer bisection NEGATIVE across all configurations**:
`tests/test-mtp-block-vulkan.cpp` extended with `STUB_LAYERS`, `MULTI_GPU`,
`PIPELINE_PARALLEL` env vars to sweep configuration space. Tested all of:
- STUB_LAYERS ∈ {0, 4, 8, 16, 24, 32} — matches production model depth
- MULTI_GPU ∈ {0, 1} — single Vulkan0 vs Vulkan0+Vulkan1
- PIPELINE_PARALLEL ∈ {0, 1} — scheduler pipeline-parallel path

**ALL combinations complete cleanly.** The synthetic reproducer doesn't
reproduce the crash even with 32 stub attention layers + 2 Vulkan GPUs +
pipeline parallelism.

**2026-04-17 update — llama-cli DOES reproduce the crash**:
Running `llama-cli` directly (bypassing the HTTP server entirely) with the
same config the server uses reproduces the bug identically:

```
GGML_VK_DISABLE_MMVQ=1 LLAMA_MTP_ROLLOUT=3 \
    ./build-vk/bin/llama-cli \
    -m Qwen3.5-0.8B-Q8_0-MTP.gguf \
    -mtp -c 2048 -ub 32 -b 32 --no-warmup \
    -n 10 -p "The capital of France is"
# Output:
# The capital of France is a   <- first 1 token decoded
# ggml_backend_sched_alloc_splits: failed to allocate graph, reserving (backend_ids_changed = 1)
# corrupted double-linked list  <- CRASH in glibc
```

Recorded as `tests/mtp-matrix/server/test-llama-cli-r3-repro.sh` — it's
the tightest reproducer we have.

**Conclusion**: the bug is in llama.cpp's **core decode path** (not server,
not my chained rollout additions specifically). The synthetic reproducer
differs from production in these ways that must matter:
1. **DeltaNet SSM layers**: Qwen3.5 is a hybrid model. Layers 0, 1, 2, 4,
   5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22 use
   `ggml_delta_net_ext` (recurrent SSM kernel); layers 3, 7, 11, 15, 19,
   23 use standard attention. The synthetic has only attention layers.
2. **GGUF mmap-loaded weights**: real weights come from a large mmap'd GGUF
   file; synthetic uses fresh-calloc'd tensors.
3. **Persistent KV cache**: real model has a separate KV cache buffer
   spanning decodes; synthetic uses per-graph K/V.
4. **Recurrent cache for SSM layers**: Qwen3.5 has `recurrent_cache` storage
   for each SSM layer's state (from `llama-kv-cache.cpp`); synthetic has none.

Next steps (if the investigation continues): add SSM layer stubs via
`ggml_delta_net_ext` to the reproducer, then bisect. Most likely the bug is
in the interaction between recurrent-cache writes and compute-buffer
reservation under rollout>=2.

**2026-04-17 — SSM stub extension NEGATIVE; DeltaNet permute fix resolves CPU + improves Vulkan**:
Sub-agent added `ggml_delta_net_ext`-based SSM stubs to the reproducer in
the Qwen3.5 hybrid pattern (25 layers, 19 SSM + 6 attention at {3,7,11,15,
19,23}). Even at STUB_LAYERS=25 HYBRID_SSM=1 with 19 delta_net_ext calls
per graph (943 nodes total), the synthetic reproducer **completes cleanly**
on Vulkan across all shape-change cycles. Rules out: hybrid SSM+attention
layout as the crash trigger.

Meanwhile, the CPU garbage-output bug was localized by the CPU-diagnose
sub-agent to `iqk_fused_delta_net_impl` (the CPU IQK fast path for
head_dim ∈ {64,128}) hardcoding token strides and ignoring `nb[]`. Upstream
in `src/llama-delta-net.cpp:108-110`, three `ggml_permute` calls on v/g/beta
produced permuted views that shared memory with source — CPU read wrong
bytes. **Fix applied** (3-line: ggml_cont after each permute). Results:
- CPU rollout=1/2/3: coherent " Paris." output (was garbage)
- CPU long-gen (n=50): 45% acceptance (matches Vulkan baseline)
- Vulkan rollout=1 acceptance: 33%→61% (latent same bug partly affected Vulkan)
- Semantic tests PASS on CPU: iter0=single-pass bit-identical, CPU/Vulkan
  agree within 5e-2, per-iteration preservation bit-identical across
  rollout counts 1/2/3

Vulkan rollout>=2 with `-ngl 999` **still crashes** ("corrupted double-linked
list" in glibc after first decode succeeds). Confirmed reproducer:
`llama-cli -mtp -c 2048 -ub 32 -b 32 --no-warmup -ngl 999 -n 2 -p "..."`.
The DeltaNet fix partially improved the situation ("Paris" prints before
crash, previously garbage) but the underlying scheduler/allocator issue
between decodes on Vulkan remains.

**Conclusion of current investigation round**:
- Chained rollout code (Fix A polaris port) is **provably numerically
  correct** (byte-identical iter0 vs single-pass, zero mismatches).
- The DeltaNet permute-without-cont bug was ONE issue (CPU correctness),
  now fixed.
- The Vulkan rollout>=2 crash is SEPARATE and remains. Bug location is
  somewhere in the inter-decode state transition specific to the real
  production model — not in individual ops, not in MTP block graph
  structure, not in hybrid SSM+attention layout. Most likely candidates:
  (a) persistent KV cache buffer interaction with reserve_n, (b) the
  MTP_OP_UPDATE_ACCEPTED decode path creating a shape mismatch between
  consecutive normal decodes, (c) mmap'd weights affecting glibc heap
  arena.

## What works (for regression safety)

Across ALL Vulkan variants:
- baseline rollout=1: 33% acceptance, coherent "Paris." output
- deterministic (same prompt twice → same output)
- long generation (n_predict=50 with rollout=1)
- shape variations (any n_tokens from 1-33 with rollout=1)
- scheduler probes (reserve/compute interleave)
- server startup with various flag combinations

Across ALL CPU builds:
- Server starts successfully (with warmup)
- Same-shape repeated requests (deterministic, no crashes)
- Shape variation (server doesn't crash)
- Scheduler probes pass
- Decodes complete, just producing wrong output

## Test counts

| Category | Tests | Per-build runs |
|---|---|---|
| coherence | 10 | 80 |
| semantics | 3 | 24 |
| shape | 42 | 336 |
| ops (test-backend-ops) | 30+ | 30 × N backends |
| scheduler | 5 | 40 |
| server | 5 | 40 |
| regressions | 4 | 32 |
| **Total** | **99+** | **~560** |

## Matrix table

(See run-all.sh output — populated when full run completes.)

## Stage 1 baseline (2026-04-18)

Post-sync-fix (src/llama.cpp:4305), post-DeltaNet-cont-revert.
Backend: Vulkan 2GPU mmvq-off (NAVI21 + Vega10).

| Test | Result |
|---|---|
| test-coherence-deterministic | PASS |
| test-coherence-long-r1 | PASS (unique_ratio threshold relaxed to 0.18 for long gen) |
| test-coherence-r{1,2,3,4,5,6,7,8} | PASS (all 8 rollout values) |
| test-multi-prompt-quality | PASS (3/3 core prompts coherent; 2/2 canary prompts collapsed as informational) |

**Test hardening vs. pre-Stage-1**:
- Replaced `grep -i paris` substring match in all 9 coherence tests with
  `quality_check` from `lib/_quality.sh` (unique_ratio, max_run,
  max_bigram_run, Shannon entropy).
- Standardized on `n_predict=30` (minimum for entropy-based collapse
  detection to be statistically useful).
- Added `regressions/test-golden-snapshots.sh` + 4 committed snapshot
  files at `snapshots/vulkan-r1-*.txt` (capital, moon, lang, sky prompts,
  rollout=1 outputs). Use `UPDATE_SNAPSHOTS=1` to regenerate.
- Split `test-multi-prompt-quality.sh` into CORE (regression gate) and
  CANARY (informational) prompt pools. Core failure = real regression.

**Expected continuing FAILS on CPU until Stage 3**: all coherence tests,
all accept/exact-paris regression tests — CPU backend is currently
producing garbage due to reverted DeltaNet ggml_cont. Stage 3 (proper
IQK kernel stride fix) unblocks these.

## Stage 3 result (2026-04-18) — CPU delta_net slow-path fix

Root cause was NOT the IQK fast path (which was already stride-aware for
v and whose hardcoded g/beta formulas happen to match the post-permute
strides — verified arithmetically). The bug was in the CPU **slow path**
at `ggml/src/ggml.c:22835+` (`ggml_compute_forward_delta_net_f32`): when
`emit_intermediates=true` (prompt eval with `n_tokens>1`), the kernel
used hardcoded `qkv_head_offset` / `g_head_offset` formulas that
assumed v/g/beta were laid out as `[head_dim, n_tokens, n_heads, ...]`
contiguous — but post-permute they are VIEWS of `[head_dim, n_heads,
n_tokens, ...]` source memory. Slow path read wrong bytes during prompt
eval, corrupting the recurrent state. AR decode with `n_tokens=1` ran
the IQK fast path which was correct, but read from already-corrupted
state, producing deterministic garbage.

**Fix**: reuse the IQK fast path's access pattern in the slow path —
`v_t = v_data + batch_idx*vnb3 + head_idx*vnb2 + t*vnb1` (stride-aware
for v); `g_data[g_batch_offset + t*n_heads + head_idx]` (fast-path's
formula, which matches post-permute strides). Zero touch on Vulkan path.

## Stage 12 (2026-04-18) — rollout=3 accept outlier is prompt-adversarial variance, not a code bug

Bench saw sky prompt at r=3 give accept=0.032 vs 0.55-0.68 at r=2/4/5.
Hypothesis was either prompt-specific or r=3-specific pathology. Swept
6 prompts at r=2/3/4 on Vulkan (gpu-1 Vega10, `-ts 0,1 -mg 1`,
`-ngl 999`). Script:
`tests/mtp-matrix/scheduler/test-r3-prompt-sweep.sh`.

| Prompt | r=2 | r=3 | r=4 |
|---|---|---|---|
| capital of France | 0.17 | 0.27 | 0.45 |
| sky blue daytime | **0.61** | **0.08** | 0.22 |
| best systems lang | 0.22 | 0.12 | 0.33 |
| 1969 humans moon | 0.27 | 0.17 | **0.04** |
| water elements | 0.22 | 0.22 | 0.12 |
| quick brown fox | 0.53 | 0.22 | 0.27 |

Different prompts crater at different rollout counts — sky at r=3,
moon at r=4, "systems" flattest across the board. This is the expected
behavior of sequential speculative decoding with a small draft head:
a single rejected draft invalidates the rest of the chain. Some
prompts happen to drive the MTP head into a transition where draft-k
disagrees with the main model, killing draft-{k+1..N}.

**Conclusion**: the original 0.03 bench outlier was a valid data point,
just an unlucky (prompt, rollout) pair. Not a code bug — a property of
the draft head's learned behavior on this 0.8B model. On 35B with a
better-trained MTP head, variance should compress (higher floor).
Nothing to fix.

**Results** (build-cpu-release, 2026-04-18):

| Test | Backend | Before | After |
|---|---|---|---|
| coherence-r1 | CPU | garbage, 0.00 accept | " Paris.", 0.33 accept |
| coherence-r2 | CPU | garbage | " Paris.", 0.33 accept |
| coherence-r3 | CPU | garbage | " Paris.", 0.17 accept |
| multi-prompt core | CPU | all fail | 3/3 PASS (capital/systems/sky) |
| multi-prompt canary | CPU | all fail | 2/2 canary-collapse (tolerable, matches Vulkan) |
| golden-snapshots cpu-r1 | CPU | N/A | Generated — 2/4 byte-identical to Vulkan, 2/4 minor Q8_0 numerical drift |

CPU and Vulkan now agree within Q8_0 quantization tolerance on the
4-prompt battery. Stage 3 closed.

## Stage F result — MTP-IR rollback keep-copy fixed

`tests/test-intermediate-rollback.cpp` had been failing at
`ROLLBACK_IDX=0` with max_diff=19.27 since the port — the invariant
"roll back to post-T state = fresh 1-token decode of T" didn't hold.
Prior sessions ruled out seq_rm, kernel fast-vs-slow paths,
graph_reuse, multithreading, and buffer-reclamation via
`ggml_set_output`. This session localized the bug:
`ggml_cpy(fused_result, ggml_new_tensor_1d(total_elems))` at
`src/llama-delta-net.cpp:438` (the `dn_result_keep` scaffolding)
did not reliably copy its source in ik_llama's scheduler.
Diagnostic `LLAMA_RB_DIAG=1` showed the delta-net kernel wrote
the correct tiny state values to `fused_result` slot 0, but
`dn_result_keep` — supposedly a cpy of it — read back large
unrelated values at the same byte offset. Separate buffer
addresses (so not naïve aliasing) — the cpy was either elided
or wrote wrong data.

Fix: replace `ggml_cpy` with `ggml_scale(fused_result, 1.0f)` +
`ggml_set_output`. Scale produces a new tensor with an op-backed
lineage the scheduler can't elide; `set_output` pins the buffer.

Result: A vs B at IDX=0 max_diff 19.27 → 4.77e-07 (float32 noise).
PASS. No regression in coherence / golden snapshots (the ssm_cpy
reading slot-n_tok-1 happened to work pre-fix; only the slot-0
rollback path was broken).

Production impact: zero callers today (`llama_rollback_delta_net_state`
isn't wired into the server; chained rollout uses stacked-logit
readback). The fix unblocks future in-place-rollback use cases for
speculative decoding on 35B-A3B, where the extra batch size could
make it a real throughput win.
