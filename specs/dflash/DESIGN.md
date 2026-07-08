# PHASE_DFLASH — sm_75 implementation design

Target: implement DFlash speculative decoding on `2026-q2-next` (branched from `2026-q2`) of `yarn-agentic` and `ik_llama.cpp`. This document synthesises three reference implementations (vLLM, SGLang, upstream llama.cpp) against our local fork state and the hardware constraints of 2× Quadro RTX 6000 (Turing, sm_75), and picks the design that maximises shipped speedup on this hardware.

The companion Allium behavioural contract lives at `dflash_speculative.allium`. This doc commits the implementation shape; the spec commits the correctness obligations.

---

## 1. The single most consequential decision

**Inject target features via per-anchor K/V cache writes in the draft, not via T5-style cross-attention with per-step encoder rebuilds.**

Both vLLM (`qwen3_dflash.py::precompute_and_store_context_kv`) and SGLang (`dflash_worker.py::_append_target_hidden_to_draft_kv*`) do this. Upstream `llama.cpp` PR #22105 does the opposite — uses `llama_cross` machinery with an encoder context that rebuilds every step. The upstream PR's author flags this as the dominant cost in the "Future Performance Work" section ("graph reuse = 0 because `cross.n_enc` grows monotonically"), and a community datapoint on RTX 3090 (architecturally close to our sm_75) measured **−44.6%** vs baseline on 35B-A3B MoE.

Concretely:

- Projection runs **once** per accepted bonus token, not once per draft denoising step.
- Draft forward becomes plain self-attention against pre-written K/V cache slots — no encoder/decoder dual-context, no `cross.v_embd` monotonic growth, no graph-cache invalidation per cycle.
- The drafter's self-attention path on sm_75 routes to **`mma_new`** (per PHASE5 unit tests) which is byte-deterministic at production shape — we keep the existing kernel determinism guarantees.

The cost is one extra `K_proj + V_proj + RMSNorm(K) + RoPE(K) + cache_write` per draft layer per accepted token. vLLM and SGLang have fused Triton kernels for it on Hopper; **we write the Turing-tuned bespoke equivalent from day one**. Reason: the Gate 0 vLLM measurement showed DFlash round time is ~70% fixed overhead (~108 ms of 148 ms total at spec=15, dominated by PIECEWISE CUDA graphs + drafter forward + per-step kernel launches). On sm_75 we don't pay vLLM's torch.compile/Pythonic tax, but per-layer scalar K/V projection would still cost meaningful time per cycle. **Fused KV projection is load-bearing for Gate 6, not a deferred perf knob.** Scalar reference exists only as a correctness oracle for the unit tests; production code path is always fused. See companion `kernel-design.md` for the kernel signatures and SM_75 register/SMEM budget.

**Drafter is small.** Confirmed from `/opt/models/qwen36-27b-dflash/config.json`: 5 layers (4 sliding-window + 1 full), hidden_size=5120, num_attention_heads=32, num_kv_heads=8 (GQA factor 4), head_dim=128, intermediate_size=17408. Target_layer_ids = [1, 16, 31, 46, 61] (one extracted feature per drafter layer). This reframes the "is a mega-kernel viable on sm_75?" question — the research caution was about 30-layer mega-kernels; 5-layer is well within the published envelope.

## 2. Reuse, don't port

ik_llama.cpp's `common/speculative.cpp` is 1646 lines (vs upstream's ~500), mostly MTP-IR plumbing. The MTP-IR API surface (`llama_mtp_op_type`, `llama_mtp_fused_result`, top-2 cache, draft-argmax cache) is **dead weight under DFlash** — DFlash is an external-draft method, not a head-on-target. We keep it (removing it is a separate cleanup arc) but route the DFlash path through `examples/speculative/` instead of through the MTP API. This matches PHASE_DFLASH_SCOPING.md Option C, modified by §1 above.

What we explicitly reuse from the existing fork:

- The **accept-decision** in `common/speculative.cpp` (argmax_match, longest-prefix, bonus token). The Allium spec `AcceptPrefixDecision` covers both MTP and DFlash with the same semantics; the implementation is one function.
- The **target-side verify forward** — verify is just a multi-position decode of `[bonus, draft_0, …, draft_{N-1}]`. Standard target forward at ne[1] = block_size+1. No new kernels.
- The **slot allocator** for per-seq KV partitioning. np=1 only, so this is trivial; but the existing infrastructure already handles draft-vs-target slot mapping.
- The **GGUF loader** for shared tensors. ik_llama.cpp's loader supports tensor aliases/views; the drafter's `tok_embeddings` and `output` weights point to the target's via `target.*` keys.

What we explicitly do NOT take from upstream's PR #22105:

- **No `llama_cross` / cross-attention plumbing.** Replaced by KV-cache-write (see §1).
- **No encoder/decoder dual-context pattern.** A single draft context with its own KV cache suffices.
- **No EAGLE3 base port.** Upstream's #22105 sits atop EAGLE3 PR #18039 (also draft). The EAGLE3 base contributes the dual-context machinery, which we don't need. We implement DFlash directly.
- **No `ENCODER → DECODER` graph-type override.** Single graph build per draft forward.

## 3. Architecture summary

```
  ┌──────────────────────────────────────────────────────────────┐
  │ Target (Qwen3.6-27B, IQ4_KS, ik_llama.cpp existing kernels)  │
  │                                                              │
  │   ┌────────────┐                          ┌──────────────┐   │
  │   │ verify(    │                          │ extract_     │   │
  │   │   bonus,   │ → logits per position →  │ residual_    │   │
  │   │   block)   │                          │ stream(L_i,  │   │
  │   └────────────┘                          │   anchor_pos)│   │
  │                                           └──────┬───────┘   │
  └──────────────────────────────────────────────────┼───────────┘
                                                     │
                              hidden features at K source layers
                              (one position: the new anchor)
                                                     │
                                                     ▼
       ┌─────────────────────────────────────────────────────────┐
       │ Projection (loaded from drafter GGUF)                   │
       │                                                         │
       │   concat([h_L1, h_L2, …, h_LK]) → fc → hidden_norm      │
       │                                                         │
       │              fused feature ∈ ℝ^(draft_hidden)           │
       └────────────────────────┬────────────────────────────────┘
                                │
                  per draft layer i:
                                │
                                ▼
       ┌─────────────────────────────────────────────────────────┐
       │ Inject (drafter's k_proj_i, v_proj_i)                   │
       │                                                         │
       │   K_i = RoPE(RMSNorm_k(k_proj_i(fused)))                │
       │   V_i = v_proj_i(fused)                                 │
       │   write (K_i, V_i) → draft_kv_cache[layer=i, pos=anchor]│
       └─────────────────────────────────────────────────────────┘

                                │
            for each new accepted anchor (every cycle)
                                │
                                ▼
       ┌─────────────────────────────────────────────────────────┐
       │ Drafter forward (single pass, block_size positions)     │
       │                                                         │
       │  input = [anchor_token, MASK, MASK, …, MASK]            │
       │  positions = [anchor_pos, +1, +2, …, +block_size-1]     │
       │                                                         │
       │  self-attention, layer-type-dependent mask:             │
       │    sliding layers: causal SWA, window = 2048            │
       │    full layer    : bidirectional (non-causal)           │
       │  K/V at anchor_pos come from prewritten cache slots     │
       │  K/V at anchor_pos+k for k≥1 are computed from MASK     │
       │                                                         │
       │  output: logits at positions 1..block_size-1            │
       │  draft_tokens = greedy_argmax(logits)                   │
       └────────────────────────┬────────────────────────────────┘
                                │
                                ▼
       ┌─────────────────────────────────────────────────────────┐
       │ Target verify ([bonus, draft_0, …, draft_{N-1}])        │
       │ Accept longest-prefix; commit bonus.                    │
       │ Loop.                                                   │
       └─────────────────────────────────────────────────────────┘
```

Match this to the Allium contracts: `ExtractFeatures` (top box), `ProjectAndFuse` (projection box), `InjectKV` plus the inline `K_i/V_i` math (inject box, satisfies `InjectionConsumedAtEveryLayer`), `DraftBlockEmit` (drafter forward), `TargetVerifyBlock` + `AcceptPrefixDecision` + `AdvanceState` (verify + accept + commit).

## 4. File-level change plan

For ik_llama.cpp on branch `production/2026-q2-next`. Numbers are rough estimates; the gates in §6 bind the actual work.

| File | Change | Rough LOC |
|---|---|---|
| `src/llama-arch.h` / `src/llama-arch.cpp` | Register `LLM_ARCH_DFLASH`; tensor name mappings (`LLM_TENSOR_DFLASH_FC`, `LLM_TENSOR_DFLASH_HIDDEN_NORM`); GGUF metadata keys (`dflash.target_layer_ids`, `dflash.block_size`, `dflash.mask_token_id`, `dflash.swa_window`, `dflash.target_arch`, `dflash.target_n_embd`). | +60 |
| `src/llama-model.cpp` | DFlash arch dispatch; drafter weight loading (fc, hidden_norm, intermediate layers); shared embed/lm_head from target_model. | +200 |
| `src/llama-build-context.cpp` | New `build_dflash_drafter()` graph (standard transformer with causal-SWA mask). New `build_dflash_inject_kv()` graph (per-layer K_proj/V_proj/norm/RoPE on the fused feature, write to draft KV). Hook the existing qwen3_5_text builder to capture residual-stream snapshots at the K source-layer indices via `ggml_set_output`-style markers. | +250 |
| `src/llama-context.cpp` | `llama_set_dflash(ctx_tgt, model_dft)` API; `extract_dflash_features(ubatch)` async D→H copy at target post-step; `apply_inject_kv(ctx_dft, features, anchor_pos)` runs inject graph and writes to draft KV; per-step orchestration hook. | +250 |
| `include/llama.h` | Public C API surface: `llama_set_dflash`, `llama_get_dflash_block_size`, `llama_get_dflash_mask_token_id`, `llama_get_dflash_swa_window`, `llama_dflash_extract_features`, `llama_dflash_draft_block`. No new types beyond `llama_dflash_state` opaque handle. | +50 |
| `common/speculative.cpp` | `common_speculative_dflash_init/free/draft/accept`. The draft step runs the block-emit, the accept step reuses existing `argmax_match` longest-prefix logic. | +180 |
| `convert_hf_to_gguf.py` | `class DFlashModel` for ik_llama extending the existing Qwen 3.5/3.6 converter base (port adapted from upstream llama.cpp PR #22105's converter and cross-checked against vLLM PR #40898); `--target-model-dir` flag; write fc, hidden_norm, metadata; share tokenizer/vocab from target; BF16→FP16 cast for drafter weights + target's AutoRound-preserved-precision `linear_attn.in_proj_a/b`. | +200 |
| `examples/speculative-simple/speculative-simple.cpp` | Add `--dflash` flag; wire `common_speculative_dflash_*` into the existing speculative loop. | +50 |
| `tools/server/server-context.cpp` | 20-line gate: `if (params.speculative.dflash && params.n_parallel == 1) llama_set_dflash(...)`. Error out at np>1. | +20 |
| Tests | `tests/test-dflash-extract.sh` (residual-stream capture binding test); `tests/test-dflash-inject.sh` (per-layer K/V write binding); `tests/test-dflash-block.cpp` (drafter forward + accept under known prompt); reuse `test_flash_attn_ext_batched_det` at the verify shape (ne[1] = block_size+1). | +250 |

**Total estimated:** ~1500 lines new + adapter glue. Well under PR #22105's +1900 because we skip the encoder/decoder dual-context, the cross-attention plumbing, and the EAGLE3 base.

## 5. Where we differ from each reference impl on hot details

### Block-emit mask
- **vLLM:** layer-type-dependent. sliding_attention layers get `causal=True` (causal SWA); full_attention layers get `causal=False` (bidirectional). Asserted at `vllm/v1/spec_decode/dflash.py:435-444`.
- **SGLang:** `custom_mask=None` defaults to causal across all layers — does NOT match the trained drafter's full_attention layer.
- **Upstream llama.cpp:** no mask (full attention, every position attends to all) — wrong for the SWA layers.
- **z-lab transformers reference:** `is_causal=False` on the drafter forward (`dflash/model.py`), enabling the trained model's internal layer-type-dependent masking.
- **Us:** **layer-type-dependent**, matching vLLM. For Qwen3.6-27B-DFlash specifically: layers 0–3 causal-SWA window=2048, layer 4 bidirectional. Read per-layer from drafter GGUF metadata (`dflash.layer_types`). The block-diffusion bidirectional mixing happens at layer 4 — that's where every mask position attends to every other mask position in the block.

### Sampling at draft step
- **vLLM:** sampler config.
- **SGLang:** `topk=1` always.
- **Upstream:** forces greedy top_k=1 inside DFlash state regardless of CLI.
- **Us:** **greedy only at first landing.** Argmax over draft logits at each position. Probabilistic sampling adds a rejection-sampling layer to AcceptPrefixDecision that's currently scoped out (`ProbabilisticVerifyOutOfScope` invariant in the spec). Add later.

### Projection fan-out
- **vLLM:** one fused `F.linear` emitting `[L * 2 * kv_size, hidden]` for all draft layers at once + fused norm+RoPE Triton kernel.
- **SGLang:** same fused path on supported hardware, per-layer fallback otherwise.
- **Upstream:** the encoder graph IS the projection (one mul_mat + RMSNorm), then per-layer K/V are computed inside the decoder graph via standard wk/wv on the projected feature.
- **Us:** **bespoke fused CUDA kernel from day one** — fused `K_proj + V_proj + RMSNorm + RoPE + cache_write` per layer per anchor, tuned for sm_75 register/SMEM budget. Lives at `ggml-cuda/dflash-inject-kv.cu`. Scalar reference path exists in the same file under a unit-test-only compile flag and is driven by `tests/test-dflash-inject-fused.cpp` for bit-identical correctness validation. **Production code never branches on a scalar-vs-fused mode.** With 5 drafter layers and MAL≈3 anchors per cycle, total inject work is 15 fused-kernel CTAs per cycle — small but on the hot path. Fused execution + WMMA m16n16k16 fp16 path is what keeps it under the per-cycle budget. See `kernel-design.md` for kernel signature, threadblock geometry, and the Allium-invariant ↔ kernel binding table.

### Source-layer indices
- **vLLM:** from drafter checkpoint config (`dflash_config.target_layer_ids`).
- **SGLang:** `build_target_layer_ids(num_target, num_draft)` — evenly spaced in `[1, num_target-3]`, overridable.
- **Upstream:** GGUF metadata key `dflash.target_layer_ids` with `+1` offset from the HF checkpoint config.
- **Us:** **GGUF metadata, read at load time, no CLI override.** The drafter was trained against specific indices; using different indices at inference is wrong. The Allium spec's `FeatureSourceFixedPerDeployment` binds this.

### Block size
- **vLLM:** default 16, configurable via `num_speculative_tokens`.
- **SGLang:** default 16, configurable.
- **Upstream:** GGUF metadata, default 16.
- **Us:** **operating block_size=4** (verify shape `ne[1]=5`), with `--dflash-block-size N` CLI override defaulting to whatever the drafter GGUF declared and a lower configured override at the launch profile. Gate 0 vLLM measurement on Qwen3.6-27B + Qwen3.6-27B-DFlash + INT4 target showed:
  - At `num_speculative_tokens=15` (drafter-declared block_size): per-position acceptance decays from 0.74 → 0.01 across positions 0-14; **positions 7+ are essentially noise**. MAL = 3.28.
  - At `num_speculative_tokens=4`: per-position acceptance is **0.83, 0.64, 0.48, 0.33** (higher than positions 0-3 at spec=15 — drafter concentrates its mass on fewer positions). MAL = 2.91 (only -11% vs spec=15 despite 73% less verify work).
  - Per-token cost ratio (DFlash/vanilla): improves from 1.45× (spec=15) to ~1.03× (spec=4) at np=1.

  We ship at the smaller block_size. The drafter is empirically robust to mask reduction; the earlier DESIGN concern ("drafter sees mask pattern it never saw in training") didn't materialise. Block size remains a launch-profile knob; sweep {4, 5, 6, 8} at Gate 6 to pick the production setting.

  **Verify-shape kernel implication:** `ne[1]=5` is far gentler than `ne[1]=17` on Turing FA kernels — closer to known-deterministic shapes from PHASE5 unit tests (`ne[1] ∈ {2, 4, 8}`). Gate 5's determinism binding becomes lower-risk.

### Multi-slot
- **vLLM:** supports through standard speculative path with **batched verify across slots** (single target forward over all slots' tokens). dflash.py:289-305. **Empirically: accept rate collapses to 0% in many rounds at np=8** (Gate 0 measurement); outputs remain coherent because rejection fallback uses target argmax, but spec decoding becomes "vanilla + drafter overhead" = strictly slower.
- **SGLang:** supports (batched verify), plus mandatory **`--mamba-scheduler-strategy extra_buffer`** for hybrid targets — a ping-pong track buffer per request for the recurrent state at np>1 + overlap scheduling (memory_pool.py:540-735).
- **Upstream llama.cpp:** hard gated to np=1.
- **Us:** **np=1 only at this landing.** Gate 0 measurement on vLLM (the reference oracle) showed DFlash at np>1 doesn't deliver speedup on any stack — at np=4: speedup 0.29×; at np=8: speedup 0.21× — because the drafter's logits drift under batched matmul, the rejection sampler's argmax_match rejects most drafts, and DFlash degrades to "vanilla + drafter overhead."
  Gate 3.5 (DFlash multi-slot byte-determinism with identical prompts, np=2) came back GREEN — outputs are byte-identical across slots. But the failure mode at np>1 with diverse prompts is *not* output corruption; it's accept-rate collapse. The np>1 question is therefore answered as "structural perf wall" not "correctness wall."
  **Per-slot dispatch stays the multi-slot path if we ever add np>1** (correctness-deterministic by construction at |B|=1), but it's behind a hard gate. Gate 7 is closed: batched verify at np>1 doesn't pay even on the reference impl. See companion memory `project_continuous_batching_vs_perslot_dispatch.md` for the bandwidth-amortization analysis showing vanilla batched > DFlash batched at np>1 on this hardware.

### Hybrid target recurrent state at np>1

Qwen3.6-27B has **48 of 64 layers as `linear_attention`** (DeltaNet) and **16 as `full_attention`**. The DeltaNet layers have a recurrent state per slot that must be saved-and-revertable across the speculative cycle: verify mutates the state across all `block_size=16` positions optimistically; on rejection the state must revert to the accept-prefix boundary.

SGLang's `HybridReqToTokenPool` implements this via `req_index_to_mamba_ping_pong_track_buffer_mapping` — size 2 for overlap scheduling, size 1 for non-overlap. **At our `block_size=16` (vs MTP's `--draft 3`), the per-cycle recurrent-state revert footprint is ~16× more aggressive than the MTP path. At np=8 that's 128 simultaneous in-flight states to track and potentially revert.**

Spec invariant `HybridTargetRecurrentStateTracking` codifies this requirement. Our implementation must provide an equivalent of SGLang's mamba ping-pong tracking, regardless of whether we adopt overlap scheduling (we don't, at first landing) — even sequential per-slot dispatch needs to revert recurrent state on rejection.

### Distributed parallelism constraints

Spec invariants `DPAttentionNotSupported` and `PipelineParallelismRequiresPpSizeEq1` mirror SGLang's hard-block enforcement (`server_args.py:3247-3306`). ik_llama.cpp currently has neither DP nor PP, so these are forward-looking guards. Tensor parallelism (which we use: `--tensor-parallel-size 2` for our 2× RTX 6000 split) is supported.

### Multimodal bypass is implementation work, not inherited

vLLM PR #40898 has a no-op `_warn_if_multimodal` override (`dflash.py:120-124`) that *allows* multimodal inputs without implementing actual routing. **Inheriting that "support" is unsafe** — drafter logits at vision tokens are statistical noise (no vision tower in drafter weights), and the verify path will reject them slowly. ik_llama.cpp must implement bypass at the SERVER layer: detect vision/video/audio tokens in the incoming prompt and route around the speculative path before `llama_dflash_draft_block` is called. Spec invariant `MultimodalTurnsRoutedAroundDrafter` now explicitly names this as caller-side enforcement, not inherited.

### VRAM budget at np=8

Production 2× Quadro RTX 6000 = 48 GiB aggregate. Drafter K/V (4 SWA layers cap at window=2048, 1 full layer scales with ctx) is ~10% of target K/V at every scale; the binding cost is **target K/V**.

| ctx/slot | Target KV @ np=8 | Drafter KV @ np=8 | Total (with 28 GiB target + 3.5 GiB drafter + 3 GiB scratch) | Fits |
|---|---|---|---|---|
| 256k | ~36 GiB | ~2.7 GiB | ~73 GiB | ✗ |
| 128k | ~18 GiB | ~1.4 GiB | ~54 GiB | ✗ |
| **64k** | **~9 GiB** | **~0.7 GiB** | **~44 GiB** | ✓ |
| 32k | ~4.6 GiB | ~0.4 GiB | ~39 GiB | ✓ |

**Production profile target for np=8: ctx-per-slot ≤ 64k.** This is the binding constraint, not a perf knob. Spec invariant `ContextBudgetAtNp8` codifies it.

### Determinism guarantees
- **vLLM:** silent.
- **SGLang:** silent.
- **Upstream:** silent.
- **Us:** **per-deployment determinism is a closure gate.** Under fixed (block_size, target_layer_ids, draft GGUF), runs are byte-identical. Cross-deployment is not bit-stable for the same kernel-batch-shape reason we know from multi-slot. Make the binding test before merging.

### CLI surface
- **Upstream:** `--dflash` flag in `llama-speculative-simple`.
- **Us:** `--dflash` plus `--dflash-block-size N` plus `--dflash-think on|off` (the equivalent of `LLAMA_SPEC_NO_THINK` for Qwen3-family — major accept-rate lever per upstream PR notes). The server picks these up via the existing `params_base.speculative.*` plumbing.

## 5.5 Batch-invariance recipe (np>1 ship target requires this)

We design for np>1 from day one (np=1 is a subset). Gate 5b binds: drafter logits per slot must be **bit-identical across np ∈ {1, 2, 4, 8}** for the same slot input. vLLM's Gate 0 measurement showed what happens otherwise — accept rate collapses to 0% at np=8 because batched-matmul logit drift trips `argmax_match`. Bit-invariance is enforced by the Thinking Machines Lab 3-kernel pattern (see [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) and llama.cpp PR #16016 for a CUDA reference implementation).

The three batch-invariant kernels and their constraints:

- **RMSNorm** — one CUDA block per row. No cross-block reduction. Inner sum via `__shfl_xor_sync` warp-shuffle butterfly; block-level via `__syncthreads` + SMEM tree where needed. **No `atomicAdd<float>` ever.**
- **MatMul / GEMM** — fixed compile-time tile dims for every matmul on the drafter path. No Split-K, no global_reduce. Each output tile is fully accumulated within one CTA across the entire K dimension. Persistent-style: tiles are dispatched by `blockIdx`, not by arrival order.
- **Attention (verify-shape FA)** — fix split-**size**, not split-count. `KV_BLOCK_SIZE` is a compile-time constant. The number of splits varies with sequence length, but each split's accumulation is independent of how many concurrent rows the kernel is processing. One block per output row.

**Forbidden on the drafter path:**
- **Marlin INT4 kernels** — Marlin uses Split-K with `global_reduce` barrier; per-row output is M-dependent (confirmed by source reading of `IST-DASLab/marlin/marlin/marlin_cuda_kernel.cu`). Cannot be made batch-invariant without rewriting `global_reduce`. **Build infrastructure forbids linking Marlin on the drafter path.** Drafter weights stay fp16 (no quantization) for the first landing; if drafter ever quantized in the future, ggml's MMQ Q4_0 path is the only allowed kernel (already batch-invariant on sm_75).
- **`ggml_cuda_flash_attn_ext_wmma_f16` as-is** — PHASE45 D10.e probe (`MEMORY.md:3300-3500`) measured row asymmetry at layer 3 onward (δ=1.27e-3, doubling every ~4 layers). The verify kernel must be a new file (`dflash-verify-attn.cu`) implementing the fixed-split-size pattern from scratch; we don't inherit WMMA-FA's row-asymmetry bug.
- **Cross-block `atomicAdd<float>`** — non-deterministic accumulation order. Inner reductions via warp-shuffle, block-level via SMEM tree, grid-level via explicit kernel ordering (grid.sync).
- **Occupancy-tuned heuristics for tile shape** — every tile dim must be a compile-time constant. No "pick best tile based on M".

**Reference implementations to study (in this order):**

1. [llama.cpp PR #16016](https://github.com/ggml-org/llama.cpp/pull/16016) — deterministic inference mode in CUDA. Implements the TML 3-kernel pattern. Closest reference for what we need to write.
2. [thinking-machines-lab/batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) — canonical Triton implementation (architecture-agnostic).
3. [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing) — only public FA forward kernel actually tuned for sm_75 with head_dim=128, ~63% of peak on T4. Read for the verify-attn layout (we write our variant; this is the structural reference).
4. [CUTLASS sm_75 paths](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/warp) — `ldmatrix` swizzle and per-warp tile patterns.

**Numerical determinism layered constraints:**
- `wmma::mma_sync` is itself deterministic on sm_75 (tensor cores produce identical output for identical input).
- `ldmatrix.sync` on sm_75 has **no per-lane predication** — all 32 lanes must provide valid SMEM addresses (sm_80+ tolerated unused lanes). For verify-shape kernels with M<16, this means padding SMEM addresses, not masking lanes.
- SMEM swizzle pattern (or `+8 halves` row-stride padding) is required to avoid 32-way bank conflicts on the K-major operand of `ldmatrix`. CUTLASS-style XOR swizzle `(row * stride) ^ ((row & 0x7) << 3)` is conflict-free for fp16 K=64.
- Block-to-SM mapping is non-deterministic, but block_idx → output tile mapping is fixed at kernel design time. Tiles assigned by `blockIdx`, not by arrival order. SM-arrival order is irrelevant to output values.

**Acceptance gate**: Gate 5b binds bit-invariance via test-dflash-determinism-np-invariance (new test). The test hashes per-slot drafter logits at np∈{1,2,4,8} for the same slot input and asserts all hashes match.

## 6. Gates — order of work and falsifiable stop conditions

These are gates in the §5-of-CLAUDE.md sense: each must close before the next opens. Skipping a gate is forbidden per `feedback_no_skip_tests.md`.

### Gate 0 — M0: reference measurement on vLLM (DONE — RED with caveats)

Before any ik_llama.cpp implementation work. Run vLLM PR #40898 build + Qwen3.6-27B INT4 AutoRound target + Qwen3.6-27B-DFlash drafter on 2× RTX 6000 sm_75. Measure speedup vs vanilla decode at np=1, np=4, np=8 with `num_speculative_tokens` sweep. **This is the reference oracle, not a ship target** — we expect vLLM's bespoke-kernel ceiling to be higher than what the vLLM stack delivers.

Result (2026-05-12, `data/gate0-np1-np4-np8.json`, `data/gate0-spec4-np-sweep.json`, `data/gate0-mtp-sweep.json`):

| config | DFlash tok/s | vanilla tok/s | speedup |
|---|---:|---:|---:|
| spec=15 np=1 | 22.20 | 31.53 | 0.70× |
| spec=4 np=1 | 24.46 | 30.96 | 0.79× |
| spec=4 np=4 | 33.14 | 113.60 | 0.29× |
| spec=4 np=8 | 32.92 | 154.91 | 0.21× |
| MTP-method spec=3 np=1 | **44.19** | 28.91 | **1.53×** |

Findings:
- DFlash with separate Qwen3.6-27B-DFlash drafter on INT4 target: doesn't beat vanilla on vLLM at any np. Drafter-target dtype mismatch (BF16-trained drafter vs INT4 target) reduces MAL to 3.28 (paper implies 4.8-6.4).
- vLLM MTP method (using target's built-in MTP layers): **1.53× speedup at np=1** with MAL=2.87. Multi-slot stable (no accept-rate collapse) — confirms architectural advantage of integrated MTP heads over separate drafter.
- vLLM fixed overhead per DFlash round: ~108 ms (PIECEWISE CUDA graphs + drafter forward + per-step Triton dispatch). **This is the headroom for our bespoke sm_75 implementation** — we don't pay this tax.

- **PASS for ik_llama.cpp implementation: vLLM oracle is in hand.** Sufficient empirical signal to design the bespoke kernels: target round time < 50 ms (vs vLLM's 118 ms), expected ~60-80 tok/s at np=1 spec=4, gives 1.8-2.4× over MTP `--draft 3` baseline. Proceed to kernel design.
- **FAIL on speedup ambition:** if profiled fixed overhead in our impl is comparable to vLLM's, the speedup ambition collapses. Diagnose before Gate 4.

Token cost (already incurred): ~80k. Gate 0 closed.

### Gate 1 — converter binding

Port `class DFlashModel` into ik_llama.cpp's `convert_hf_to_gguf.py`, extending the existing Qwen 3.5/3.6 converter base. Convert `z-lab/Qwen3.6-27B-DFlash` (target = `Qwen/Qwen3.6-27B` Int4 AutoRound) into a `.gguf` that has all expected tensors (fc, hidden_norm, drafter layers) and metadata (target_layer_ids, block_size, mask_token_id, swa_window, target_arch, target_n_embd, layer_types). Verify by `gguf_dump` and tensor-name match against the upstream-converted version on the same source repo.

- **PASS:** Tensors and metadata match.
- **FAIL:** Missing or mismatched. Diagnose; don't proceed.

Token cost: 15-25k.

### Gate 2 — extract hook

Implement `dflash_extract_features` against the Qwen 3.6 build graph (`Qwen3_5ForConditionalGeneration` arch class — hybrid linear_attention + full_attention). Hook the residual-stream snapshot at K source-layer indices (`target_layer_ids = [1, 16, 31, 46, 61]`). Build a binding test that runs a known prompt through `Qwen3.6-27B` target and captures hidden states at the source layers; compare against an upstream-converted reference run (vLLM PR #40898's extract path) on the same prompt + same target weights.

- **PASS:** Byte-identical hidden-state snapshots at all source-layer indices.
- **FAIL:** Either the hook fires at the wrong tensor or the snapshot mis-orders the layer/position dimensions. Fix; re-run.

Token cost: 15-25k.

### Gate 3 — fused InjectKV kernel + drafter forward

**Two sub-gates that must both close:**

**Gate 3a — Fused InjectKV correctness (unit-test, no full pipeline yet).**
Implement `ggml-cuda/dflash-inject-kv.cu` per `kernel-design.md`. Write the scalar reference path in the same file (compiled only under unit-test flag). Drive both paths from `tests/test-dflash-inject-fused.cpp` with random inputs across the full drafter layer-config space (head dims, KV head counts, RoPE base, RMSNorm eps).

- **PASS:** Fused output byte-identical to scalar reference across all 5 drafter layers.
- **FAIL:** Mismatch. Diagnose at the fused-kernel level; scalar reference is the ground truth.

**Gate 3b — End-to-end drafter forward with fused inject.**
Plumb `apply_inject_kv` + `build_dflash_drafter` through the existing graph. Build a binding test: given fixed target features and a known anchor token, run drafter forward at `block_size=4` and compare draft logits against the upstream / vLLM impl on identical inputs.

- **PASS:** Byte-identical (or within 1e-5 NMSE) draft logits at block_size=4.
- **FAIL:** Logits differ. Diagnose: inject path, SWA mask, K_proj/V_proj weights, or fusion correctness (re-run Gate 3a).

Token cost: 40-60k (combined; fused kernel + plumbing).

### Gate 4 — full block-emit + accept loop

Plumb `common_speculative_dflash_*` into `examples/speculative-simple`. Run the "quicksort, thinking on" benchmark on Qwen3.6-27B target + Qwen3.6-27B-DFlash drafter at np=1 on the production TP=2 layout.

- **PASS:** Within 10% of the Gate-0 vLLM oracle measurement (24.46 tok/s at spec=4 np=1) on the same hardware. We expect to beat it once Gate 6 measurements run; Gate 4 only binds that the end-to-end loop produces coherent output with non-zero accept rate.
- **FAIL:** Substantial regression. Drafter is producing low-acceptance blocks, or the accept loop is mis-counting. Diagnose.

Token cost: 30-50k.

### Gate 3.5 — DFlash multi-slot empirical test (vLLM) (DONE — GREEN with caveats)

Run on 2026-05-12. Same prompt at np=2 with greedy decode against `Qwen/Qwen3.6-27B` target + `z-lab/Qwen3.6-27B-DFlash` drafter. Result: `data/gate35-dflash-determinism.json` — verdict **GREEN**, n_diff_pairs=0, all 5 streams byte-identical.

**Caveat surfaced by subsequent Gate 0 multi-prompt measurements:** byte-determinism with identical prompts at np=2 is necessary but not sufficient for "DFlash works at np>1." With *diverse* prompts at np=8, accept rate collapses to 0% in many rounds (the rejection-sampler argmax_match path becomes brittle to logit drift from batched matmul). Outputs remain coherent (rejection fallback uses target argmax) but spec decoding doesn't deliver speedup. See §5 Multi-slot for full discussion.

Net: Gate 3.5 GREEN does NOT unlock Gate 7. The np>1 question is closed on the empirical evidence: DFlash at np>1 doesn't pay. Resolved `OQ-DFLASH-INHERITS-MTP-MULTISLOT-BUG` in the Allium spec.

Token cost (already incurred): ~10k.

### Gate 5 — 27B determinism (np=1 single-slot)

Run the determinism fixture (`scripts/test-mtp-multislot-determinism.sh` adapted to DFlash) at np=1 on Qwen3.6-27B + Qwen3.6-27B-DFlash. Compare to non-DFlash np=1 production baseline for byte-equivalence after a fixed-temperature greedy run. **Also bind FA kernel determinism at ne[1]=17 (verify shape with block_size=16) on sm_75** — this is the kernel-shape question the earlier block-size decision deferred to this gate.

- **PASS:** Within-deployment byte-identical across 3 runs with same prompt + fixed block_size; FA at ne[1]=17 byte-deterministic across 3 runs.
- **FAIL on output determinism:** Diagnose (likely DFlash KV cache state isn't fully deterministic across cycles — check anchor advance, slot eviction).
- **FAIL on ne[1]=17 FA determinism:** specialize the FA dispatcher for this shape, or extend the kernel-determinism test matrix. Block-size stays at 16; do not fall back to 8 (would mis-match training).

Token cost: 10-20k.

### Gate 5b — Drafter np-invariance (NEW; binds np>1 ship target)

After Gate 5 (np=1 determinism). Verify that drafter logits per slot are bit-identical across np ∈ {1, 2, 4, 8} for the same slot input. The test runs the bespoke drafter forward kernel at each np value with N identical-content slots and hashes per-slot output logits. All hashes must match.

- **PASS:** SHA-256 of per-slot drafter logits identical across np ∈ {1, 2, 4, 8}. The TML 3-kernel batch-invariance recipe (§5.5) is correctly implemented. Gate 7 unlocks.
- **FAIL:** Some slot's logits differ across np values. Diagnose: which kernel introduced shape-dependence (likely a matmul tile heuristic or an attention split-count variation). Fix per §5.5 constraints. Re-run.

Test file: `tests/test-dflash-determinism-np-invariance.cpp`. Runs at every PR.

Token cost: 15-25k.

### Gate 7 — Batched verify at np>1 (REOPENED, conditional on Gate 5b)

vLLM's Gate 0 measurement showed DFlash and MTP both lose to vanilla batched at np>1 on the vLLM stack. Root cause: drafter logits drift under batched matmul → rejection sampler argmax-mismatches → accept rate collapses to 0% → spec decode becomes "vanilla + drafter overhead" = slower.

**Our bespoke kernels are designed to defeat this**: §5.5 commits the TML 3-kernel batch-invariance recipe + Gate 5b binds it. **If Gate 5b passes**, accept rate at np>1 should preserve at np=1 levels (MAL ≈ 2.91 at spec=4). With preserved accept rate, DFlash at np=8 produces ~3 × N = 24 tokens per round vs vanilla's 8, giving ~3× aggregate speedup over vanilla batched. **This is the actual prize.**

Open Gate 7 only after Gate 5b GREEN.

- **PASS:** Aggregate DFlash throughput at np=8 ≥ 1.8× vanilla batched throughput at np=8 (measured on same hardware, same prompt set). Ship np=8 profile.
- **NEUTRAL (1.0–1.8×):** DFlash wins at np>1 but not by enough to justify the operational complexity. Document; ship np=1 only as Gate 6.
- **FAIL (<1.0×):** Gate 5b passed but DFlash still loses to vanilla batched. Diagnose: kernel-level inefficiency in our verify pass, or DFlash's verify-cost growth at higher np exceeds its accept-rate advantage. Document; ship np=1 only.

Token cost: 30-50k.

### Gate 6 — Qwen3.6-27B speedup

Run the production-prompt set on Qwen3.6-27B + DFlash at np=1, IQ4_KS target, `block_size=4` (sweep {4, 5, 6, 8} to confirm operating point) with thinking ON (production behavior — Qwen 3.6 is a thinking model).

**Pre-Gate baseline measurement** (per `feedback_anchor_to_measured_baselines.md`): fresh measurement of MTP `--draft 3` on the *same prompt set, same hardware, same build* before any DFlash comparison. Don't anchor on the memory entry's 33.5 tok/s figure.

- **PASS:** ≥1.5× speedup over the fresh MTP baseline at np=1 with thinking ON. Ship to a new profile (`qwen36-27b-x1-dflash.sh`); leave existing MTP profile in place as fallback.
- **NEUTRAL (1.0×-1.5×):** DFlash works but doesn't clear the ship bar. Document the gap honestly; keep MTP as production; ship DFlash as a tunable option for users who prefer DFlash for other reasons (lower variance, different acceptance curve). Don't dress this as "GREEN."
- **FAIL (<1.0×):** DFlash with this drafter on this hardware doesn't beat MTP. Stay on MTP. Document the negative result with the cost breakdown that surfaced the wall.

Realistic expectation calibration from Gate 0: vLLM DFlash at spec=4 = 24.46 tok/s with 70% fixed overhead. If our bespoke kernels halve the fixed overhead (~108 ms → ~50 ms), we land in the 45-70 tok/s range. MTP baseline is ~33.5. So PASS is plausible but not assured. **Don't pre-commit to PASS** — let the kernel work measure.

Token cost: 20-30k.

**Total budget if everything passes:** ~140-220k. Gate 0 was a real measurement (closed with data), Gate 3 grew (fused kernel + plumbing), Gate 7 is reopened conditional on Gate 5b GREEN.

## 7. Risk surface

### R1 — Drafter still under training
The HF card warns: *"This model is still under training, and inference engine support may not be fully available yet due to architectural changes, including causal SWA layers."* The drafter weights may change. Mitigation: pin to a specific Qwen3.6-27B-DFlash revision (commit + safetensor hashes captured at convert time and recorded in MEMORY.md); re-pin only by deliberate decision. Treat any upstream weight change as a separate phase that re-binds Gate 5 + Gate 6.

### R2 — Per-layer projection cost (RESOLVED by §5 design choice)
Scalar K_proj+V_proj+norm+RoPE on every accepted anchor token would be ~5-10 ms × 5 drafter layers × MAL≈3 anchors ≈ ~75-150 ms per cycle — would consume Gate 6's entire perf budget even at the smaller 5-layer count. §5 commits to the fused kernel from day one for exactly this reason. R2 is no longer "risk to monitor"; it's a design constraint that drives the InjectKV kernel work. The risk shifts to **R2′ — fused kernel correctness**, bound by Gate 3a.

### R3 — `--target-model-dir` converter coupling
The drafter GGUF references the target's vocab + tokenizer at convert time. If the user later swaps the target's vocab (e.g., adding tool tokens), the drafter must be re-converted. This is an operational hazard but matches upstream's design — flag in the launch profile.

### R4 — Multi-slot drift
If anyone enables np>1 with DFlash on, target features extracted from a multi-slot batch are entangled across slots (same as the multi-slot determinism bug). Hard gate at np=1 in the server is essential; an env override that bypasses it would silently corrupt outputs. Don't add an override.

### R5 — Kernel batch-shape sensitivity at verify (REDUCED)
With block_size=4 (§5 update), verify batch is `ne[1] = 5`. PHASE5 unit tests cover `ne[1] ∈ {2, 4, 8}` and route to `mma_new` deterministically; `ne[1]=5` falls in the deterministic gap but is much closer to known-good shapes than the original `ne[1]=17`. Gate 5's determinism binding still mandatory but lower-risk.

### R6 — Drafter SWA on Turing
We have not stress-tested causal-SWA self-attention on sm_75 at the drafter's layer count. Gate 3 binds the drafter forward correctness; Gate 5 binds determinism. If a kernel dispatch falls back to a non-deterministic path under SWA, the gates catch it.

## 8. What success looks like at end of Gate 6

- New profile `/home/llm/profiles/qwen36-27b-x1-dflash.sh` (alongside existing `qwen36-27b-x1-mtp.sh`)
- Drafter GGUF on `/mnt/archive` or `/opt/models/recast-out/`
- Production server can swap between MTP and DFlash by symlink flip on `/home/llm/profiles/active.sh`
- Branch `production/2026-q2-next` carries the implementation; merge to `production/2026-q3` (next release branch) once verified
- Measured speedup ≥1.5x over MTP at np=1 on production prompts
- Per-deployment byte determinism preserved
- Binding tests live under `tests/` in ik_llama.cpp + harness scripts in yarn-agentic's `scripts/`

## 9. What success doesn't try to achieve

- **(MOVED IN-SCOPE 2026-05-12) np>1 concurrency for spec decoding** — was closed by Gate 0 empirically (DFlash/MTP both lose to vanilla batched at np>1 on vLLM). The vLLM failure mode was batch-shape-dependent drafter logit drift collapsing accept rate, NOT a structural property of multi-slot spec decoding. Our bespoke kernels are designed to defeat this via the TML 3-kernel batch-invariance recipe (§5.5); Gate 5b binds it. If Gate 5b passes, np>1 is the actual prize (Gate 7). See companion memory `project_continuous_batching_vs_perslot_dispatch.md` for the architectural reasoning.
- **Multimodal image-text-to-text.** Separate workstream (`project_qwen36_27b_multimodal_exploration` memory entry).
- **Replacing MTP entirely.** MTP stays as the fallback profile; DFlash adds an alternative. The user can flip between them. Per Gate 6 NEUTRAL outcome handling, even if DFlash doesn't clear 1.5× we ship as an option.
- **Tree drafting / multi-branch DFlash.** Block-emit only. Tree drafting on hybrid recurrent attention is unsolved (`project_tree_fanout_hybrid_recurrent_blocker` memory entry).
- **Thinking-mode suppression for accept-rate.** Qwen 3.6 is a thinking model by design. We measure with thinking ON (production behavior). The Gate 0 finding that `/no_think` only marginally improves MAL (3.28 → 3.55) confirms this isn't the lever to pull.

## 10. Anchor on the existing memory

The four most load-bearing existing memory entries for this workstream:

- `project_mtp_multislot_determinism_investigation_failed.md` — Why np>1 stays gated. The same surface bites DFlash if anyone touches it.
- `project_production_2026q2_landing.md` — Current production tip. The new branch forks from this; the running service is unaffected.
- `feedback_anchor_to_measured_baselines.md` — Don't compute uplift % against estimates. Gate 6 must compare against a freshly-measured MTP baseline on the same hardware.
- `feedback_probe_before_implementing.md` — Gate 0 (M0) is the probe. Skipping it would repeat the PHASE4 mistake of building 100k+ tokens of scaffolding on an untested premise.

## 11. Open implementation choices that the spec leaves to this design

Resolved from `dflash_speculative.allium`'s open questions, where this doc commits:

- **OQ-SOURCE-LAYERS:** GGUF metadata, no CLI override. (§5 "Source-layer indices")
- **OQ-DENOISE-SCHEDULE:** single_step only at first landing. (§5 "Sampling at draft step")
- **OQ-BLOCK-SIZE-FOR-TURING:** **operating block_size=4**, sweep {4, 5, 6, 8} at Gate 6 to pick production setting. The drafter is empirically robust to mask reduction — the earlier concern about training-mask mismatch did not materialize on the vLLM measurement. (§5 "Block size")
- **OQ-QUANT-MIX:** Target IQ4_KS, drafter BF16. Shared embed/lm_head materialized from the target's IQ4_KS tensor — drafter never re-quantizes. (Allium `SharedEmbedAndLMHead` invariant.) **Note:** Gate 0 confirmed BF16-drafter / INT4-target combo costs MAL ≈ 1.5-3 vs paper's 4.8-6.4. We accept this as the operating point; the published 2.4-3.2× speedup is not the target — Gate 6's 1.5× over MTP is.
- **OQ-SWA-WINDOW:** read from drafter GGUF metadata `dflash.swa_window`. Surfaced via `llama_get_dflash_swa_window`.
- **OQ-THINK-MODE-EQUIVALENT:** Qwen 3.6 is a thinking model by design. **Ship with thinking ON (production behavior).** Gate 0 showed `/no_think` is only marginally effective (MAL 3.28 → 3.55) and the model frequently ignores the directive. The `--dflash-think on|off` CLI flag remains for experimentation but defaults to ON. (Inverted from original "default off" per upstream's measurements — those measurements were on a non-thinking model variant.)
- **OQ-MULTI-SLOT:** np=1 only. Hard gate. (Reinforced by Gate 0 data showing np>1 doesn't pay for any spec method on this hardware.)
- **OQ-VLLM-PR-MERGE-RISK:** we anchor on the empirical Gate 0 vLLM measurements + the published paper, not on the PR's implementation choices. vLLM PR is correctness oracle.
- **OQ-FAILURE-MODE-ON-MISSING-DRAFT:** at first landing, hard fail (matches MTP-IR's current pattern). Move to graceful-fallback after the path is shipped and measured.
- **OQ-FUSION-PATH:** **bespoke fused CUDA kernel from day one** (resolved 2026-05-12). Scalar reference exists only as unit-test oracle. See `kernel-design.md` for kernel signatures and SM_75 budget. (§5 "Projection fan-out")
- **OQ-NP-SCOPE:** np>1 is design target from day one; np=1 is a subset (resolved 2026-05-12). Reopens Gate 7 conditional on Gate 5b GREEN. (§5 "Multi-slot", §5.5 "Batch-invariance recipe")
- **OQ-DRAFTER-PRECISION:** drafter weights stay fp16 (no quantization). Drafter was trained against BF16 targets; quantizing drafter weights compounds drift on top of target INT4 perturbation. ~3.3 GiB on disk fits comfortably at fp16 (~1.65 GiB/GPU at TP=2). Marlin kernels are explicitly forbidden on the drafter path (build-time guard) — if drafter ever quantized later, only ggml MMQ Q4_0 is allowed (already batch-invariant on sm_75). (§5.5)
- **OQ-DRAFTER-FORWARD-STRUCTURE:** persistent single mega-kernel — all 5 drafter layers + lm_head in one cooperative launch with grid.sync between layers. The original "30-layer mega-kernel on sm_75 is outside published envelope" concern doesn't apply at 5 layers. (Lock #11, resolved 2026-05-12)
- **OQ-VERIFY-ATTN:** dedicated `ggml-cuda/dflash-verify-attn.cu` written from scratch using Thinking Machines Lab's fixed-split-size pattern + sm_75-specific PTX `mma.sync.m16n8k8` (NOT WMMA C++ API, NOT the existing `wmma_f16` FA path — both have batch-shape sensitivity that violates Gate 5b). Reference layout from `ssiu/flash-attention-turing` for head_dim=128. (§5.5)
- **OQ-BATCH-INVARIANCE-RECIPE:** TML 3-kernel pattern + llama.cpp PR #16016 as CUDA reference. Bit-identical drafter logits across np ∈ {1,2,4,8} for same slot input is the Gate 5b binding test. (§5.5)

## 12. Companion documents

- **`dflash.allium`** — the behavioural contract. 19 top-level invariants + 35 in-contract @invariants. Updated 2026-05-12 with Gate 3.5 + Gate 0 resolutions.
- **`kernel-design.md`** — the bespoke sm_75 kernel design (next deliverable). Locks the InjectKV fusion signature, drafter-→inject buffer layout, verify-shape attention contract, and the Allium-invariant ↔ kernel binding table. Must land before kernel code in `ggml-cuda/`.
- **`DFlashCycle.tla`** + **`DFlashMultiSlot.tla`** — TLA+ models. Phase 1 (cycle) 851 states clean; Phase 2 (multi-slot) 6067 distinct states clean. Bind structural correctness of the cycle and per-slot dispatch.
- **`allium-tla-binding.json`** — Allium↔TLA+ binding manifest. 24 tla_helpers whitelist; check-bindings.py enforces 54/54 Allium invariants coverage.
- **`upstream-pr-drafts.md`** — three small vLLM PRs (combine_hidden_states dtype cast, FlexAttention view→reshape, BLOCK_M/N POW2) maintained locally as runtime monkey-patches in `scripts/vllm_sm75_patches.py`. These keep the vLLM oracle running on sm_75.
- **Companion memories**: `project_continuous_batching_vs_perslot_dispatch.md`, `project_qwen36_27b_multimodal_exploration.md`, `project_mtp_multislot_determinism_investigation_failed.md`, `project_production_2026q2_landing.md`.
