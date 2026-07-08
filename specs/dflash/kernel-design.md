# PHASE_DFLASH — sm_75 / TU102 kernel design

Companion to `DESIGN.md`. This document locks the **CUDA kernel signatures, data layouts, SMEM/register budgets, stream sequence, and Allium-invariant ↔ kernel binding table** for the bespoke DFlash implementation in ik_llama.cpp on `production/2026-q2-next`.

`DESIGN.md` commits the implementation shape and gate sequence. This doc commits the kernel-level contracts.

Per CLAUDE.md §5: changes to this file commit and push immediately, separately from other changes. Per `feedback_surface_tradeoff_decisions.md`: any deviation from the contracts here must be surfaced explicitly.

---

## 1. Locked design decisions

Twenty architectural decisions locked through Q&A round 2026-05-12 (`DESIGN.md` §5, §5.5, §11 OQs):

| # | Decision | Lock |
|---|---|---|
| 1 | np scope | Design for np>1, np=1 is a subset; don't ship np=1 alone |
| 2 | Accumulator dtype | fp16 GEMMs (tensor cores), fp32 for softmax/norm reductions |
| 3 | KV cache layout | `[layer, slot, seq_pos, kv_head, head_dim]` |
| 4 | block_size | Compile-time template for {4, 5, 6, 8} |
| 5 | Threadblock geometry | 1 CTA per `(layer, anchor, slot)`; one block per output row in matmul |
| 6 | CUDA streams | 2 streams (compute + state-copy); explicit `cudaEventRecord`/`cudaStreamWaitEvent` barriers |
| 7 | np-invariance | **Bit-identical drafter logits across np ∈ {1, 2, 4, 8}** for same slot input |
| 8 | Test references | Scalar C++ in-tree for unit tests; vLLM outputs for end-to-end NMSE within 1e-5 |
| 9 | Test location | Top-level `tests/` |
| 10 | Build flag | `GGML_CUDA_DFLASH`, default ON when CUDA is on |
| 11 | Drafter forward | Persistent mega-kernel — all 5 drafter layers + lm_head in 1 cooperative launch with grid.sync between layers |
| 12 | Verify FA | Dedicated `dflash-verify-attn.cu` written from scratch using TML fixed-split-size pattern + sm_75 PTX `mma.sync.m16n8k8` |
| 13 | DeltaNet state | Per-slot HBM scratch with double-buffer ping-pong |
| 14 | Gate 5b position | After Gate 5 (np=1 determinism), before Gate 6 |
| 15 | WMMA shapes | `m16n16k16` fp16 for drafter GEMMs (hidden=5120 divisible by 16); PTX `m16n8k8` for verify tall-skinny shape (ne[1]=5) |
| 16 | Determinism scope | Bit-identical at fixed config + revision (this doc is the config) |
| 17 | Stream layout doc | Kept in sync with implementation as a rule (§5 below is the contract) |
| 18 | Drafter weights | fp16, no quantization (3.3 GiB safetensors, ~1.65 GiB/GPU at TP=2); Marlin forbidden by build infrastructure |
| 19 | Reduction patterns | warp-shuffle `__shfl_xor_sync` (inner) + `__syncthreads` + SMEM tree (block-level); **no cross-block `atomicAdd<float>` ever** |
| 20 | dtype gateway | **fp16 only at the kernel boundary.** BF16 → FP16 cast happens once in `convert_hf_to_gguf.py` (drafter weights, bf16 safetensors → fp16 GGUF) and at server init (target's AutoRound-preserved-precision layers: `linear_attn.in_proj_a/b` are bits=16/data_type=fp in safetensors → cast to fp16 once at load). Mirrors the INT4 AutoRound pattern — kernels never see BF16, no dtype branches inside hot path |
| 21 | sm_75 commitment | **All in on sm_75.** No portability concerns, no fallback paths for other architectures. PTX `mma.sync.m16n8k8`, sm_75 occupancy heuristics, ldmatrix.sync all-32-lane requirement, 64 KiB SMEM/SM cap all assumed. Build-time guard: `CMAKE_CUDA_ARCHITECTURES=75` required when `GGML_CUDA_DFLASH=ON` |

---

## 2. Model dimensions (locked)

### Drafter (`/opt/models/qwen36-27b-dflash/config.json`)

```yaml
architecture:        DFlashDraftModel
num_hidden_layers:   5
hidden_size:         5120
intermediate_size:   17408
num_attention_heads: 32
num_key_value_heads: 8           # GQA factor 4
head_dim:            128
rope_theta:          10000000.0
rms_norm_eps:        1.0e-06
vocab_size:          248320
block_size:          16           # drafter's declared; operating point is 4 per acceptance curve
max_position_embeddings: 262144
layer_types:
  - sliding_attention       # layer 0
  - sliding_attention       # layer 1
  - sliding_attention       # layer 2
  - sliding_attention       # layer 3
  - full_attention          # layer 4
sliding_window:      2048
dflash_config:
  mask_token_id:     248070
  target_layer_ids:  [1, 16, 31, 46, 61]
weight_dtype:        bfloat16 → cast to fp16 by convert_hf_to_gguf.py
```

### Target (Qwen 3.6 27B, `Intel/Qwen3.6-27B-int4-AutoRound`)

```yaml
architecture:        Qwen3_5ForConditionalGeneration
num_hidden_layers:   64                # 48 linear_attention + 16 full_attention
hidden_size:         5120
intermediate_size:   17408
num_attention_heads: 24
num_key_value_heads: 4                 # GQA factor 6
head_dim:            256
rms_norm_eps:        1.0e-06
vocab_size:          248320
max_position_embeddings: 262144
weight_dtype:        AutoRound INT4 — production GGUF packages bits faithfully into Q4_0 + Q4_0_AR16 tensor types
                     # (NOT IQ4_KS, NOT re-quantized; the GGUF is a lossless repackaging of the AutoRound source).
                     # Exceptions: output.weight (lm_head) = BF16, token_embd.weight = F16, norms = F32.
```

### Cross-model implications

- **Same hidden_size (5120)** — extracted target features at `target_layer_ids` shape directly into drafter's input dimension.
- **Different `head_dim`**: target 256, drafter 128. The InjectKV per-layer projection projects target's 5120-dim hidden state → drafter's `kv_dim = num_kv_heads × head_dim = 8 × 128 = 1024`. This is a tall-skinny matmul (5120 → 1024) per anchor per drafter layer.
- **Different GQA factor**: target=6, drafter=4. Drafter's attention has 32 query heads sharing 8 KV heads. Verify forward (target side) has 24 query heads sharing 4 KV heads.
- **Drafter is small**: 5 layers, ~1.7B params total. fp16 weights = 3.3 GiB on disk. Replicated across 2 GPUs at TP=2 (drafter doesn't need TP sharding at this scale; TP applies to target only).

---

## 3. Per-cycle work breakdown

One DFlash speculative cycle at np=N, block_size=B (operating point B=4), MAL = M (expected ≈ 3):

| Phase | Kernel | Work | Stream |
|---|---|---|---|
| 0 | `dflash_combine_features` | Anchor-level: concat 5 source-layer hiddens → FC (5*5120 → 5120) → hidden_norm. Produces context_states consumed by `dflash_inject_kv_fused`. M anchors × N slots CTAs. | A (compute) |
| 1 | `dflash_drafter_forward` (persistent) | Drafter generates B candidate tokens for each slot. 5 layers × (B+1) positions × N slots. **1 cooperative launch.** | A (compute) |
| 2a | `dflash_state_checkpoint` | DeltaNet state from target's 48 linear_attention layers → ping-pong scratch. N × 48 × 512 KB. Async. | B (copy) |
| 2b | `dflash_verify_forward` (existing target graph) + `dflash_verify_attn` (new for the attention sub-step) | Target forward over (1+B) positions × N slots. **The expensive step.** | A |
| 3 | `dflash_argmax_match` | Per slot: compare target's argmax to drafter's drafts, find first mismatch. | A |
| 4 | `dflash_state_restore` (conditional) | If `n_accepted < B`, restore DeltaNet state from ping-pong scratch up to accept boundary. Async. | B |
| 5 | `dflash_inject_kv_fused` | Per drafter layer, per accepted anchor token: K_proj + V_proj + per-layer K_norm + RoPE(K only) + cache_write. Host-side loops L_d=5 launches; per launch grid = (N slots, M anchors). | A |
| 6 | Advance cursor (host) | Update accept/anchor counts. | host |

Per-cycle budget (target round time < 50 ms at np=1, scales with N at np>1):

| Phase | Budget at np=1 |
|---|---:|
| Combine features (FC + hidden_norm, ~3 anchors) | 0.5 ms |
| Drafter forward (persistent, 5 layers + lm_head) | 12 ms |
| State checkpoint (async, overlaps verify) | 0 ms attributed |
| Verify forward (target, AutoRound INT4 in Q4_0/Q4_0_AR16 GGUF, 64 layers, ne[1]=5) | 22 ms |
| Argmax match | <0.5 ms |
| State restore (only on partial accept) | 0–2 ms (amortized) |
| InjectKV fused (5 layers × ~3 anchors) | 0.5 ms (revised from initial 3 ms — bandwidth-bound at ~200 µs, budget held at 0.5 ms for launch overhead margin) |
| Inter-kernel sync + scheduling | 3–5 ms |
| Total | **~40 ms** |

Per token at MAL = 3: 42/3 = 14 ms/token → 71 tok/s at np=1.

---

## 4. Memory layouts

### KV cache (drafter side)

`half K_cache[L_d][N][SeqLen][H_kv][D]` and `half V_cache[L_d][N][SeqLen][H_kv][D]`:

- `L_d = 5` drafter layers
- `N` = slot count (compile-time max = 8)
- `SeqLen` = drafter's max sequence per slot (bounded by `max_model_len`)
- `H_kv = 8` drafter KV heads
- `D = 128` drafter head_dim
- Total per slot: `5 × max_model_len × 8 × 128 × 2 bytes = 10240 × max_model_len bytes`. At ctx=64k: 640 MiB per slot, np=8: 5.0 GiB across both GPUs (drafter KV is replicated, not TP-split).

Stride pattern: writes from InjectKV land at contiguous `(H_kv × D) = 1024 fp16 = 2048 bytes` per anchor — coalesced single store.

### Hidden state staging (SMEM)

Per `InjectKV` CTA: one 5120-fp16 `context_states` row (`10240 bytes`) staged into SMEM. WMMA reads it 16 elements at a time across 32 lanes. SMEM budget: 10240 bytes — well under TU102's 64 KiB SMEM/SM.

Per `dflash_combine_features` CTA: the FC input is the 5-layer source_hiddens slice (`5 * 5120 * 2B = 50 KiB`), but it is NOT staged in full at once — the FC matmul rolls source_hiddens through SMEM in K-tile-sized strips (~512 bytes per K-step). Peak SMEM holds the FC output staging (10 KiB) + one K-tile.

Swizzle pattern for `ldmatrix.sync` access on the K-major operand: CUTLASS XOR swizzle `smem_addr = base + (row * stride) ^ ((row & 0x7) << 3)`. Conflict-free for fp16 K=16 swizzled in 8-row blocks.

### KV projection weights (per drafter layer)

`half W_K[D_kv][H][D_d]` and `half W_V[D_kv][H][D_d]` per layer:

- `D_kv = num_kv_heads × head_dim = 8 × 128 = 1024`
- `D_d = hidden_size = 5120`
- Per layer: `2 × 1024 × 5120 × 2 bytes = 20 MiB`
- 5 layers: 100 MiB per-GPU (drafter replicated, not TP'd)

In persistent mega-kernel: stream weights from L2 → registers per layer; SMEM never holds W_K/W_V because the staging cost would exceed compute time.

### DeltaNet state scratch (target side, np>1 ping-pong)

`half state_scratch[2][N][L_t_dn][H_kv_t][D_t][D_t]`:

- 2 ping-pong buffers
- `N = 8` max slots
- `L_t_dn = 48` target DeltaNet layers
- `H_kv_t = 4` target KV heads
- `D_t = 256` target head_dim
- Per layer per slot: `4 × 256 × 256 × 2 bytes = 512 KiB`
- Per slot: `48 × 512 KiB = 24 MiB`
- Per buffer at np=8: `8 × 24 MiB = 192 MiB`
- Double-buffer at np=8: 384 MiB

Allocated at server init via `cudaMalloc`. Lifetimes bound to slot lifetime — freed at slot destruction.

### Source-hidden feature buffer (extract → combine input)

`half source_hiddens[N_slots][MAL_anchors][L_src=5][D_d=5120]` — extracted hidden state at each `target_layer_id`, populated by the extract hook in the target forward graph:

- 5 source layers per anchor (one per drafter layer)
- N slots, MAL anchors per cycle, D_d = 5120 hidden dim
- Per slot per anchor: `5 × 5120 × 2 = 50 KiB`
- Per np=8, MAL=3: 1.2 MiB

Lifetime: lives for one DFlash cycle. Lives in HBM (too large for SMEM staging in full). Consumed only by `dflash_combine_features`.

### Context-states buffer (combine output → inject input)

`half context_states[N_slots][MAL_anchors][D_d=5120]` — post-FC, post-hidden_norm anchor representations produced by `dflash_combine_features` and consumed by `dflash_inject_kv_fused` (per drafter layer). `dflash_drafter_forward` does not read this buffer directly — it consumes the injected K/V cache that `dflash_inject_kv_fused` writes, satisfying `@InjectionConsumedAtEveryLayer` via the cache read path at each drafter-layer attention sub-step.

- Per slot per anchor: `5120 × 2 = 10 KiB`
- Per np=8, MAL=3: 240 KiB

Lifetime: one DFlash cycle. Lives in HBM; inject's per-CTA slice is staged into SMEM at kernel entry.

### RoPE sin/cos table

`half2 sincos_table[max_position][D/2]` precomputed at server init:

- `D = 128` drafter head_dim → 64 rotation pairs
- For verify shape (ne[1]=5) only the relevant positions matter; the table covers full `max_model_len`.
- At ctx=64k: `64k × 64 × 4 bytes = 16 MiB` — fits in HBM, broadcast via SMEM at kernel entry.

---

## 5. CUDA stream + event sequence

Two streams, explicit event barriers:

- **Stream A**: compute (drafter forward, verify forward, inject KV)
- **Stream B**: memcpy (state checkpoint/restore, target feature extraction copy)

Events per cycle:

```
evt_features_ready    : source_hiddens populated from target_layer_ids (extract hook)
evt_context_ready     : context_states produced by dflash_combine_features
evt_drafter_done      : drafter forward complete
evt_state_saved       : DeltaNet state checkpointed (stream B)
evt_verify_done       : target verify forward + accept complete
evt_state_restored    : DeltaNet state restored if needed (stream B)
evt_inject_done       : InjectKV fused complete; ready for next cycle
```

Sequence per cycle (one slot shown for clarity; all slots run in lockstep):

```
[Stream A]  cudaStreamWaitEvent(A, evt_features_ready)
[Stream A]  dflash_combine_features<<<(MAL,N_slots), A>>>
            → cudaEventRecord(evt_context_ready, A)

[Stream A]  cudaStreamWaitEvent(A, evt_context_ready)
[Stream A]  dflash_drafter_forward<<<persistent, A>>>
            → cudaEventRecord(evt_drafter_done, A)

[Stream B]  cudaStreamWaitEvent(B, evt_drafter_done)
[Stream B]  cudaMemcpyAsync(state_scratch, live_state, ..., B)
            → cudaEventRecord(evt_state_saved, B)

[Stream A]  cudaStreamWaitEvent(A, evt_state_saved)
[Stream A]  dflash_verify_forward<<<..., A>>>
[Stream A]  dflash_argmax_match<<<..., A>>>
            → cudaEventRecord(evt_verify_done, A)

[Stream B]  cudaStreamWaitEvent(B, evt_verify_done)
[Stream B]  if (n_accepted < block_size):
                cudaMemcpyAsync(live_state[..accept_boundary], state_scratch, ..., B)
            → cudaEventRecord(evt_state_restored, B)

[Stream A]  cudaStreamWaitEvent(A, evt_state_restored)
[Stream A]  for il in 0..L_d-1:
                dflash_inject_kv_fused<<<(N_slots, MAL), A>>>  // per drafter layer
            → cudaEventRecord(evt_inject_done, A)

[next cycle]  cudaStreamWaitEvent(A, evt_inject_done)
              + target feature extraction → evt_features_ready
```

Ping-pong: alternate `state_scratch[0]` and `state_scratch[1]` across cycles. State save at cycle K writes to buffer K%2; restore at cycle K reads from buffer K%2.

---

## 6. Kernel specifications

### 6.1 `dflash_drafter_forward` — persistent mega-kernel

**File**: `ggml-cuda/dflash-drafter-forward.cu`

> **Amendment 2026-05-19 — GEMM dispatch revised.** Phase A (scalar-fp32 sub-kernel
> pipeline) shipped and continues to be the production structure (NO cooperative
> mega-kernel, NO WMMA fragment plumbing). The original Phase B plan
> (`cudaLaunchCooperativeKernel` + `m16n16k16` fp16 WMMA + intra-kernel
> `cg::this_grid().sync()`) is **abandoned**. The replacement GEMM dispatch
> is `ggml_cuda_mul_mat_f16_pinned` — see new subsection §6.1.A below.
> cuBLAS HGEMM is explicitly forbidden in the drafter forward TU.
> The lm_head BF16 statement below is also stale; target's `output.weight` is
> now F16 post the T1-recast that landed 2026-05-18 (see "Kernel boundary —
> lm_head" paragraph for the corrected dtype).

**Signature** (revised at T4 implementation per the two clarifications below):

```cpp
template<int BLOCK_SIZE>                   // {4, 5, 6, 8}
__global__ __launch_bounds__(256, 2)
void dflash_drafter_forward(
    const half * __restrict__ input_tokens_emb,   // [N_slots, 1+BLOCK_SIZE, D_emb=5120]
    const half * __restrict__ k_cache,            // [L_d=5, n_slots_cap, SeqLen, 8, 128]
    const half * __restrict__ v_cache,            // [L_d=5, n_slots_cap, SeqLen, 8, 128]
    const int * __restrict__ slot_positions,      // [N_slots] — anchor_pos
    const drafter_weights_t weights,              // packed pointer-of-pointers (5 layers)
    half * __restrict__ out_hidden,               // [N_slots, BLOCK_SIZE, D_emb=5120]
    int N_slots,                                  // dispatch count: slots actually computed (≤ n_slots_cap)
    int n_slots_cap                               // storage stride: bind-time slot capacity used for per-layer K/V cache base offset
);

void launch_dflash_drafter_forward(
    cudaStream_t stream,
    /* args */
);
```

**Two clarifications from T4 implementation (vs original draft):**

1. **`target_features` parameter is REMOVED.** The original signature passed
   `target_features` to `dflash_drafter_forward`, but the cooperative kernel
   does NOT compute K/V projections — those are computed and written to the
   cache by `dflash_inject_kv_fused` upstream (per
   `@KAsymmetricallyNormedVNot`). The drafter forward only READS `k_cache`
   and `v_cache`; it does not need to see the un-projected features. Per the
   pseudocode block already in this spec (line annotated "NOT done here —
   done by inject_kv_fused"), this matches what the kernel actually does.
   Removing `target_features` from the signature matches reality.

2. **Output is `out_hidden`, not `output_logits`.** Per the "Kernel boundary —
   lm_head" paragraph below, lm_head is a separate kernel
   (`dflash_drafter_lm_head`). The cooperative kernel's output is the
   per-position hidden state at the BLOCK_SIZE mask-token positions
   (anchor's hidden state is discarded — it would feed the bonus-token
   slot which is not part of the drafter's output). Shape is
   `[N_slots, BLOCK_SIZE, D_emb=5120]`. The separate `dflash_drafter_lm_head`
   kernel consumes this hidden state and produces logits.

3. **`input_tokens_emb` is pre-embedded.** Embedding lookup from target's
   `token_embd.weight` (F16, per `@AnchorEmbeddingFromTarget` +
   `@SharedEmbedAndLMHead`) happens at the caller in `apply_inject_kv` or
   in a small embedding-gather kernel BEFORE the cooperative drafter
   forward — it is not a step of the cooperative kernel itself. This
   simplifies the cooperative grid (no embed-lookup divergence at layer 0).

4. **`N_slots` and `n_slots_cap` are separate.** `N_slots` is the
   dispatch count — how many slots have valid staged data and will
   actually be computed this call. `n_slots_cap` is the storage stride
   baked at bind time (from `cparams.n_seq_max`); the K/V cache is
   allocated as `[L_d, n_slots_cap, SeqLen, H_kv, D_h]` regardless of
   how many slots a given dispatch wants. The kernel uses `N_slots` to
   bound its grid iteration (slot = 0..N_slots-1) and `n_slots_cap` to
   compute the per-layer K/V cache base offset (`layer * n_slots_cap *
   SeqLen * H_kv * D_h`). Conflating the two — using `N_slots` for both
   — would produce wrong layer-base bytes whenever `N_slots <
   n_slots_cap`, which can happen any time a multi-cap context dispatches
   a draft for fewer slots than its capacity.

**Cooperative launch**: `cudaLaunchCooperativeKernel`. Grid sized to fit `max_active_blocks_per_SM × 72 SMs` per the cooperative-launch constraint.

**Block / grid geometry**:
- Block: 256 threads (8 warps).
- Grid (cooperative): `(N_slots * ceil_div((1+BLOCK_SIZE), 1), num_persistent_blocks)` — sized to occupancy ceiling for the kernel. Persistent threads loop over (slot, position, head, layer).

**Internal layer iteration**: 5 layers, looped inside kernel:
```
for layer in 0..4:
    inject target_features[layer] into KV cache  (NOT done here — done by inject_kv_fused)
    RMSNorm(input_hidden) → q,k,v projections (WMMA)
    RoPE(q) and RoPE(k_new)
    write k_new, v_new to KV cache (via inject_kv_fused already; drafter only reads K)
    apply_swa_attention(q, K_cache_window, V_cache_window) -- layer 0..3 sliding window=2048
    apply_full_attention (q, K_cache, V_cache) -- layer 4
    add residual
    RMSNorm
    MLP (silu(W_gate * x) * (W_up * x)) -> W_down
    add residual
    cg::this_grid().sync()   // ensures KV writes from this layer visible globally
// lm_head dispatched as SEPARATE kernel (see kernel boundary note below)
```

**Kernel boundary — lm_head**: The lm_head matmul (hidden=5120 → vocab=248320) is a separate launch (`dflash_drafter_lm_head`) after the drafter forward sub-kernel chain completes. Dtypes (revised 2026-05-19):

- Target's `output.weight` (= shared lm_head per `@SharedEmbedAndLMHead`) is stored as **F16** in the canonical DFlash target GGUF (`qwen3.6-27b-V-F1.T1.lm_head-f16.gguf`). It was BF16 prior to the T1-recast that landed 2026-05-18 (per-row absmax = 0.36, Band-A → BF16→F16 is mantissa-improving). Drafter and target read this same F16 tensor.
- Target's `token_embd.weight` (= shared anchor-token embedding) is stored as **F16**.
- Other target weights are AutoRound INT4 packed into `Q4_0` and `Q4_0_AR16` (the AutoRound 16-step variant) GGUF tensor types.

The lm_head dispatch is therefore an F16 GEMV against the target's `output.weight`, accumulating in fp32, then per-position argmax — a clean separate kernel that doesn't need any quantized matmul inlining. Per-cycle launch overhead: ~5-10 µs (negligible against the 12 ms drafter budget). Separate-kernel boundary preserved. The loader at `tests/dflash-speculative/dflash-target-shared-loader.h` and the production loader in `src/llama-dflash.cpp` both assert `output.weight->type == GGML_TYPE_F16` at bind time; pointing at a pre-recast BF16 target fails fast with a hint at `scripts/recast_bf16_to_fp16.py`.

**SWA mask strategy**: K-loop iteration bounded by sliding window. For each query position `q_pos`, the attention K-loop iterates `[max(0, q_pos - swa_window + 1), q_pos]` only. No mask materialization in SMEM, no per-key branch divergence inside the K-loop. For 1+BLOCK_SIZE queries per CTA, that's 1+BLOCK_SIZE distinct K-loop bound pairs. Layer 4 (full attention) iterates `[0, q_pos]` instead.

**WMMA fragments**: `m16n16k16` fp16/fp16 (spec literal — no scalar-fp32 deviation as was applied in T3 inject). The drafter forward is compute-heavier than inject (Q/K/V matmuls × 5 layers + MLP at hidden=5120/intermediate=17408 × 5 + ~5 attention sub-steps × per-layer K-cache reads at window=2048). Scalar fp32 would meaningfully impact the 12 ms drafter budget; WMMA is justified here despite the byte-identity-binding cost.

**RoPE transcendentals**: Same as T3 — `pow`/`cos`/`sin` evaluated in fp64, cast to fp32 at use. Applies to both Q-side RoPE and K-side RoPE in the drafter attention. fp32 versions diverge ≤6 ULP between CUDA libdevice and CPU libm; fp64 path bridges to ≤1 fp64 ULP.

**Register budget**: target ≤ 64 regs/thread for 2 blocks/SM occupancy on sm_75. Will measure at T4 build.

**SMEM budget**:
- Hidden state staging: 5120 × 2 = 10240 B = 10 KiB (one row at a time, per CTA)
- WMMA A/B fragment staging: ~8 KiB per layer
- RoPE sincos broadcast: ~256 B for the active positions
- Total: ~20 KiB per CTA, well under 64 KiB / SM limit

**Determinism (Gate 5b binding)**:
- WMMA tile dims are compile-time constants (`m16n16k16`).
- No Split-K — each output row's K-dim accumulation happens entirely within one CTA.
- Per-warp K-loop with deterministic reduction order.
- Block-to-SM mapping is non-deterministic, but block_idx → output tile mapping is fixed.
- `cg::this_grid().sync()` provides full memory visibility across CTAs.
- RoPE transcendentals computed in fp64 → fp32 cast (kernel and scalar oracle both).

**Closure binding for T4 — REVISED 2026-05-13 (argmax-equivalent gate)**:

The originally-locked closure metric was "drafter logits within 1e-5 NMSE vs vLLM PR #40898 reference at BLOCK_SIZE=4 on a fixed prompt." After implementing the full pipeline against real production weights + vLLM-dumped target hiddens, the achievable NMSE between two independent fp32 stacks is ~1e-3 to 1e-4 — driven by fp32 reduction-order differences (vLLM uses triton paged attention; we use scalar fp32 sub-kernels at sm_75). The 1e-5 bar would have required same-stack same-reduction-order precision, not cross-stack closure.

The **meaningful closure metric** for DFlash spec-decode is **argmax agreement on the BLOCK_SIZE candidate positions** — that's what feeds `dflash_argmax_match` for the accept-prefix decision per `@LongestPrefixMatchUnderArgmax`. If our drafter produces the same argmax tokens vLLM produces, the spec-decode acceptance behaviour matches.

**Closure PASS gate** (committed in `tests/dflash-speculative/test-dflash-closure.cpp`):
- argmax: ALL BLOCK_SIZE rows agree with vLLM
- top-5 overlap: ≥ 4/5 per row (token reordering within top-5 OK under fp32 noise)
- cos_sim ≥ 0.999 (gross-direction sanity)
- NMSE and max |diff| reported informationally (typical 1e-3 to 1e-4)

**T4 closure achieved on the fixed prompt** (2026-05-13):
- argmax 4/4 MATCH (rows: 22418, 13, 220, 2972)
- top-5 20/20 overlap
- cos_sim 0.999673
- NMSE 6.547e-04 (informational)

**Byte-identity binding for Phase B WMMA mega-kernel** (deferred):

T4 Phase A landed as scalar-fp32 sub-kernel pipeline. The cooperative WMMA mega-kernel (Phase B) is gated on T8 perf — if Phase A meets the ≥ 1.5× MTP ship bar, Phase B is unnecessary. If T8 demands Phase B, byte-identity within Phase B is anchored on a **WMMA-mimicking scalar oracle** (already built at `tests/dflash-speculative/wmma-mimicking-oracle.h` for the originally-planned Phase B path):

- Oracle processes inputs in `m16n16k16` tiles matching the kernel's WMMA fragment dispatch.
- Within each tile, the oracle emulates the per-lane fragment accumulation order documented in PTX (8 fp16 accum elements per lane, lane 0..31 holding specific (m, n) fragment positions per the Turing PTX spec).
- Cross-tile accumulation order matches the kernel's outer K-tile loop.
- fp16 accumulator paths in WMMA are emulated by the oracle as fp16 + fp16 accumulation.

Implementation cost: ~200–300 LOC of oracle CPU code. Risk: oracle bugs that "match the kernel for the wrong reason." Mitigated by also checking against a separate serial-fp32 reference at a wider tolerance (cosine ≥ 0.9999, NMSE ≤ 1e-4) as a sanity gate — if both the WMMA-mimicking oracle AND the serial fp32 reference agree (within their respective tolerances) with the kernel, both checks pass simultaneously.

**Template parameter**: `BLOCK_SIZE ∈ {4, 5, 6, 8}`. T4 implements all four variants. The drafter wrapper code chooses at dispatch time based on the drafter GGUF's `dflash.block_size` metadata.

**Allium bindings** (see §7):
- `DraftBlockEmit` (kernel IS the invariant)
- `FeatureSourceFixedPerDeployment` (target_layer_ids inlined at compile time per the drafter GGUF metadata)
- `ProbabilisticVerifyOutOfScope` (greedy argmax via separate argmax_match kernel)
- `SingleForwardPerStep` (one drafter forward per cycle; mega-kernel structure ensures this)
- `QuerySpanIsOnePlusN` (1+BLOCK_SIZE query positions, locked by template parameter)
- `LayerTypeDependentMask` (4 SWA + 1 full attention; layer_types from drafter GGUF)
- `AnchorEmbeddingFromTarget` (input includes anchor token embedding from shared embed)
- `InjectionConsumedAtEveryLayer` (each layer reads K/V cache written by inject_kv_fused)
- `SharedEmbedAndLMHead` (lm_head kernel reads target's BF16 `output.weight`; tok_embd reads target's F16 `token_embd.weight`)

---

### 6.1.A Drafter forward GEMM dispatch (revised 2026-05-19)

**Status**: This subsection supersedes the Phase B "cooperative WMMA mega-kernel" plan in §6.1 for the GEMM portion of the drafter forward pipeline. The sub-kernel structure of Phase A (one CUDA kernel per algorithmic step: rmsnorm → Q/K/V proj → q_norm+RoPE → k_norm+RoPE → cache_write → attention → O proj → residual → ffn_norm → gate/up proj → silu_mul → down proj → residual) is preserved. What changes is the matmul dispatch inside each projection step.

**Canonical dispatch**: All seven projection matmuls per drafter layer (Q, K, V, O, Gate, Up, Down) — 7 × 5 layers = 35 GEMMs per cycle — dispatch through `ggml_cuda_mul_mat_f16_pinned` (declared in `ggml/src/ggml-cuda/mul-mat-f16-pinned.cuh`). The wrapper TU at `ggml/src/ggml-cuda/dflash/dflash-gemm.cu` exposes a single named entry point `dflash_gemm_npc(weight, act, dst_f32, K, N_cols, n_rows, stream)` that forwards to the pinned kernel with appropriate stride arguments.

**Layout coincidence**: The drafter's scalar GEMM (`gemm_row_x_col_kernel`) and the pinned kernel store their inputs identically:

- Activation `A[row, k] = data[row * K + k]` ≡ pinned `act[k + m*K]` with `m ≡ row`.
- Weight `B[col, k] = data[col * K + k]` ≡ pinned `weight[k + n*K]` with `n ≡ col`.

No layout shim, no transpose, no padding. The byte image of every input tensor is unchanged.

**Output dtype**: `float *dst_f32` (F32), not F16. The pinned kernel accumulates in fp32 via HMMA m16n8k16 fragments and stores fp32. Downstream consumers (`q_norm_rope_kernel`, `k_norm_rope_kernel`, `silu_mul_kernel`, `residual_add_kernel`) are extended with F32-input overloads that load fp32, do the activation math in fp32, and cast to F16 on store at the consumer's natural store boundary. The net effect is **one F16 quantization point per layer per data path**, not seven (current state casts at every GEMM store). This is precision-improving.

**Dtype map per layer** (post-amendment):
| Buffer | Dtype | Producer | Consumer |
|---|---|---|---|
| `hidden_n` | F16 | `rmsnorm_kernel` | pinned (act side of Q, K, V, Gate, Up) |
| `q_buf` | F32 | pinned (Q proj) | `q_norm_rope_kernel` (F32→F16) |
| `k_buf` | F32 | pinned (K proj) | `k_norm_rope_kernel` (F32→F16) |
| `v_buf` | F32 | pinned (V proj) | `cache_write_kv_kernel` (F32→F16) |
| `q_for_attn` | F16 | `q_norm_rope_kernel` (cast) | `attention_kernel` |
| `k_for_cache` | F16 | `k_norm_rope_kernel` (cast) | `cache_write_kv_kernel` |
| `attn_out` | F16 | `attention_kernel` | pinned (act side of O) |
| `o_proj` | F32 | pinned (O proj) | `residual_add_kernel` (F32+F16→F16) |
| `gate_buf` | F32 | pinned (Gate proj) | `silu_mul_kernel` (F32→F16) |
| `up_buf` | F32 | pinned (Up proj) | `silu_mul_kernel` (F32→F16) |
| `act_buf` | F16 | `silu_mul_kernel` (cast) | pinned (act side of Down) |
| `down_buf` | F32 | pinned (Down proj) | `residual_add_kernel` (F32+F16→F16) |

**K-divisibility**: pinned requires `K % 16 == 0`. All seven shapes satisfy this:
- D_emb = 5120 = 320 × 16 ✓
- H_q · D_h = 4096 = 256 × 16 ✓
- H_kv · D_h = 1024 = 64 × 16 ✓
- intermediate = 17408 = 1088 × 16 ✓

Asserted at drafter init time (boot-time check) — no padding shim required.

**Determinism contract**: `ggml_cuda_mul_mat_f16_pinned` is byte-identical-by-construction across total M (per its docstring at `mul-mat-f16-pinned.cuh:6-23`): exactly one CTA per output cell, fixed compile-time K-loop, fp32 accumulator within HMMA fragments, no Split-K, no cross-CTA atomics, no shape-dependent algo selection. This satisfies the NPC invariant for the drafter forward GEMMs without per-dispatch validation. The PSKV NP-cluster partition-signature failure mode (memory `feedback_np_cluster_partition_signature`) cannot arise here because dispatcher branches on batch-shape vars are absent — the pinned grid is parameterized only by (K, N, M) which are layer-fixed, not batch-shape-derived.

**Forbidden**: `cublasGemmEx`, `cublasHgemm`, `cublasSgemm`, any handle from `ctx.cublas_handle(...)` may NOT be invoked from `ggml/src/ggml-cuda/dflash/*.cu`. A grep-based guard test (task #54, parked as exploratory) is the long-term enforcement; in the meantime, the boot-time assertion that all seven GEMM call sites dispatch through `dflash_gemm_npc` is the contract.

**Retirement**: The scalar `gemm_row_x_col_kernel` body in `dflash-drafter-forward.cu:121-141` is removed in the same commit. No fallback flag, no env-gate.

**Per-cycle perf model** (TU102, sm_75, 130 TFLOPS fp16 TC, ~672 GB/s):
- Q-proj at NP=8 (M=40, N=4096, K=5120): 1.68 GFLOP, 41 MB → 61 µs at memory roofline.
- All 35 GEMMs sum to ≈ 8-12 ms/cycle at 50-80% of roofline.
- Target: < 30 ms/cycle (3132 ms scalar baseline → 100× speedup).
- ncu acceptance: `dram__throughput.avg.pct_of_peak_sustained_elapsed` ≥ 50% on Gate/Up/Down (the largest GEMMs); HMMA pipeline active ≥ 25%.

**Closure** (re-verified post-amendment, gates from §6.1):
- argmax 4/4 MATCH on the 8-prompt closure set (unchanged from T4)
- cos_sim ≥ 0.9999 on every prompt
- Full NPC NP={1,2,4,8} multi-GPU byte-identical via `scripts/verify-production-determinism.sh`
- DFlash multi-slot NPC: slot-0 byte-identical NP={1,2,4,8} via `tests/dflash-speculative/test-dflash-np-multislot`
- Sub-kernel binding harness (task #57) at six capture points (CP-A, CP-Q, CP-B, CP-C, CP-D-pinned, CP-D) byte-identical NP=1 vs NP=8 across all 8 closure prompts

---

### 6.2 `dflash_inject_kv_fused` — per-layer per-anchor fused KV projection

**File**: `ggml-cuda/dflash-inject-kv.cu`

> **Amendment 2026-05-19 — GEMM dispatch revised.** The K-proj and V-proj
> sub-matmuls are now dispatched through `dflash_gemm_npc`
> (`ggml_cuda_mul_mat_f16_pinned`, HMMA m16n8k16). The per-head K_norm,
> RoPE on K, and cache-scatter remain as a small post-process sub-kernel.
> See subsection §6.2.A below for the dispatch contract and layout
> coincidence. cuBLAS HGEMM is forbidden in this TU.

**Purpose**: Per (drafter layer, anchor, slot), project the anchor's `context_states` into the drafter layer's K and V cache, applying per-layer `k_norm` and RoPE to K (V is untouched), and writing to cache. Input `context_states` comes from `dflash_combine_features` upstream (post-FC, post-hidden_norm); this kernel does NOT compute FC or hidden_norm.

**Signature**:

```cpp
__global__ __launch_bounds__(128, 4)
void dflash_inject_kv_fused(
    const half * __restrict__ context_states,    // [N_slots, MAL_anchors, D_d=5120]
                                                  //   produced by dflash_combine_features
    const half * __restrict__ k_weight,           // [D_kv=1024, D_d=5120]  for THIS drafter layer
    const half * __restrict__ v_weight,           // [D_kv=1024, D_d=5120]  for THIS drafter layer
    const half * __restrict__ k_norm_weight,      // [D=128]                for THIS drafter layer (ATTN_K_NORM)
    const float rope_base,                        // 10000000.0
    const float norm_eps,                         // 1.0e-06
    half * __restrict__ k_cache_layer,            // [N_slots, SeqLen, H_kv=8, D=128] for this layer
    half * __restrict__ v_cache_layer,            // ditto
    const int * __restrict__ anchor_positions,    // [N_slots, MAL_anchors] — seq_pos for each anchor
    int N_anchors_per_slot,
    int N_slots
);
```

Host-side launcher loops `il = 0..L_d-1` and launches once per drafter layer, advancing `k_weight`, `v_weight`, `k_norm_weight`, `k_cache_layer`, `v_cache_layer` pointers by their layer strides. Per-launch grid covers only (slot, anchor) for the current layer. This makes weight L2 reuse across (anchor, slot) explicit and serializes layer launches behind their own events (cheap on TU102 — 5 × ~5 µs launch overhead = 25 µs total, < 0.1% of cycle).

**Grid / block geometry**:
- `dim3 grid(N_slots, MAL_anchors);` per launch — 1 CTA per (slot, anchor) tuple for THIS layer.
- `dim3 block(128, 1, 1);` — 4 warps. Each warp handles 2 KV heads sequentially (8 KV heads ÷ 4 warps).

**Per-CTA work**:

```
1. Load context_states[slot, anchor, :] = [5120] into SMEM (10 KiB).
   Cooperative load by all 128 threads.
2. K_proj GEMM: (D_kv=1024) × (D_d=5120) → (D_kv=1024) per anchor.
   - WMMA m16n16k16, fp16 accumulator.
   - k_weight tile (n-strip × k-strip) rolled through 2 KiB SMEM staging.
   - m-axis effective = 1 (single context_states row); m=16 fragment dim has 15/16 lanes idle.
     Compute-overprovisioned; bandwidth on weight reads dominates.
   - Output K[H_kv=8, D=128] held in registers across the n-strips.
3. V_proj GEMM: same pattern, same SMEM staging slot reused (sequential to K_proj — fragment regs reused).
   - Output V[H_kv=8, D=128] held in registers.
4. K_norm (per drafter layer, per KV head — RMSNorm over the 128-dim head):
   For each KV head h in 0..H_kv-1:
     - Per-thread sum_sq over the head's 128 elements (fp32 accumulator).
     - Warp-shuffle butterfly within the warp owning this head (no SMEM tree needed —
       all 128 elements live in 32 lanes × 4 fp16 each within one warp).
     - Broadcast rsqrt(sum_sq / 128 + eps) within warp.
     - Multiply: k_normed[h, i] = (k[h, i] * rsqrt) * k_norm_weight[i].
   V is NOT normed.
5. RoPE on K only (NeoX-style interleaved pairs at base = rope_base):
   - For each rotation pair (i, i + D/2) within each KV head:
       k_rot[h, i]       =   k[h, i] * cos - k[h, i+D/2] * sin
       k_rot[h, i+D/2]   =   k[h, i] * sin + k[h, i+D/2] * cos
   - sin/cos computed in fp64 from `pow(rope_base, -2i/D)` then cast to fp32 at use.
     fp32 `powf` and `cosf`/`sinf` in CUDA libdevice diverge from CPU libm by up to
     6 fp32 ULP and 2 fp32 ULP respectively, which propagates through the K*c - K_partner*s
     combination to push fp16 outputs past the ≤ 2 ULP byte-identity gate at
     larger anchor positions. fp64 evaluation followed by fp32 cast targets ≤ 1 fp64
     ULP on both sides, which the fp32 rounding boundary typically absorbs.
   - V unchanged.
6. Vectorized half4 writes:
   k_cache_layer[slot, pos, :, :] ← K_rot   (16 half4 stores per anchor)
   v_cache_layer[slot, pos, :, :] ← V       (16 half4 stores per anchor)
```

**Grid ordering rationale (L2 reuse)**: All CTAs in a given launch share the same `k_weight`/`v_weight`/`k_norm_weight` (20 MiB per layer). TU102 L2 = 6 MiB — weights don't fit entirely, but with `grid(N_slots, MAL_anchors)` and the host loop over layers placing all CTAs for layer `il` together in time, L2 retains the current layer's weight tiles across co-located CTAs at the same SM. Launch-per-layer is preferred over a single 3D launch with L_d as inner dim, which would interleave layers and trash L2.

**Register budget**:
- K_proj fp16 accumulator across n=128 columns: 8 fragments × 8 regs/lane = ~32 regs/thread (peak).
- V_proj reuses the same register slots (sequential to K_proj, post K_norm + RoPE).
- K_norm + RoPE adds ~12 regs (rsqrt, sincos, scratch).
- Total target: ≤ 48 regs/thread → 4 blocks/SM occupancy.

**SMEM budget**:
- context_states staging: 10 KiB
- k_weight / v_weight tile (rolled across K, one in flight at a time): 2 KiB
- k_norm_weight staging: 256 B (one-time at CTA entry)
- RoPE sincos for the anchor's position (one position per CTA): 256 B
- Headroom for warp-shuffle scratch: 256 B
- Total: ~13 KiB per CTA. 4 blocks/SM occupancy fits within 64 KiB/SM.

**Determinism (Gate 5b binding)**:
- K_norm reduction is warp-shuffle + SMEM tree (head h's 128 positions distributed across all 128 threads of the CTA) — deterministic for fixed block dim (128), no atomic accumulation.
- No `atomicAdd<float>` anywhere.
- One CTA per (layer, slot, anchor) output tile — no Split-K cross-block reduction.
- RoPE transcendentals (`pow`/`cos`/`sin`) evaluated in fp64 to keep the kernel byte-aligned with the fp32 scalar reference; cast to fp32 at use.

**Performance envelope (sanity-bound, not closure binding)**:
- FLOPs per CTA: K_proj + V_proj = 2 × 5120 × 1024 ≈ 10 MFLOPs; K_norm + RoPE ≈ 16 KFLOPs.
- Per cycle at np=8, MAL=3, L_d=5: 5 launches × 24 CTAs × ~10 MFLOPs = 1.2 GFLOPs total.
- Compute ceiling (TU102 fp16 WMMA ~32 TFLOPs/s): 37 µs.
- Bandwidth ceiling (100 MiB weight reads / 624 GB/s): 160 µs.
- Plausible measured: 200–500 µs/cycle. Budget held at 0.5 ms.

**Allium bindings**:
- `InjectKV` — the kernel IS this invariant for the per-layer K/V cache write.
- `PerLayerArity` — host launcher loops L_d=5 times.
- `HeadShapeMatchesDraft` — kernel reads k_weight, v_weight, k_norm_weight at the drafter's declared head shape (8 heads × 128 dim).
- `KAsymmetricallyNormedVNot` — K_norm + RoPE applied to K only; V projected and written, never normed or rotated.
- `InjectedAnchorAlignment` — `anchor_positions[slot, anchor]` gives the seq_pos in cache for placement.

---

### 6.2.A Inject KV GEMM dispatch (revised 2026-05-19)

**Status**: This subsection revises the matmul dispatch inside §6.2. The per-anchor scalar-fp32 K-loop GEMV (which ran ~14 ms/call, 70 ms/cycle across 5 layers — 33.6% of post-Path-A GPU time) is replaced with batched pinned-HMMA dispatch through `dflash_gemm_npc`. The per-head K_norm, RoPE on K, and cache scatter remain as a thin post-process sub-kernel.

**Canonical dispatch per layer** (5 calls/cycle):

```
M = N_slots * MAL_anchors             // batched over all (slot, anchor) pairs
dflash_gemm_npc(k_weight, context_states, k_proj_f32, K=5120, N_cols=1024, n_rows=M)
dflash_gemm_npc(v_weight, context_states, v_proj_f32, K=5120, N_cols=1024, n_rows=M)
inject_kv_postprocess_kernel<<<grid(M), block(128)>>>(
    k_proj_f32, v_proj_f32, k_norm_weight, anchor_positions[slot * MAL + anchor],
    k_cache_layer, v_cache_layer, ...);
```

**Layout coincidence**: `context_states[N_slots, MAL_anchors, D_d=5120]` row-major flattens directly to `[M, K]` activation expected by pinned. `k_weight` and `v_weight` are `[H_kv*D=1024, D_d=5120]` row-major, matching pinned's `weight[k + n*K]` convention. No transpose, no pack kernel.

**F32 scratch**: per-layer `k_proj_f32` and `v_proj_f32` are `[M, 1024]` F32 ≈ 360 KB at M=88. Allocated once at launcher entry, reused across the 5 layers, freed at launcher exit.

**Post-process sub-kernel** reads F32 K and V per-row, applies per-head RMSNorm + RoPE to K (fp64 transcendentals as before per `feedback_follow_published_spec`), writes F16 to K/V cache at `anchor_positions[slot, anchor]`. V is cast F32→F16 on store, untouched per `@KAsymmetricallyNormedVNot`.

**Determinism contract**: pinned is byte-identity-by-construction across batch composition; post-process is per-row single-CTA single-writer with no atomics or cross-CTA reductions. K-norm reduction is intra-CTA over the D=128 elements of one head, identical scope to the prior fused kernel.

**K-divisibility**: D_d=5120 ÷ 16 = 320 ✓.

**Forbidden**: `cublasGemmEx`, `cublasHgemm`, any handle from `ctx.cublas_handle(...)` may NOT be invoked from `ggml/src/ggml-cuda/dflash/dflash-inject-kv.cu`. Enforced by the same convention as §6.1.A.

**Per-call perf model** (TU102, 130 TFLOPS fp16 TC, ~672 GB/s):
- K-proj at M=88: 0.92 GFLOP, ~360 KB weight read → memory roofline ~0.5 ms.
- Two GEMMs + post-process per layer ≈ 1 ms × 5 layers = ~5 ms/cycle (vs current 70 ms = **14× collapse**).

**Closure** (re-verified post-amendment): unchanged from §6.2 — closure 8/8 prompts argmax 4/4 vs vLLM, NMSE/cos preserved, full multi-GPU NPC verify still PASS.

---

### 6.3 `dflash_verify_attn` — verify-shape attention

**File**: `ggml-cuda/dflash-verify-attn.cu`

**Signature**:

```cpp
template<int BLOCK_SIZE, int HEAD_DIM>     // BLOCK_SIZE ∈ {4,5,6,8}; HEAD_DIM = 256 (target)
__global__ __launch_bounds__(128, 2)
void dflash_verify_attn(
    const half * __restrict__ q,                  // [N_slots, 1+BLOCK_SIZE, num_heads=24, HEAD_DIM]
    const half * __restrict__ k_cache,            // [N_slots, SeqLen, num_kv_heads=4, HEAD_DIM]
    const half * __restrict__ v_cache,            // ditto
    const int  * __restrict__ slot_seq_lens,      // [N_slots]
    const int  * __restrict__ kv_block_offsets,   // for paged KV, if used; else null
    half       * __restrict__ output,             // [N_slots, 1+BLOCK_SIZE, num_heads, HEAD_DIM]
    int N_slots,
    float scale
);
```

**Algorithm**: Thinking Machines Lab fixed-split-size + sm_75 PTX `mma.sync.m16n8k8` for the tall-skinny QK product.

**Key choices**:
- `KV_BLOCK_SIZE = 64` (compile-time constant). Number of splits over K dimension = `ceil(seq_len / 64)`.
- Per output row: one CTA. Output row = (slot, query_position_within_block, query_head).
- Per CTA work:
  1. Load Q row from HMM → registers (one fp16 row of HEAD_DIM=256)
  2. Loop over KV blocks (each KV_BLOCK_SIZE positions × HEAD_DIM):
     - Load K block → SMEM (tiled, swizzled for ldmatrix)
     - Compute QK product using `mma.sync.m16n8k8` PTX (fp32 accumulator for softmax precision)
     - Online softmax update (max, exp, sum) in fp32 registers
     - Load V block → SMEM
     - Compute AV product, accumulate in fp32
  3. Final normalize: O_final = O_acc / sum_final
  4. Cast to fp16, vectorized half4 write

**Determinism (Gate 5b binding)**:
- Number of K splits varies with seq_len, but each split's contents (which K positions land in which split) is determined by KV_BLOCK_SIZE and slot's KV layout — independent of how many other slots are concurrently processed.
- Online softmax updates happen entirely within one CTA — no cross-block reduction.
- No `atomicAdd<float>` anywhere.
- PTX `mma.sync.m16n8k8` is deterministic per spec.

**Per-CTA SMEM**:
- K tile (one KV_BLOCK_SIZE × HEAD_DIM swizzled): 64 × 256 × 2 = 32 KiB
- V tile (same): 32 KiB
- → 64 KiB. This MEETS the TU102 SMEM cap. **Constraint**: cannot double-buffer K and V simultaneously; tile-by-tile alternation. Or reduce KV_BLOCK_SIZE to 32 → 16 KiB each, allows double-buffer.

Tradeoff documented; pick KV_BLOCK_SIZE = 32 with double-buffer at Gate 3 if KV_BLOCK_SIZE = 64 single-buffer fails the determinism+perf budget.

**Reference impls to study**:
- [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing) — head_dim=128 FA on sm_75, ~63% of peak on T4. Layout reference.
- [TML batch_invariant_ops Triton FA](https://github.com/thinking-machines-lab/batch_invariant_ops) — fixed-split-size pattern.
- [llama.cpp PR #16016 deterministic FA](https://github.com/ggml-org/llama.cpp/pull/16016) — CUDA implementation of fixed-split-size.

**Allium bindings**:
- `TargetVerifyBlock`
- `VerifyOutputArbitratedByTarget`

---

### 6.4 `dflash_state_checkpoint` / `dflash_state_restore`

**Status (2026-05-13)**: NOT IMPLEMENTED — and the parallel T6.A
mechanism that was briefly landed has been REMOVED. The originally-
spec'd `dflash_state_checkpoint` / `_restore` kernels (and the T6.A
ping-pong scratch in `src/llama-dflash.cpp`) were not needed for the
T6.α coherence path: ik_llama.cpp already has a PER_STEP-aware
checkpoint mechanism (`llama_spec_ckpt_*`) that handles partial-accept
restore at any `accepted_step`. T6.A's parallel scratch was
removed at the post-T6-closure cleanup to reclaim ~150 MiB of GPU
memory per bind and eliminate the maintenance burden of two parallel
state mechanisms.

**Canonical mechanism — `llama_spec_ckpt_*` (PER_STEP mode)**:

The pre-existing API in `include/llama.h:840-847`:

```c
LLAMA_API int  llama_spec_ckpt_init   (struct llama_context *, int mode, int max_tokens);
LLAMA_API bool llama_spec_ckpt_save   (struct llama_context *, llama_seq_id);
LLAMA_API bool llama_spec_ckpt_restore(struct llama_context *, llama_seq_id, llama_pos n_past, int accepted_step);
LLAMA_API void llama_spec_ckpt_discard(struct llama_context *);
```

How it works (PER_STEP mode):
- `llama_spec_ckpt_init(AUTO, max_tokens=BS+1)`: allocates
  `ckpt.per_step_ssm[il]` and `ckpt.per_step_qkv[il]` sized for
  `max_tokens` rows per layer, plus a `s_l_shadow[il]` snapshot tier.
  Selects PER_STEP unless mixed CPU/GPU recurrent layers or
  n_seq_max > 1, in which case falls back to GPU_FALLBACK / CPU.
- `llama_spec_ckpt_save(ctx, seq_id)`: captures the shadow
  (current s_l state) AND sets `kv.save_per_step_ssm = true`.
- The next `llama_decode` with `n_tokens > 1` triggers
  `delta_net::build_qkv` to copy intermediate SSM states to
  `ckpt.per_step_ssm[il]` and the qkv_mixed conv features to
  `ckpt.per_step_qkv[il]` (one entry per batch token).
- `llama_spec_ckpt_restore(ctx, seq_id, n_past, accepted_step)`:
  the CUDA kernel `ggml_backend_cuda_per_step_restore_layers`
  (see `ggml/src/ggml-cuda/per-step-restore.cu`) stitches s_l[il]
  to "state after the first `accepted_step+1` batch tokens" by:
    1. Phase 1 (conv state): for each (col, d), source is
       `qkv[d + src_token * conv_dim]` where
       `src_token = accepted_step - (d_conv-2) + col`. Falls back
       to the shadow when src_token < 0.
    2. Phase 2 (SSM state): byte-copy
       `ssm_ptrs[il] + accepted_step * ssm_state_dim`
       to `s_l[il] + conv_state_dim`.
  Then sets `cells[seq_id].pos = n_past + accepted_step` and
  calls `llama_kv_cache_seq_rm(seq_id, n_past + accepted_step + 1, -1)`.

For DFlash: `accepted_step = n_accepted` restores state to "after
id_last + n_accepted accepted drafts" — exactly the state needed
for the next decode at position `P + n_accepted + 1` (the bonus
position). The bonus is then committed by a single-token decode at
that position.

**Caveat (`save_per_step_ssm` is global)**: the flag stays set until
`llama_spec_ckpt_discard`. While set, any `llama_decode` with
`n_tokens > 1` attempts the per-step save into a buffer sized for
`max_tokens` rows. The caller must `_discard` before any post-cycle
multi-token decode (e.g., between cycles, or use single-token bonus
decodes only).

**Original signatures (deferred)**:

```cpp
__global__
void dflash_state_checkpoint(
    const half * __restrict__ live_state,  // [L_t_dn=48, N_slots, H_kv_t=4, D_t=256, D_t=256]
    half       * __restrict__ scratch,     // ping-pong[K%2]
    int N_slots, int n_dn_layers
);

__global__
void dflash_state_restore(
    const half * __restrict__ scratch,
    half       * __restrict__ live_state,
    const int  * __restrict__ n_accepted_per_slot,  // for partial restore
    int N_slots, int n_dn_layers, int block_size
);
```

These are pure memcpy kernels (no math). Implementation likely just `cudaMemcpyAsync` per slot — but kernel form lets us implement partial restore (only revert the rejected suffix of state mutations).

**Determinism**: trivially deterministic (pure memory move).

**Allium bindings**:
- `HybridTargetRecurrentStateTracking`

---

### 6.5 `dflash_argmax_match`

**File**: `ggml-cuda/dflash-argmax-match.cu`

**Signature**:

```cpp
__global__
void dflash_argmax_match(
    const half * __restrict__ target_logits,      // [N_slots, 1+BLOCK_SIZE, V=248320]
    const int  * __restrict__ draft_tokens,       // [N_slots, BLOCK_SIZE]
    int        * __restrict__ n_accepted,         // [N_slots] output
    int        * __restrict__ bonus_token,        // [N_slots] output
    int N_slots, int block_size
);
```

**Algorithm**: per slot, one warp:
1. Each lane scans `V/32 ≈ 7760` logit entries from target_logits[slot, position=0, :] for argmax
2. Warp-shuffle reduction to find global argmax per position
3. Compare to draft_tokens[slot, position]
4. First-mismatch position = n_accepted (via `__ballot_sync` + `__ffs`)
5. Bonus token at position n_accepted = target's argmax there

**Determinism**: warp-shuffle reductions are deterministic. Single-warp argmax has well-defined tie-breaking (lowest-index lane wins, by convention).

**Allium bindings**:
- `AcceptPrefixDecision`
- `BonusPosIsAnchorPlusNAcceptedPlusOne`

---

### 6.6 `dflash_combine_features` — anchor-level FC + hidden_norm

> **Amendment 2026-05-19 — GEMM dispatch revised.** The FC matmul is now
> dispatched through `dflash_gemm_npc` (`ggml_cuda_mul_mat_f16_pinned`,
> HMMA m16n8k16) batched across all (slot, anchor) pairs. The per-row
> RMSNorm + hidden_norm-weight multiply remains as a thin sub-kernel
> reading the F32 GEMM output and storing F16 `context_states`.
> See subsection §6.6.A below. cuBLAS HGEMM is forbidden in this TU.


**File**: `ggml-cuda/dflash-combine-features.cu`

**Purpose**: Produce `context_states` consumed by `dflash_inject_kv_fused` (5 per-layer launches per anchor). The drafter forward never reads `context_states` directly — it consumes the injected K/V cache that inject writes, which satisfies `@InjectionConsumedAtEveryLayer` via the per-layer attention's cache read. Implements the upstream of vLLM's `precompute_and_store_context_kv` pipeline:

1. Channel-wise concat of 5 source-layer hiddens — `[5, 5120] → [25600]` per anchor (implicit: source_hiddens already laid out with `L_src` as the inner axis).
2. FC matmul — `[5120, 25600] · [25600] = [5120]` per anchor (DFLASH_FC weight).
3. hidden_norm RMSNorm with `[5120]` weight (DFLASH_HIDDEN_NORM).

This is **anchor-level** work (runs once per anchor per cycle), not per-drafter-layer. The 5 per-drafter-layer inject CTAs all consume the same `context_states[slot, anchor, :]`.

**Signature**:

```cpp
__global__ __launch_bounds__(128, 4)
void dflash_combine_features(
    const half * __restrict__ source_hiddens,     // [N_slots, MAL_anchors, L_src=5, D_d=5120]
                                                   //   from extract hook at target_layer_ids
    const half * __restrict__ fc_weight,          // [D_d=5120, L_src * D_d=25600] (DFLASH_FC)
    const half * __restrict__ hidden_norm_weight, // [D_d=5120] (DFLASH_HIDDEN_NORM)
    const float norm_eps,                         // 1.0e-06
    half * __restrict__ context_states,           // [N_slots, MAL_anchors, D_d=5120]  (output)
    int N_slots, int MAL_anchors
);
```

**Grid / block geometry**: 1 CTA per (slot, anchor):
- `dim3 grid(MAL_anchors, N_slots);`
- `dim3 block(128, 1, 1);` — 4 warps.

**Per-CTA work**:

```
1. FC matmul (scalar fp32 per-thread accumulator):
   - Each thread owns OUTS_PER_THREAD = D_d / N_THREADS output rows
     (= 40 outputs/thread at 128 threads), assigned warp-major:
       my_row = o * N_THREADS + tid  for o in 0..OUTS_PER_THREAD-1
     so consecutive warp lanes access consecutive fc_weight rows
     (coalesced HBM reads).
   - For each k_base in 0..K, step K_TILE (= 128):
       Cooperative-load source_hiddens k-strip into SMEM (128 fp16 = 256 B)
       For each thread, for each of its OUTS_PER_THREAD rows:
         For kk in 0..K_TILE-1:
           acc[o] += __half2float(fc_weight[my_row, k_base + kk])
                   * __half2float(src_smem[kk])
   - Per-thread accumulators stay in registers throughout.

2. hidden_norm RMSNorm:
   - Per-thread sum_sq over its OUTS_PER_THREAD accumulators (fp32)
   - Warp-shuffle butterfly within each warp
   - SMEM tree across the 4 warps
   - Broadcast rsqrt(sum_sq / 5120 + norm_eps)

3. Apply norm + vectorized write:
   For each output o in 0..OUTS_PER_THREAD-1:
     my_idx = o * N_THREADS + tid
     out = acc[o] * rsqrt * __half2float(hidden_norm_weight[my_idx])
     context_states[slot, anchor, my_idx] = __float2half(out)
```

**Implementation note — deviation from earlier WMMA design**:

An earlier sketch specified WMMA m16n16k16 fp16 for the FC matmul. This was incompatible with the byte-identity binding against the fp32 scalar reference:

1. WMMA fragment-internal reduction order on Turing is well-defined per-PTX but does not match a serial K-order scalar reference; byte-identity would fail at the LSB across a large fraction of output positions.
2. fp16 accumulators (WMMA's default for fp16 inputs at m16n16k16) lose precision over 25600 multiply-adds. The FC for D_d=5120 has sum_sq ~ O(D_d), and fp16 accumulation is inadequate for byte-identity with an fp32 reference.
3. Performance: FC matmul is bandwidth-bound (250 MiB fc_weight read / 624 GB/s = 400 µs ceiling). WMMA's ~100 µs compute vs scalar fp32's ~240 µs compute is dominated by the bandwidth wall.

Output is kept in per-thread fp32 registers throughout (not SMEM as the earlier sketch had) to avoid the fp32→fp16→fp32 round-trip that would cost the precision the byte-identity test requires.

The RMSNorm sum_sq still uses a parallel warp-shuffle + SMEM-tree reduction; this introduces a small fp32 reduction-order divergence from the scalar reference (which sums in serial fp32 order). Empirically this produces 1-ULP fp16 differences at < 1 % of positions (measured: 6 / 5120 = 0.117 %, all 1 ULP, at the first random seed). The test driver's PASS criterion is "byte-identical OR every disagreement ≤ 1 ULP AND rate ≤ 1 %" — > 1 ULP at any position or rate > 1 % indicates a real precision bug.

**Register budget**: 40 fp32 accumulators / thread + loop counters + addresses + RMSNorm temps. Measured via `--ptxas-options=-v`: **64 registers/thread**. On sm_75: 64 regs × 128 threads × 4 bytes = 32 KiB/block; 65 KiB regs/SM ÷ 32 KiB = 2 blocks/SM register-limited, but warp limit (32 warps/SM ÷ 4 warps/block = 8 blocks/SM) is higher — actual occupancy capped by register pressure to 2 blocks/SM. The earlier ≤40 reg target was based on the WMMA sketch's small fragment accumulator (~8 regs); the scalar-accumulator design naturally lands at 64. Future tuning option: increase block to 256 threads → 20 outs/thread → ~32 regs/thread → 4 blocks/SM. Not pursued in T3 because compute is sub-dominant (bandwidth-bound).

**SMEM budget**:
- source_hiddens K-tile staging: 128 fp16 = 256 B
- warp_sums (RMSNorm cross-warp reduction): 4 × 4 = 16 B
- Headroom + alignment: <1 KiB
- **Total measured: 272 bytes per CTA.** Far under the earlier ~13 KiB estimate because output is register-resident.

**Determinism**:
- No WMMA → rule about WMMA fragment shape doesn't apply.
- FC per-thread accumulator is private; no cross-thread accumulation during FC.
- RMSNorm uses warp-shuffle + SMEM tree — deterministic for fixed block dim (128).
- One CTA per (slot, anchor) output row — no Split-K cross-block reduction.
- No `atomicAdd<float>` anywhere.
- Run-to-run output is bit-identical (Gate 5b np-invariance binding precondition).

**Performance envelope**:
- FLOPs per CTA: 5120 × 25600 = 131 MFLOPs (FC) + ~5 KFLOPs (RMSNorm).
- Per cycle at np=8, MAL=3: 24 CTAs × 131 MFLOPs = 3.1 GFLOPs.
- Compute ceiling (TU102 fp32 cuda cores ~13 TFLOPs/s, fp16 HFMA2 ~26 TFLOPs/s): scalar fp32 path ~240 µs.
- Bandwidth ceiling (FC weight 250 MiB / 624 GB/s): 400 µs. fc_weight is shared across all 24 CTAs → with one big-launch grid, L2 reuse drops the effective bandwidth substantially.
- Plausible measured: 200–500 µs/cycle. Budget held at 0.5 ms.

**Allium bindings**:
- `FuseProjectionFcWeight` — kernel applies DFLASH_FC weight directly; conversion shape [5120, 25600] preserved end-to-end.
- `FeatureWidthMatchesTarget` — input D_d = 5120 = target hidden_size.
- `CombineOrderFCThenHiddenNorm` — kernel does FC then hidden_norm in that order; reversal would break byte-identity vs scalar reference with non-identity hidden_norm_weight.
- `ContextStatesAnchorLevel` — output shape [N_slots, MAL_anchors, D_d]; the 5 per-drafter-layer inject CTAs all consume the same context_states[slot, anchor, :].

**Structural note vs vLLM**:
vLLM's `precompute_and_store_context_kv` (qwen3_dflash.py:441–448) does `rms_norm(context_states, hidden_norm_weight)` THEN a separate `F.linear(normed_context_states, fused_kv_weight)`. Our split here is FC first (no preceding RMSNorm) then hidden_norm. **This matches vLLM's mathematical pipeline if and only if `source_hiddens` is already raw (not pre-normed)** — which it is: the extract hook at `l_out-<il>` captures the post-residual-add hidden state in the target's residual stream, which is what vLLM's `combine_hidden_states` receives. Verification: T2 cross-stack cosine ≥ 0.99988 confirms the captured value matches vLLM's `combine_hidden_states` input.

Cross-check: vLLM's `combine_hidden_states` (qwen3_dflash.py:656) does FC FIRST, then hands the result to `precompute_and_store_context_kv` which does hidden_norm. Same order as our kernel; our kernel just fuses the two into one launch.

---

### 6.6.A Combine features GEMM dispatch (revised 2026-05-19)

**Status**: This subsection revises the matmul dispatch inside §6.6. The per-(slot, anchor) scalar-fp32 K-loop FC matmul (which ran ~108 ms/cycle — 51.6% of post-Path-A GPU time) is replaced with one batched pinned-HMMA dispatch. The per-row RMSNorm + hidden_norm-weight scale + F16 store remains as a thin sub-kernel.

**Canonical dispatch** (one call/cycle):

```
M = N_slots * MAL_anchors
dflash_gemm_npc(fc_weight, source_hiddens, fc_out_f32, K=25600, N_cols=5120, n_rows=M)
combine_features_norm_kernel<<<grid(M), block(256)>>>(
    fc_out_f32, hidden_norm_weight, norm_eps, context_states_f16);
```

**Layout coincidence**: `source_hiddens[N_slots, MAL_anchors, L_src=5, D_d=5120]` row-major is byte-identical to `[M, L_src*D_d=25600]` when (slot, anchor) collapse to M — the channel-wise concat of the 5 source-layer hiddens IS the contiguous K-axis (per the prior kernel's comment at `dflash-combine-features.cu:78-82`). `fc_weight[D_d=5120, L_src*D_d=25600]` row-major matches pinned's `weight[k + n*K]` convention with `n=col`, `k=k`. No transpose, no pack kernel.

**F32 scratch**: `fc_out_f32[M, 5120]` ≈ 1.8 MB at M=88. Allocated via `cudaMallocAsync` inside the launcher, freed at exit.

**Norm sub-kernel**: one CTA per row, threads stride D_d=5120, fp32 RMSNorm + hidden_norm_weight scale + F16 store. Block-wide reduction via warp-shuffle + SMEM tree (same pattern as the prior kernel's norm step).

**Determinism contract**: pinned is byte-identity-by-construction across batch composition; norm sub-kernel is per-row single-CTA single-writer; no atomics or cross-CTA reductions. No batch-shape-dependent control flow.

**K-divisibility**: K=25600 ÷ 16 = 1600 ✓.

**Forbidden**: `cublasGemmEx`, `cublasHgemm`, any handle from `ctx.cublas_handle(...)` may NOT be invoked from `ggml/src/ggml-cuda/dflash/dflash-combine-features.cu`.

**Per-call perf model** (TU102):
- FC at M=88: 11.5 GFLOP, ~250 MB weight read → memory roofline ~375 µs.
- One pinned call + norm sub-kernel ≈ 3 ms/cycle estimated (vs current 108 ms = **36× collapse**).

**Closure**: unchanged from §6.6 — closure 8/8 prompts argmax 4/4 vs vLLM, full multi-GPU NPC verify still PASS.

---

### 6.7 Bonus-position re-decode (T6 mechanism)

**Status**: introduced at T6 to close the T5 inherited late-stream-coherence gap.

**Problem**: per cycle, target's verify decode batch is
`[id_last @ P, c1 @ P+1, ..., cBS @ P+BS]`. After accept (n_accepted = k < BS):
- Positions [P, P+k] are committed (id_last + accepted draft tokens).
- Position P+k+1 = bonus token; its KV entry in target was decoded with input
  `c_{k+1}` (the REJECTED drafted token), NOT with `bonus`.
- For the next cycle's `combine_features`, the cb_eval extract buffer row
  at index `P+k+1` reflects `c_{k+1}`'s hidden, not `bonus`'s hidden.

**Mechanism**: after each cycle's accept decision (when `n_accepted < BS`),
re-decode the bonus token at position `P+k+1` as a single-token target
decode. This:
- Overwrites the wrong KV entry at `P+k+1` with the correct one (input = bonus).
- Triggers the cb_eval hook, which appends one row to the extract buffer
  — overwriting the wrong row from the verify batch (after the
  T6.8 trim-on-seq_rm — see §6.8).
- Mutates the DeltaNet recurrent state by one step, advancing from
  the position-`P+k` checkpoint (restored via §6.4) through `bonus` to
  the correct position-`P+k+1` state.

**When `n_accepted == BS` (full accept)**: bonus row is the post-block row
at batch index BS, decoded with input `cBS`. If `cBS == bonus_token`
(equivalent to "drafter predicted the bonus correctly too"), no re-decode
needed. Otherwise re-decode the bonus at position `P+BS+1` similarly.

**Placement**: inside `llama_dflash_draft` (C API self-contained per locked T6
Q&A). The function does the target verify decode internally OR the caller
drives the verify decode; either way, after the accept decision is computed
the C API issues the bonus re-decode and re-reads the extract buffer. This
keeps the cycle atomic from the caller's perspective.

**Test hook**: `llama_dflash_get_cycle_stats(ctx, &n_cycles, &n_decoded, &n_re_decodes)`
lets tests assert exactly one bonus re-decode per cycle.

**Allium bindings**:
- `AtomicityPerCycle` (the cycle is all-or-nothing, including bonus re-decode)
- `DraftKVRollbackOnRejection` (the wrong KV / hidden row at the bonus
  position is corrected via re-decode)
- `EffectiveSeqLensSubtractsRejected` (post-cycle seq state matches
  anchor_pos + n_accepted + 1, including the corrected bonus position)

**Hybrid model concern**: the re-decode advances DeltaNet recurrent state.
For the state to be CORRECT, the recurrent state must be at the
position-`P+k` checkpoint when the re-decode runs. This is the
`dflash_state_restore` step in §6.4 — the layered foundation that makes
this re-decode work on Qwen 3.6 27B.

---

### 6.8 cb_eval extract hook contract (T6)

**Status**: T2 introduced the cb_eval hook for extracting target's residual
stream at the 5 source-layer indices. T5 discovered that the hook's
overwrite-per-ubatch semantics broke multi-ubatch prefill, fixed by
switching to append-per-ubatch. T6 closes the remaining contract: the
hook's buffer must stay in sync with target's effective seq_len.

**Append-per-ubatch (T5 fix, already landed)**:
The hook appends `n_tokens × D_emb` floats per fire. Buffer length after a
sequence of decodes = total positions decoded (regardless of how the
decoder split into ubatches).

**Trim-on-`seq_rm` (T6 new)**:
When target's KV cache rolls back via `llama_kv_cache_seq_rm(ctx, seq_id,
p_start, p_end)`, the extract buffer for each registered source layer
must trim correspondingly:
- If `p_end == -1` (open-ended remove): truncate buffer to the first
  `p_start * D_emb` floats.
- If `p_end > p_start` (range remove): remove the slice
  `[p_start * D_emb, p_end * D_emb)` from the buffer.

Implementation: extend `llama_kv_cache_seq_rm` (or its internal entry)
to invoke a per-source-layer trim callback when DFlash is bound.
Alternatively: expose a separate `llama_dflash_trim_extract(ctx, p_start, p_end)`
public API and have the example call it alongside `seq_rm`. The
latter avoids cross-cutting changes to the KV cache code path; the
former is more semantically robust.

**Decision (locked)**: expose `llama_dflash_trim_extract(ctx, p_start, p_end)`
and call it alongside `seq_rm` from `llama_dflash_draft`'s internal accept
path. Keeps the KV cache code path untouched and keeps the contract
self-contained in the DFlash module.

**Test hook**: `test-dflash-extract-trim-on-seq-rm.cpp` (T6.C):
- Drive 3 cycles with planted partial accepts.
- After each cycle, snapshot the extract buffer length.
- Assert length == anchor_pos at the end of cycle N+1 for each cycle N.
- Assert no stale rows past anchor_pos after seq_rm.

**Allium bindings**:
- `EffectiveSeqLensSubtractsRejected` (extract buffer length tracks
  target's effective seq_len)
- `DraftKVRollbackOnRejection` (drafter-side state — including the
  extract buffer — rolls back with target's KV)

---

## 7. Allium-invariant ↔ kernel binding table

Each Allium invariant from `dflash.allium` is bound to a specific kernel or test:

| Allium invariant | Kernel / location | Binding mechanism |
|---|---|---|
| `DraftBlockEmit` | `dflash_drafter_forward` | Output logits at all BLOCK_SIZE positions; greedy argmax at host post-kernel |
| `FeatureSourceFixedPerDeployment` | drafter loader + persistent kernel internals | target_layer_ids read from GGUF metadata at load; persistent kernel inlines them |
| `FuseProjectionFcWeight` | `dflash_combine_features` | DFLASH_FC weight loaded as one tensor [5120, 25600]; kernel applies it before any K/V proj |
| `FeatureWidthMatchesTarget` | `dflash_combine_features` | Input source_hiddens dim D_d=5120 matches target hidden_size |
| `PerLayerArity` | `dflash_inject_kv_fused` host launcher | Host loops L_d=5 times, one launch per drafter layer |
| `HeadShapeMatchesDraft` | `dflash_inject_kv_fused` | Kernel reads per-layer k_weight/v_weight/k_norm_weight at drafter's head shape (8×128) |
| `KAsymmetricallyNormedVNot` | `dflash_inject_kv_fused` | K_norm + RoPE applied to K only; V projected then written without norm/rotation |
| `InjectedAnchorAlignment` | `dflash_inject_kv_fused` | anchor_positions[slot, anchor] sets cache write position |
| `InjectionConsumedAtEveryLayer` | `dflash_drafter_forward` | Each of 5 layers consumes context_states (from `dflash_combine_features`) at its inject point |
| `TargetVerifyBlock` | `dflash_verify_attn` + existing target verify graph | Verify shape ne[1]=BLOCK_SIZE+1; one block per output row |
| `LongestPrefixMatchUnderArgmax` | `dflash_argmax_match` | Target argmax wins on every mismatch; longest accepted prefix bounded by first |
| `AcceptPrefixDecision` | `dflash_argmax_match` | Contract-level: implements the longest-prefix accept decision |
| `BonusPosIsAnchorPlusNAcceptedPlusOne` | `dflash_argmax_match` (bonus_token output) | Bonus at position n_accepted, sampled from target |
| `QuerySpanIsOnePlusN` | verify call site | `ne[1] = 1 + num_spec_tokens` enforced by build_dflash_verify_batch() |
| `EffectiveSeqLensSubtractsRejected` | post-verify state advance | seq_pos += n_accepted + 1 (bonus), NOT BLOCK_SIZE |
| `NumRejectedTokensFlowsBackToProposer` | post-verify rejection accounting | n_rejected = BLOCK_SIZE - n_accepted; passed to next drafter call |
| `HybridTargetRecurrentStateTracking` | `dflash_state_checkpoint` / `dflash_state_restore` | Save before verify; conditionally restore on partial accept |
| `SharedEmbedAndLMHead` | drafter loader | Embed (F16) and lm_head (BF16) shared with target's `token_embd.weight` / `output.weight` at server init; drafter never owns these tensors |
| `PerSlotVerifyDispatchAtMultiSlot` | verify call site | Verify kernel is one-block-per-output-row → slot-isolated by construction |
| `NoCrossSlotRegionOverlap` | KV cache layout `[layer, slot, ...]` | Slot dimension in cache addressing → can't overlap |
| `SyncBeforeStepAdvance` | stream sequence (§5) | Explicit `cudaEventRecord`/`Wait` between cycles |
| `ProbabilisticVerifyOutOfScope` | `dflash_argmax_match` | Greedy argmax only; no rejection sampling layer |
| `DPAttentionNotSupported` | server init | Hard check at startup |
| `PipelineParallelismRequiresPpSizeEq1` | server init | Hard check at startup |
| `MultimodalTurnsRoutedAroundDrafter` | server prompt routing | Caller-side check before drafter dispatch |
| `ContextBudgetAtNp8` | server profile | Profile config enforces ctx ≤ 64k at np=8 |

(35 in-contract @invariants from `dflash.allium` map to one or more of the above; see `allium-tla-binding.json` for the full manifest.)

---

## 8. Determinism guarantees

The bit-identical Gate 5b binding requires:

1. **No `atomicAdd<float>`** anywhere in the drafter forward, verify attn, or inject KV. Build-time grep gate on the relevant files.
2. **All matmul tile dims are compile-time constants** — no occupancy-tuned heuristics.
3. **One CTA per output tile** — no Split-K cross-block reductions on the drafter or verify paths.
4. **Warp-shuffle for inner reductions, SMEM tree for block-level** — no atomic-based reductions.
5. **Block-idx → output tile mapping is fixed at kernel design time** — SM-arrival order doesn't affect output values.
6. **WMMA tile shape compile-time fixed** (`m16n16k16` for drafter, `m16n8k8` PTX for verify-attn).
7. **FA split-SIZE not split-count** — `KV_BLOCK_SIZE` is a compile-time constant.

The test harness:
- `tests/test-dflash-combine-features.cpp` — combine output bit-identical to scalar reference (FC + hidden_norm) across dim sweep
- `tests/test-dflash-inject-fused.cpp` — fused vs scalar bit-identical for all 5 drafter layers (K_proj+V_proj+K_norm+RoPE+write)
- `tests/test-dflash-determinism-np-invariance.cpp` — drafter logits bit-identical across np ∈ {1, 2, 4, 8}
- `tests/test-dflash-determinism-ne5.cpp` — verify-attn output bit-identical across 3 runs at fixed config
- `tests/test-dflash-state-revert.cpp` — DeltaNet save/restore round-trip is bit-identical

---

## 9. SMEM / register budgets summary

| Kernel | SMEM/CTA | Regs/thread (target) | Threads/block | Occupancy target |
|---|---:|---:|---:|---|
| `dflash_drafter_forward` | ~20 KiB | ≤ 64 | 256 | 2 blocks/SM (cooperative) |
| `dflash_combine_features` | 272 B (measured) | 64 (measured) | 128 | 2 blocks/SM (register-limited) |
| `dflash_inject_kv_fused` | 4368 B (measured) | 74 (measured) | 128 | 2 blocks/SM (register-limited) |
| `dflash_verify_attn` (KV_BLOCK_SIZE=32 + double-buffer) | ~32 KiB | ≤ 64 | 128 | 2 blocks/SM |
| `dflash_argmax_match` | <1 KiB | ≤ 32 | 32 (1 warp) | 8+ blocks/SM |

Total SMEM at peak concurrent (inject + verify simultaneous on different streams): well within 64 KiB / SM × 72 SMs.

---

## 10. Implementation order

Maps to `DESIGN.md` §6 gate sequence:

| Step | Deliverable | Gate |
|---|---|---|
| 1 | `convert_hf_to_gguf.py::DFlashModel` | Gate 1 (converter binding) |
| 2 | `dflash_extract_features` hook into Qwen3 build graph | Gate 2 (extract hook) |
| 3 | `dflash_combine_features` kernel + `dflash_inject_kv_fused` kernel + scalar references + unit tests | Gate 3a |
| 4 | `dflash_drafter_forward` persistent kernel + `dflash_argmax_match` + plumbing | Gate 3b, Gate 4 |
| 5 | `dflash_verify_attn` kernel + ne=5 determinism test | Gate 5 |
| 6 | `tests/test-dflash-determinism-np-invariance.cpp` | Gate 5b |
| 7 | Production profile + Qwen3.6-27B Gate 6 measurement | Gate 6 |
| 8 | (conditional) batched verify at np>1 | Gate 7 |

---

## 11. Open kernel-level questions (to resolve during implementation)

These don't block the design freeze; resolved during the relevant gate:

- **`dflash_verify_attn` SMEM budget**: KV_BLOCK_SIZE=64 single-buffer (32 KiB K + 32 KiB V) saturates TU102's 64 KiB SMEM/SM. KV_BLOCK_SIZE=32 + double-buffer (8 KiB × 4 = 32 KiB) leaves room but doubles split count. Decide based on Gate 5 perf measurement.
- **Persistent drafter kernel cooperative-launch grid size**: depends on actual register pressure measured at Gate 3a. Initial guess 144 blocks (2/SM × 72 SMs); may shrink to 72 if register-bound.
- **LM head matmul shape**: hidden=5120 → vocab=248320 is a very wide matmul. May want to fuse into the persistent kernel or break out as a separate kernel; decide at Gate 3b.
- **Sliding-window attention mask construction** inside drafter mega-kernel: precompute mask once at server init vs compute on the fly inside the kernel. SMEM cost vs compute cost tradeoff.

---

## 12. Source-of-truth references

External research (cited 2026-05-12, deep-research session, see commit history for the agent reports):

- [NVIDIA Cooperative Groups documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
- [NVIDIA Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)
- [Thinking Machines Lab — Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [llama.cpp PR #16016 — Deterministic inference mode (CUDA)](https://github.com/ggml-org/llama.cpp/pull/16016)
- [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing)
- [CUTLASS sm_75 paths](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/warp)
- [Bruce-Lee-LY — MMA PTX programming](https://bruce-lee-ly.medium.com/nvidia-tensor-core-getting-started-with-mma-ptx-programming-508e44a6cb7d)
- [Lei Mao — NVIDIA Tensor Core MMA TN Layout](https://leimao.github.io/blog/NVIDIA-Tensor-Core-MMA-Instruction-TN-Layout/)
- [Lei Mao — CUDA Shared Memory Swizzling](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/)

Internal:
- `specs/dflash/DESIGN.md` — implementation shape
- `specs/dflash/dflash.allium` — behavioural contract
- `specs/dflash/DFlashCycle.tla` + `DFlashMultiSlot.tla` — TLA+ models
- `specs/dflash/allium-tla-binding.json` — Allium ↔ TLA binding manifest
- `scripts/vllm_sm75_patches.py` — vLLM oracle runtime patches
- `data/gate0-*.json` — Gate 0 empirical measurements

---

## 13. Change discipline

Per CLAUDE.md §5: any change to this document requires a commit + push immediately, separate from code changes. The contracts here are what implementation depends on; drift between this document and the kernels is a defect.

When a kernel implementation surfaces a constraint that requires deviating from this document:
1. STOP. Don't silently work around it.
2. Surface the constraint to the user, propose the design change.
3. Get explicit confirmation.
4. Update this document and commit/push BEFORE changing the kernel.

This is the rule from `feedback_surface_tradeoff_decisions.md` applied to kernel design.
