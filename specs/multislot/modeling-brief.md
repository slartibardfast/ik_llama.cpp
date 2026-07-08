# Modeling Brief — PHASE45 D10.e Multi-Slot Decode Determinism

System: ik_llama.cpp's batched decode pipeline for Qwen 3.6 27B (hybrid:
DeltaNet recurrent ~75% of layers + standard attention ~25%) on 2× RTX
6000 PCIe (sm_75) under `--parallel N`.

**Category**: B — Concurrent / Lock-Free / Runtime. Shared state across
concurrent slots (recurrent state buffers, GPU compute dispatch, KV
cache); bug manifests as a determinism violation, not a message-passing
protocol issue.

## Two-Phase Modeling Plan

The user directive is: capture what we have FIRST in TLA+, then expand
to multi-slot MTP with safety invariants. The artifacts mirror this:

| Phase | Spec file | Scope | Captures |
|-------|-----------|-------|----------|
| 1 | `Decode.tla` | Current batched plain decode (no MTP) | The bug as it manifests today: `Compute(state, token, batch_seqs)` depends on `batch_seqs` |
| 2 | `DecodeMTP.tla` (extends `Decode`) | Multi-slot speculative decode | Adds draft/verify/rollback + safety invariants |
| 2.5 | `Fix.tla` (extends `DecodeMTP`) | Per-slot dispatch fix | Models the `PerSlotMode` toggle + proves invariants hold |

Phase 1 has NO safety properties — it's a faithful capture, not a
verification. Phase 2 adds MTP-specific safety. Phase 2.5 adds the fix
and shows it discharges Phase 2's safety obligations.

## Bug Family A — Batch-shape state divergence in DeltaNet

**Mechanism**: `delta_net::build_layer_attn_linear` (file
`src/llama-delta-net.cpp:679-733`) dispatches based on `all_same_seq`:

- Single-seq batch (M=1) → `all_same_seq=true` → fast-path call to
  `build_layer_attn_linear_core` once with full N-token input.
- Multi-seq batch (M=N) → `all_same_seq=false` → blocks loop, one
  sub-call to `build_layer_attn_linear_core` per contiguous-seq block.

**Phase 1 spike (D10.e.0.P) confirmed forcing always-blocks-path
produces identical δ values to baseline.** Therefore the fast-vs-blocks
topology divergence is NOT the sole cause. The remaining mechanism:

1. ggml graph-execution interleaving across blocks within a single
   `ggml_cgraph` — when 2+ blocks live in the same compute graph, the
   topological scheduler may interleave ops. Shared compute scratch
   buffers and cuBLAS workspace state evolve through interleaved ops
   differently than through single-block ops.
2. Graph cache topology key may include batch dimension — M=1 (n=1) and
   M=N (n=N) produce distinct cached graphs with different kernel
   choices.
3. CUDA/cuBLAS kernel choice depending on N — MMQ for q4_0 weight
   projections in QKVZ tiles by N. Different tile choice ⇒ different
   float-summation order ⇒ different bytes.

Empirical: δ(solo M=1, M=2 r0) at layer 1 (DeltaNet) = 1.19e-3.

**Modeled as**: `Compute(state, token, batch_seqs)` is a non-trivial
function of `batch_seqs` even when restricted to slot s's view.

## Bug Family B — Row asymmetry in FA mma_f16 kernel

**Mechanism**: `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1`
(`ggml/src/ggml-cuda/fattn-mma-f16.cu:154-174`). For Qwen 3.6 27B
(gqa_ratio=6 → ncols2=2) at hs=256, the tier table is {4,8,16,32}:

- M=1 ne[1]=1: tier 1, ncols1=4
- M=2 ne[1]=2: tier 1, ncols1=4 (**same template instance**)

Both M=1 and M=2 hit `mma_f16_case<256, 4, 2>`. Verified: forcing largest
tier (64/ncols2=32) leaves output unchanged. Dispatch is NOT the
mechanism.

The kernel processes ncols1=4 Q rows per block. With ne[1]=2, only 2
are real, 2 are padding. Cross-row warp reductions span all 4 row-slots
in shared memory. Per-row output depends on:
- Position of the row within the 4-slot block
- Which other rows in the block are real vs padding
- FP-summation order over warp-shared state

Kernel-internal asymmetry, not dispatch. Empirical: δ(M=2 r0, M=2 r1)
at layer 3 = 1.27e-3.

**Modeled as**: same `Compute(state, token, batch_seqs)` abstraction as
Family A. The two bugs collapse to the same control-flow argument at
this abstraction level — both are forms of "compute at slot s depends
on batch_seqs."

## Bug Family C — Composite chaos amplification

Bug A introduces δ ≈ 1e-3 per DeltaNet layer (3/4 of layers). Bug B
introduces δ ≈ 1e-3 at each std-attn layer (1/4 of layers). Lyapunov-class
chaos amplification of ~1.05× per layer over 64 layers reaches δ ≈ 0.48
at logits, crossing the discrimination margin and flipping output
tokens.

**Out of scope** for TLA+ — modeling FP arithmetic exhausts state space
without yielding control-flow insight. Captured by composition of A and B.

## Phase 1 (Decode.tla) Specification Plan

### Variables

- `pending`: function from Slots → sequence of input-token indices
- `slot_state`: function from Slots → sequence of "history witnesses"
  (an opaque value that grows only when the slot has been advanced)
- `slot_output`: function from Slots → sequence of output token indices
- `step_count`: nat, bounded by `MaxStep` for finite MC

### Actions

- `Enqueue(s)`: append a fresh input-token to pending[s]. Models
  upstream scheduler enqueuing tokens after sampling.
- `ProcessBatch(B)`: B ⊆ Slots, |B| ≥ 1, every s ∈ B has nonempty
  pending. For each s ∈ B:
  - Compute new history witness as `Compute(slot_state[s], head(pending[s]), B)`.
  - Append to slot_output[s].
  - Pop pending[s].
  - The dependence on B is what models the bug.

### Sanity Invariants (Phase 1)

- `NoSpontaneousState`: state[s] grows only via ProcessBatch.
- `PendingNotConsumedTwice`: pending[s] never repeats a token twice.
- `BoundedStep`: step_count ≤ MaxStep.

### What Phase 1 does NOT have

- No PerSlotDeterminism invariant. Phase 1 captures the buggy reality
  faithfully; if we asserted determinism here, MC would always violate
  it (the bug IS the model). Determinism is the goal of Phase 2.5.
- No MTP draft/verify. Plain decode only.
- No fix. The `Compute(...)` function in Phase 1 is the buggy version.

## Phase 2 (DecodeMTP.tla) Specification Plan

Extends Phase 1 with multi-token prediction:

### Additional Variables

- `draft[s]`: sequence of speculative draft tokens for slot s
- `draft_state[s]`: pre-draft snapshot of slot_state, for rollback
- `draft_output_ghost[s]`: speculative output, only committed on accept
- `accept_count[s]`: how many drafts were accepted at last verify

### Additional Actions

- `BuildDraft(s, n)`: produce n draft tokens speculatively for slot s.
  Updates draft[s], snapshots state into draft_state[s].
- `VerifyDraft(s, B)`: process the draft + 1 verify token for slot s
  through the model in batch with other slots in B; emit acceptance
  decision per draft token.
- `CommitAccepted(s, k)`: keep the first k drafts (accepted), advance
  slot_output, finalize slot_state.
- `Rollback(s, k)`: revert state to snapshot from `draft_state[s]`,
  drop drafts past position k.

### Phase 2 Safety Invariants

- `DraftRollbackCorrectness`: after Rollback(s, k), slot_state[s] equals
  what it would have been if only the first k drafts had ever been
  computed (equivalently: equal to applying ProcessBatch sequentially
  for the accepted prefix only).
- `NoCrossSlotDraftLeak`: BuildDraft(s, n) does not modify slot_state[t]
  for t ≠ s. Pure containment property.
- `VerifyConsistency`: the per-slot acceptance decision in VerifyDraft
  depends only on (slot_state[s], draft[s], verify_token[s]) — NOT on
  what other slots are in B.
- `MTPInvariant`: at any reachable state, slot_state[s] reflects exactly
  the committed (non-rolled-back) tokens for slot s.

The Phase 2 base spec uses the same buggy `Compute(...)` from Phase 1.
Therefore `VerifyConsistency` will be VIOLATED in MC under PerSlotMode=
FALSE. This is intended — it documents the bug formally.

## Phase 2.5 (Fix.tla) Specification Plan

Adds:
- `PerSlotMode` boolean parameter
- `ProcessBatch` is enabled only when |B| = 1 if PerSlotMode = TRUE
- `VerifyDraft` is restructured under PerSlotMode = TRUE to dispatch
  per-slot (each VerifyDraft handles one slot, with batch_seqs={s}).

### Phase 2.5 Verification Goals

- Under PerSlotMode = FALSE: Phase 2 invariants violate (we know).
- Under PerSlotMode = TRUE: Phase 2 invariants HOLD. TLC convergence is
  the certificate.
- Liveness: under fair scheduling, all enqueued tokens are eventually
  emitted. (Optional, may exceed MC budget.)

## Modeling Boundaries

- **Floating-point arithmetic**: not modeled. The `Compute` function is
  abstract; we model the *control-flow dependence* on batch_seqs, not
  the FP details that produce δ.
- **Specific kernel internals**: not modeled. Bugs A and B collapse at
  this abstraction level.
- **KV cache memory layout**: not modeled. State is opaque.
- **Layer count, head dimension, gqa_ratio**: not modeled. Numeric
  details that don't affect the control-flow argument.
- **Token vocabulary**: tokens are abstract symbols.
- **Server scheduler internals**: modeled abstractly via Enqueue/Process
  actions.

## Bug Family Coverage Table

| Family | Phase 1 captures | Phase 2 invariant | Phase 2.5 fix |
|--------|------------------|-------------------|---------------|
| A — DeltaNet batch-shape | `Compute` depends on B | `VerifyConsistency` | PerSlotMode forces \|B\|=1 |
| B — FA row asymmetry | Same `Compute` abstraction | `VerifyConsistency` | Same fix |
| C — Composite chaos | Out of scope | Out of scope | N/A |

## What TLA+ Verification Proves

If TLC reports `VerifyConsistency` violated under PerSlotMode=FALSE and
holds under PerSlotMode=TRUE, then:

1. The bug is **structural**: any execution where batched compute
   depends on `batch_seqs` violates per-slot output determinism.
2. Per-slot dispatch is **sufficient** to restore consistency,
   regardless of the specific kernel internals.
3. The implementation must ensure ProcessBatch fires only with |B|=1
   when PerSlotMode is enabled. **This is the engineering contract the
   implementation refines from.**

The spec does NOT prove that any particular tactic (per-slot streams,
cudagraph capture-replay, sequential per-slot dispatch) achieves
PerSlotMode=TRUE — that's a refinement. The spec validates the
architectural invariant.
