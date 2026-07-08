# Modeling Brief — DFlash Speculative Decoding (single + multi-slot)

System: ik_llama.cpp's DFlash speculative-decoding pipeline for
Qwen 3.6 27B target paired with z-lab/Qwen3.6-27B-DFlash drafter on
2× Quadro RTX 6000 (sm_75, 24 GiB each) under `--parallel N` for N ≤ 8.

**Category**: B — Concurrent / Lock-Free / Runtime. Shared state
across concurrent slots (target KV cache, per-slot drafter KV cache,
per-slot in-flight verify buffers); bugs manifest as either cycle-
state corruption (single-slot) or determinism violations across
slots (multi-slot). Not a message-passing protocol.

Primary sources for the model:

- Allium spec `specs/dflash/dflash.allium` — 6 contracts,
  34 invariants, 13 entities. The contracts are the speculative
  cycle actions; the invariants are the safety properties to verify.
- vLLM PR #40898 source at `/opt/models/refs/vllm-pr-40898/` and
  installed at `/opt/models/venv-vllm/lib/python3.13/site-packages/`.
  Key files: `vllm/v1/spec_decode/dflash.py` (457 lines) +
  `vllm/model_executor/models/qwen3_dflash.py` (703 lines).
- DFlash paper arXiv:2602.06036 §4.1 (block-diffusion training), §4.2
  (mask conventions), §5.4.4 (block-size generalisation).
- Prior multi-slot TLA work at `specs/multislot/StreamFix.tla` for
  concurrent dispatch primitives (cudaStream lifecycle, in-flight-
  region tracking) — directly reused here. Originally PHASE45 D10.e.

The goal is multi-slot from the start. Single-slot bug families
(A, C) establish the cycle structure; multi-slot families (B, D)
verify safe concurrent dispatch.

## Two-Phase Modeling Plan

Multi-slot is the work's point. Single-slot is established as the
"cycle works correctly at n_parallel=1" floor before adding the
multi-slot extension on top.

| Phase | Spec file | Scope | Captures |
|-------|-----------|-------|----------|
| 1 | `DFlashCycle.tla` | Single-slot 4-action cycle (draft → verify → accept → advance) | Cycle structure: rejection-propagation, anchor advancement, KV state. Bug families A and C. |
| 2 | `DFlashMultiSlot.tla` (extends `DFlashCycle`) | n_parallel ≤ 4 slots running the cycle concurrently | Per-slot dispatch + concurrent CUDA streams. Bug families B and D. |
| 2.5 | `DFlashMultiSlotFix.tla` (extends `DFlashMultiSlot`) | Per-slot dispatch as the only correct shape | Models the `PerSlotDispatch` toggle vs `SingleGridDispatch`; proves only the former discharges B's safety invariants. |

Phase 1 has safety properties (single-slot is non-trivial). Phase 2
adds the multi-slot concurrency safety. Phase 2.5 narrows to the
fix and verifies it.

## §2 Bug Families

### Family A — Rejection-propagation chain breaks (single-slot)

**Mechanism**: After `AcceptPrefixDecision` returns `n_accepted`, the
value `(block_size - n_accepted)` is the number of rejected drafts.
This count must flow forward into the *next* `DraftBlockEmit` cycle
as `num_rejected_tokens`. vLLM's `set_inputs_first_pass`
(`dflash.py:211`, `262`, `286`) consumes it for three purposes:

1. Overwrite the rejected-suffix slots in draft KV (don't treat as
   live context next step).
2. Compute `effective_seq_lens = cad.seq_lens - num_rejected_tokens`
   so attention metadata sees only the valid prefix
   (`dflash.py:283-287`).
3. Place the next anchor at
   `block.anchor_pos + n_accepted + 1` (the bonus token), not at
   `block.anchor_pos + block_size + 1` (would-be position if all
   drafts accepted).

**Failure mode**: An implementation that drops the rejected count
between cycles produces silently-wrong attention as soon as one
non-full accept occurs. The next verify sees `seq_lens` that
include rejected positions as live; the target's logits at those
positions reflect garbage state.

**Source**: spec invariants `NumRejectedTokensFlowsBackToProposer`
(AdvanceState), `EffectiveSeqLensSubtractsRejected`
(TargetVerifyBlock), `BonusPosIsAnchorPlusNAcceptedPlusOne`
(AcceptPrefixDecision).

**Modeled as**: state variable `n_rejected_prev` updated only in
`AdvanceState`, consumed only in the *next* `TargetVerifyBlock`'s
preconditions and effective_seq_lens computation. Bug-injection
flag `RejectionDropped` lets one branch model the "drop the rejected
count" failure (clear `n_rejected_prev` after AdvanceState instead
of preserving it for the next cycle).

### Family B — Multi-slot byte-divergence under single-grid dispatch

**Mechanism**: vLLM's `copy_and_expand_dflash_inputs_kernel`
(`dflash.py:246-275`) dispatches over the entire batch:
`grid = (batch_size, num_blocks)`. At `n_parallel > 1`, every slot's
inputs are packed into one kernel grid; the actual decoder forward
runs as one attention call over all slots' query spans. This is the
same dispatch *shape class* (single-grid-over-N-slots, not per-slot)
that produced byte-level non-determinism in PHASE45 D10.e on the
same hardware (`feedback_no_overlapping_benchmarks.md`,
`project_mtp_multislot_determinism_investigation_failed.md`).

The Allium spec defends with `PerSlotVerifyDispatchAtMultiSlot`:
"At n_parallel > 1, the verify forward is dispatched per-slot
(|B|=1 in every transformer compute call), NOT as a single batched
forward over all slots."

**Failure mode**: Same-prompt np=2 greedy decode produces
byte-different output streams across slots. Implementation regression
or a vLLM-equivalent batched-verify optimisation that bypasses
per-slot dispatch.

**Modeled as**: each slot has its own `cudaStream` (reused from
StreamFix.tla pattern). Two action variants:

- `DispatchPerSlot(s)` — only slot s's query span is in the forward
  call's batch (`batch_seqs = {s}`). Safe.
- `DispatchSingleGrid(S)` — slots in set S share one forward call
  (`batch_seqs = S`). Models the dangerous shape.

Determinism invariant: a slot's output stream is byte-identical
across runs iff every forward call that produced it had
`batch_seqs = {s}`. The single-grid variant violates this in the
presence of the StreamFix.tla-known kernel asymmetries.

### Family C — Anchor-pos misalignment between block + anchor (single-slot)

**Mechanism**: `DraftBlock.anchor_pos` must equal the `anchor.pos` of
the `AnchorToken` argument to `DraftBlockEmit`. This invariant
(`AnchorPosPreserved`) sounds trivial but is bug-prone in
implementations that track anchor state across multiple variables
(server state vs slot state vs in-flight block state).

If `block.anchor_pos` drifts from `anchor.pos`, downstream:

1. `TargetVerifyBlock` constructs the verify batch starting at
   `block.anchor_pos`, which writes target KV at the wrong cell
   index (`anchor.pos + 1, ...` vs `block.anchor_pos + 1, ...`).
2. `AdvanceState` advances target_kv to `block.anchor_pos +
   n_accepted + 1`, leaving a gap or overlap with the previously-
   committed prefix.
3. The next cycle's anchor reads from the wrong position.

**Source**: spec invariants `AnchorPosPreserved` (DraftBlockEmit),
`InjectedAnchorAlignment` (ProjectAndFuse, between
HiddenFeatures.position and InjectedKV.anchor_pos).

**Modeled as**: state variable `anchor_pos` per slot (single-slot
phase: scalar). Each action that produces a `block` records the
anchor_pos it was emitted at; downstream actions check equality.
Bug-injection flag `AnchorDrift` allows one branch to set
`block.anchor_pos = anchor.pos + 1` (off-by-one drift). The
invariant is then violated reproducibly.

### Family D — KV eviction race at AdvanceState (multi-slot)

**Mechanism**: At end of cycle, `AdvanceState` performs:

- `target_kv[s].n_cells += n_accepted[s] + 1` (extend by accepted +
  bonus)
- `draft_kv[s].evict(rejected_suffix)` (overwrite rejected slots)

Across slots, each slot's KV regions are seq_id-disjoint *by
construction* in vLLM. PHASE45 StreamFix.tla already proved
`NoCrossStreamRace` holds when in-flight write regions are slot-
keyed. We re-verify this property for DFlash's specific KV layout
(target KV per-slot region + drafter KV per-slot region + per-slot
fused-feature injection region).

**Failure mode**: An implementation that lets two slots' AdvanceState
actions interleave with overlapping write regions (e.g. via a shared
scratch buffer mid-eviction, or via aliased KV-cache rebinding)
corrupts both slots' state.

**Source**: spec invariants `PerSlotDraftKVCache`, `SpeculativeCycle-
Atomicity`, `TargetKVAdvancesByAcceptedPlusBonus`.

**Modeled as**: each slot owns `target_kv[s]`, `draft_kv_self[s]`,
`draft_kv_injected[s]`, three disjoint regions. `AdvanceState(s)` is
atomic per slot but may interleave with other slots'
`AdvanceState(s')` — invariant `NoCrossSlotRegionOverlap` must hold
under all interleavings. Reuses StreamFix.tla's `in_flight_writes`
function (now keyed `Slots → SUBSET Regions`).

## §3 Proposed State Variables

### Single-slot phase (DFlashCycle.tla)

| Variable | Type | Description |
|---|---|---|
| `step` | Nat | Cycle counter (0..MaxStep) |
| `pc` | {"draft","verify","accept","advance"} | Program counter — which cycle phase is in flight |
| `anchor_pos` | Nat | Absolute position of current anchor in target sequence |
| `target_kv_n_cells` | Nat | Number of committed KV cells in target cache |
| `draft_kv_self_n_cells` | Nat | Drafter's own KV cells from accepted positions |
| `draft_kv_injected_n_cells` | Nat | Drafter's injected-feature KV cells (1 per active anchor) |
| `n_rejected_prev` | Nat | Carries rejected count from prev cycle's AcceptPrefixDecision |
| `in_flight_block` | OptionalRecord | The DraftBlock currently in flight, or NONE |
| `accept_history` | Seq(Nat) | Per-cycle n_accepted; for liveness checks |

### Multi-slot extension (DFlashMultiSlot.tla)

All single-slot variables become slot-indexed (`[Slots → ...]`). Add:

| Variable | Type | Description |
|---|---|---|
| `stream_status` | [Slots → {"idle","running","done"}] | Per-slot CUDA stream lifecycle (from StreamFix.tla) |
| `in_flight_writes` | [Slots → SUBSET Regions] | KV regions a slot's in-flight stream is writing |
| `batch_seqs_history` | Seq(SUBSET Slots) | For each forward call that ran, which slots were batched together |

## §4 Proposed Actions

### Single-slot (DFlashCycle)

- `Init` — start with anchor_pos=0 (BOS), empty KVs, step=0,
  n_rejected_prev=0, pc="draft".
- `DraftBlockEmit` — pre: pc="draft", anchor live. post: pc="verify",
  in_flight_block populated with anchor_pos and an arbitrary block
  of length BlockSize. (Drafter logits abstracted as "some block".)
- `TargetVerifyBlock` — pre: pc="verify". post: pc="accept",
  produces verify_logits abstracted. Asserts
  `effective_seq_lens = cad.seq_lens - n_rejected_prev`.
- `AcceptPrefixDecision` — pre: pc="accept". post: pc="advance",
  picks `n_accepted ∈ 0..BlockSize` non-deterministically (under-
  spec), emits decision. (Greedy argmax abstracted; the cycle
  structure doesn't care about the actual prefix-match logic.)
- `AdvanceState` — pre: pc="advance". post: pc="draft", step += 1,
  target_kv += n_accepted+1, draft_kv updated, anchor advances,
  n_rejected_prev = BlockSize - n_accepted.

### Multi-slot extension (DFlashMultiSlot)

Per-slot replicas of the above. Plus:

- `DispatchPerSlot(s)` — runs slot s's current pc-action with
  `batch_seqs = {s}`. PerSlotDispatch path.
- `DispatchSingleGrid(S)` — runs all slots in S in one batched
  call with `batch_seqs = S`. SingleGrid path (the dangerous one).
- `StreamComplete(s)` — moves stream_status[s] from "running" to
  "done", commits the slot's in_flight_block to its KV.
- `SyncBarrier` — when all streams idle/done, gates the next step.

## §5 Proposed Invariants

### Safety (single-slot)

- `TypeOK` — type-correctness of every state variable.
- `PcMonotonic` — pc transitions only along the cycle: draft→verify
  →accept→advance→draft (with step++).
- `AnchorAdvancesByAcceptedPlusBonus` — across step k→k+1,
  `anchor_pos' = anchor_pos + n_accepted_k + 1`.
- `TargetKVNotShrinks` — `target_kv_n_cells'` ≥ `target_kv_n_cells`.
- `TargetKVAdvancesByAcceptedPlusBonus` — at AdvanceState,
  `target_kv_n_cells' = target_kv_n_cells + n_accepted + 1`.
- `TargetKVUnchangedDuringVerify` — TargetVerifyBlock does NOT
  mutate target_kv_n_cells.
- `NAcceptedInBounds` — at AcceptPrefixDecision,
  `0 ≤ n_accepted ≤ BlockSize`.
- `RejectionFlowsForward` — `n_rejected_prev` at TargetVerifyBlock
  step k+1 equals `BlockSize - n_accepted_k` from AcceptPrefix-
  Decision step k.
- `EffectiveSeqLensCorrect` — TargetVerifyBlock at step k+1 sees
  `effective_seq_lens = seq_lens - n_rejected_prev` (the gate of
  Family A).
- `BonusPosInvariant` — bonus_pos = block.anchor_pos + n_accepted
  + 1.
- `AnchorPosPreservedThroughCycle` — within one cycle,
  `in_flight_block.anchor_pos = anchor_pos` (the gate of Family C).

### Safety (multi-slot)

All single-slot invariants applied per slot. Plus:

- `NoCrossSlotRegionOverlap` — for any two distinct slots s, t with
  streams running, `in_flight_writes[s] ∩ in_flight_writes[t] = {}`
  (the gate of Family D; lifted from StreamFix.tla `NoCrossStream-
  Race`).
- `PerSlotDispatchOnly` — every forward call (batch_seqs in
  `batch_seqs_history`) satisfies `|batch_seqs| = 1` (the gate of
  Family B). Under `SingleGridDispatch` action mode this is
  violated; under `PerSlotDispatch` it holds.
- `SyncBeforeAdvance` — `step` advances only when
  `\A s: stream_status[s] ∈ {"idle","done"}`.

### Liveness

- `EventuallyCycleProgresses` — `WF` on AdvanceState ensures the
  cycle counter eventually advances if the system isn't stuck.

## §6 Model-Checkable Findings

Claims that should hold or fail predictably under TLC:

### §6.1 Should HOLD (safety property + correct fix)

- **F1**: With `RejectionDropped = FALSE`, `RejectionFlowsForward`
  and `EffectiveSeqLensCorrect` are inductive invariants. TLC
  finds no counterexample in `DFlashCycle.tla`.
- **F2**: With `AnchorDrift = FALSE`,
  `AnchorPosPreservedThroughCycle` holds inductively.
- **F3**: With `DispatchMode = "PerSlot"`, `PerSlotDispatchOnly`
  holds at every state in `DFlashMultiSlot.tla`. TLC explores
  `MaxStep = 4, |Slots| = 3` exhaustively.
- **F4**: `NoCrossSlotRegionOverlap` holds under per-slot dispatch
  for all interleavings.
- **F5**: `SyncBeforeAdvance` holds under the action schema.

### §6.2 Should FAIL (negative controls)

- **N1**: With `RejectionDropped = TRUE` (clear n_rejected_prev at
  AdvanceState instead of preserving), `EffectiveSeqLensCorrect`
  fails — TLC produces a counterexample where the next cycle's
  TargetVerifyBlock pre-condition is violated.
- **N2**: With `AnchorDrift = TRUE` (anchor_pos drift),
  `AnchorPosPreservedThroughCycle` fails.
- **N3**: With `DispatchMode = "SingleGrid"`, `PerSlotDispatch-
  Only` fails. Counterexample: a state where `batch_seqs_history`
  contains a set of size > 1.

The negative controls are critical: they verify that our invariants
have *teeth* (would catch the bugs they claim to bind on), not just
that they're trivially satisfied.

### §6.3 Out-of-scope for this model

- Probabilistic accept (greedy-only at first landing per Allium
  spec `ProbabilisticVerifyOutOfScope`).
- Actual logits / argmax computation (abstracted; cycle structure
  is what we verify, not numerical correctness).
- BF16 / fp16 dtype constraints (not state-machine; tracked in
  Allium invariants).
- Multi-step denoising schedule (declared out of scope by Allium
  spec `OQ-DENOISE-SCHEDULE` — drafter is trained single-step).
- Hybrid target recurrent state (DeltaNet) — modeled as opaque
  "target KV cell" that advances under accept; the recurrent
  state's own ping-pong is SGLang's concern.

## Bounds for TLC

- `MaxStep = 4` — captures at least 4 cycles (sees rejection
  propagation, anchor advancement, KV growth).
- `BlockSize = 3` — small enough for fast exploration, big enough
  for non-trivial accept/reject behaviour (n_accepted ∈ 0..3).
- `|Slots| = 3` for multi-slot phase — captures all slot
  interleaving patterns at minimal blowup.
- `|Regions| = 3 × |Slots|` for in_flight_writes — three KV
  regions per slot (target, draft_self, draft_injected).

## Implementation Pointer

When this TLA work concludes, MD-1.T1 of the ik_llama.cpp port
should:

1. Implement `AdvanceState` so `n_rejected_prev` propagates exactly
   per F1.
2. Construct the verify metadata so `effective_seq_lens` subtracts
   `n_rejected_prev` per F1.
3. Implement multi-slot dispatch as PER-SLOT (not single-grid) per
   F3. Match StreamFix.tla's V2 stream lifecycle.
4. Validate via the property-based tests at
   `ik_llama.cpp/tests/dflash-speculative/` that include the gated
   invariants.

Out-of-scope: tree-K speculation (`NoTreeDraftingNoChainRollout`
invariant rules this out at the spec level).
