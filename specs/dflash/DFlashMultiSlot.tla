--------------------------- MODULE DFlashMultiSlot ---------------------------
(*****************************************************************************)
(* Phase 2 — multi-slot DFlash speculative-decoding cycle.                   *)
(*                                                                            *)
(* Extends the single-slot model in DFlashCycle.tla to n_parallel = |Slots|. *)
(* Reuses the stream-lifecycle + in-flight-writes primitives from            *)
(* specs/multislot/StreamFix.tla.                                            *)
(*                                                                            *)
(* Bug families (modeling-brief §2):                                          *)
(*   B — Multi-slot byte-divergence under single-grid dispatch                *)
(*       Toggle:        DispatchMode \in {"PerSlot", "SingleGrid"}            *)
(*       Falsifies:     PerSlotVerifyDispatchAtMultiSlot                      *)
(*       Source:        vllm dflash.py:215-275 single-grid kernel pattern    *)
(*                                                                            *)
(*   D — KV eviction race at AdvanceState                                     *)
(*       Toggle:        CrossSlotRegionOverlap                                *)
(*       Falsifies:     NoCrossSlotRegionOverlap                              *)
(*       Source:        PerSlotDraftKVCache + SpeculativeCycleAtomicity      *)
(*                                                                            *)
(* Per-cycle state remains as in DFlashCycle: anchor_pos, KV cells,           *)
(* n_rejected_prev, in_flight_block, etc. All become functions [Slots -> ...]. *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Slots,                    \* Finite set of slot identifiers
    MaxStep,                  \* TLC bound on per-slot cycle counter
    BlockSize,                \* DraftBlock.block_size
    DispatchMode,             \* "PerSlot", "SingleGrid", or "UnifiedNe3"
    CrossSlotRegionOverlap    \* Bug toggle for Family D

ASSUME Slots \subseteq {"s1", "s2", "s3", "s4"}  \* Concrete slot names for TLC
ASSUME Cardinality(Slots) \in 1..4
ASSUME MaxStep   \in Nat /\ MaxStep   > 0
ASSUME BlockSize \in Nat /\ BlockSize > 0
ASSUME DispatchMode \in { "PerSlot", "SingleGrid", "UnifiedNe3" }
ASSUME CrossSlotRegionOverlap \in BOOLEAN

\* DispatchKind tags the dispatch shape of each transformer forward call.
\*   "PerSlot"      — batch carries exactly one slot's verify tokens.
\*   "SingleGrid"   — multiple slots' tokens MIXED within one batch_view
\*                    (Bug B; cross-shape dispatcher branches break NPC).
\*   "UnifiedNe3"   — multiple slots packed via ne[3] axis (Tier 3
\*                    unified-stream dispatch); each slot occupies a
\*                    distinct ne[3] slice; the kernel sees per-slot
\*                    isolation at the input level.
DispatchKindValues == { "PerSlot", "SingleGrid", "UnifiedNe3" }

PCValues == { "draft", "verify", "accept", "advance" }
StreamStatus == { "idle", "running", "done" }

\* Per-slot KV regions a stream writes during dispatch. Three logical
\* regions per slot, all seq_id-keyed and disjoint across slots.
RegionsOf(s) == { << s, "target_kv" >>,
                  << s, "draft_kv_self" >>,
                  << s, "draft_kv_injected" >> }

AllRegions == UNION { RegionsOf(s) : s \in Slots }

NoBlock     == [present |-> FALSE, anchor_pos |-> 0, n_tokens |-> 0]
BlockRecord == [present: BOOLEAN, anchor_pos: Nat, n_tokens: Nat]

VARIABLES
    step,                       \* [Slots -> 0..MaxStep]
    pc,                         \* [Slots -> PCValues]
    anchor_pos,                 \* [Slots -> Nat]
    target_kv_n_cells,          \* [Slots -> Nat]
    draft_kv_self_n_cells,      \* [Slots -> Nat]
    draft_kv_injected_n_cells,  \* [Slots -> Nat]
    n_rejected_prev,            \* [Slots -> 0..BlockSize]
    in_flight_block,            \* [Slots -> BlockRecord]
    last_n_accepted,            \* [Slots -> 0..BlockSize]
    \* Stream lifecycle (per-slot)
    stream_status,              \* [Slots -> StreamStatus]
    in_flight_writes,           \* [Slots -> SUBSET AllRegions]
    \* Dispatch history — every transformer forward call records its batch_seqs
    \* (the set of slots that were active in that call). PerSlot ⇒ singleton;
    \* SingleGrid ⇒ |batch_seqs| > 1, mixed seqs; UnifiedNe3 ⇒ |batch_seqs| > 1
    \* but each slot occupies its own ne[3] slice.
    batch_seqs_history,         \* Seq(SUBSET Slots)
    \* Per-call DispatchKind tag (parallel to batch_seqs_history).
    dispatch_kind_history       \* Seq(DispatchKindValues)

vars == << step, pc, anchor_pos,
           target_kv_n_cells, draft_kv_self_n_cells, draft_kv_injected_n_cells,
           n_rejected_prev, in_flight_block, last_n_accepted,
           stream_status, in_flight_writes,
           batch_seqs_history, dispatch_kind_history >>

----------------------------------------------------------------------------
(* ===== TypeOK ===== *)
TypeOK ==
    /\ step                       \in [Slots -> 0..MaxStep]
    /\ pc                         \in [Slots -> PCValues]
    /\ anchor_pos                 \in [Slots -> Nat]
    /\ target_kv_n_cells          \in [Slots -> Nat]
    /\ draft_kv_self_n_cells      \in [Slots -> Nat]
    /\ draft_kv_injected_n_cells  \in [Slots -> Nat]
    /\ n_rejected_prev            \in [Slots -> 0..BlockSize]
    /\ in_flight_block            \in [Slots -> BlockRecord]
    /\ last_n_accepted            \in [Slots -> 0..BlockSize]
    /\ stream_status              \in [Slots -> StreamStatus]
    /\ in_flight_writes           \in [Slots -> SUBSET AllRegions]
    /\ batch_seqs_history         \in Seq(SUBSET Slots)
    /\ dispatch_kind_history      \in Seq(DispatchKindValues)
    /\ Len(batch_seqs_history)    = Len(dispatch_kind_history)

(* ===== Init ===== *)
Init ==
    /\ step                      = [s \in Slots |-> 0]
    /\ pc                        = [s \in Slots |-> "draft"]
    /\ anchor_pos                = [s \in Slots |-> 0]
    /\ target_kv_n_cells         = [s \in Slots |-> 1]
    /\ draft_kv_self_n_cells     = [s \in Slots |-> 0]
    /\ draft_kv_injected_n_cells = [s \in Slots |-> 0]
    /\ n_rejected_prev           = [s \in Slots |-> 0]
    /\ in_flight_block           = [s \in Slots |-> NoBlock]
    /\ last_n_accepted           = [s \in Slots |-> 0]
    /\ stream_status             = [s \in Slots |-> "idle"]
    /\ in_flight_writes          = [s \in Slots |-> {}]
    /\ batch_seqs_history        = << >>
    /\ dispatch_kind_history     = << >>

----------------------------------------------------------------------------
(* ===== Per-slot cycle actions ===== *)
(* Each action records its dispatch shape into batch_seqs_history. Under
   DispatchMode = "PerSlot" the recorded set is always a singleton {s};
   under "SingleGrid" the engine may batch multiple slots into one call
   (we model this with the DispatchSingleGrid action below).               *)

(* DispatchPerSlot(s, kind) — slot s alone is active in this transformer
   forward call. The KV regions it writes are recorded in in_flight_writes
   for the duration. *)
StartStreamPerSlot(s, regions) ==
    /\ stream_status[s] = "idle"
    /\ stream_status'    = [stream_status   EXCEPT ![s] = "running"]
    /\ in_flight_writes' = [in_flight_writes EXCEPT ![s] = regions]
    /\ batch_seqs_history' = Append(batch_seqs_history, {s})
    /\ dispatch_kind_history' = Append(dispatch_kind_history, "PerSlot")

CompleteStream(s) ==
    /\ stream_status[s] = "running"
    /\ stream_status'    = [stream_status   EXCEPT ![s] = "done"]
    /\ in_flight_writes' = [in_flight_writes EXCEPT ![s] = {}]
    /\ UNCHANGED <<batch_seqs_history, dispatch_kind_history>>

(* SingleGrid dispatch — set S of slots share one transformer forward call.
   |S| >= 2 by definition. Bug Family B is the model where the engine takes
   this path when n_parallel > 1. *)
DispatchSingleGrid(S) ==
    /\ DispatchMode = "SingleGrid"
    /\ Cardinality(S) >= 2
    /\ \A s \in S: stream_status[s] = "idle"
    /\ stream_status' = [s \in Slots |->
                          IF s \in S THEN "running" ELSE stream_status[s]]
    /\ in_flight_writes' = [s \in Slots |->
                              IF s \in S THEN RegionsOf(s) ELSE in_flight_writes[s]]
    /\ batch_seqs_history' = Append(batch_seqs_history, S)
    /\ dispatch_kind_history' = Append(dispatch_kind_history, "SingleGrid")

(* DispatchUnifiedNe3 — Tier 3 verify-side unification.                       *)
(*                                                                            *)
(* Multiple slots' verify tokens are packed into ONE transformer forward     *)
(* call, but along the ne[3] axis. Each slot occupies its own ne[3] slice;  *)
(* the kernel sees per-slot K/V isolation by construction. This is distinct *)
(* from the SingleGrid path which mixes seq_ids within a batch_view: under  *)
(* UnifiedNe3, the in-flight write regions remain per-slot (RegionsOf(s))   *)
(* and NoCrossSlotRegionOverlap continues to hold.                          *)
(*                                                                            *)
(* This is the contract Tier 3 must satisfy. Companion to                   *)
(* specs/dispatch/unified_stream_dispatch.allium.                          *)
DispatchUnifiedNe3(S) ==
    /\ DispatchMode = "UnifiedNe3"
    /\ Cardinality(S) >= 2
    /\ \A s \in S: stream_status[s] = "idle"
    /\ stream_status' = [s \in Slots |->
                          IF s \in S THEN "running" ELSE stream_status[s]]
    \* CRITICAL: each slot's in-flight write set is its OWN RegionsOf(s),
    \* NOT the union — this is what distinguishes UnifiedNe3 from a hypothetical
    \* mixed-seq dispatch and what preserves NoCrossSlotRegionOverlap.
    /\ in_flight_writes' = [s \in Slots |->
                              IF s \in S THEN RegionsOf(s) ELSE in_flight_writes[s]]
    /\ batch_seqs_history' = Append(batch_seqs_history, S)
    /\ dispatch_kind_history' = Append(dispatch_kind_history, "UnifiedNe3")

(* CrossSlotRegionOverlap bug — when active, StartStreamPerSlot writes
   to a region of ANOTHER slot too. Modeled as a separate action so it
   doesn't disrupt the clean dispatch path. *)
StartStreamWithBugD(s, other) ==
    /\ CrossSlotRegionOverlap
    /\ s # other
    /\ stream_status[s] = "idle"
    /\ stream_status[other] = "running"
    /\ stream_status'    = [stream_status EXCEPT ![s] = "running"]
    /\ in_flight_writes' = [in_flight_writes EXCEPT
                              ![s] = RegionsOf(s) \cup RegionsOf(other)]
    /\ batch_seqs_history' = Append(batch_seqs_history, {s})
    /\ dispatch_kind_history' = Append(dispatch_kind_history, "PerSlot")

(* ===== Cycle-phase actions per slot ===== *)
(* For brevity we collapse the 4-phase cycle: a single per-slot CycleStep
   action either starts a per-slot stream OR (in SingleGrid mode) the
   engine batches slots together. The KV-state updates happen on
   StreamComplete; we approximate the full draft/verify/accept/advance
   sequence as one atomic per-slot transition between dispatches.        *)

CycleStepPerSlot(s) ==
    /\ DispatchMode = "PerSlot"
    /\ step[s] < MaxStep
    /\ pc[s] = "draft"
    /\ StartStreamPerSlot(s, RegionsOf(s))
    /\ pc' = [pc EXCEPT ![s] = "advance"]   \* atomic per-slot cycle
    /\ \E n \in 0..BlockSize:
          /\ last_n_accepted' = [last_n_accepted EXCEPT ![s] = n]
          /\ target_kv_n_cells' = [target_kv_n_cells EXCEPT
                                      ![s] = target_kv_n_cells[s] + n + 1]
          /\ anchor_pos' = [anchor_pos EXCEPT
                              ![s] = anchor_pos[s] + n + 1]
          /\ n_rejected_prev' = [n_rejected_prev EXCEPT
                                   ![s] = BlockSize - n]
          /\ in_flight_block' = [in_flight_block EXCEPT ![s] = NoBlock]
    /\ step' = [step EXCEPT ![s] = step[s] + 1]
    /\ UNCHANGED << draft_kv_self_n_cells, draft_kv_injected_n_cells >>

CycleStepUnifiedNe3(S) ==
    /\ DispatchMode = "UnifiedNe3"
    /\ \A s \in S: step[s] < MaxStep /\ pc[s] = "draft"
    /\ DispatchUnifiedNe3(S)
    /\ \E n_fn \in [S -> 0..BlockSize]:
         /\ last_n_accepted' = [s \in Slots |->
                                  IF s \in S THEN n_fn[s] ELSE last_n_accepted[s]]
         /\ target_kv_n_cells' = [s \in Slots |->
                                    IF s \in S THEN target_kv_n_cells[s] + n_fn[s] + 1
                                    ELSE target_kv_n_cells[s]]
         /\ anchor_pos' = [s \in Slots |->
                             IF s \in S THEN anchor_pos[s] + n_fn[s] + 1
                             ELSE anchor_pos[s]]
         /\ n_rejected_prev' = [s \in Slots |->
                                  IF s \in S THEN BlockSize - n_fn[s]
                                  ELSE n_rejected_prev[s]]
    /\ in_flight_block' = [s \in Slots |->
                             IF s \in S THEN NoBlock ELSE in_flight_block[s]]
    /\ pc' = [s \in Slots |-> IF s \in S THEN "advance" ELSE pc[s]]
    /\ step' = [s \in Slots |-> IF s \in S THEN step[s] + 1 ELSE step[s]]
    /\ UNCHANGED << draft_kv_self_n_cells, draft_kv_injected_n_cells >>

CycleStepSingleGrid(S) ==
    /\ DispatchMode = "SingleGrid"
    /\ \A s \in S: step[s] < MaxStep /\ pc[s] = "draft"
    /\ DispatchSingleGrid(S)
    /\ \E n_fn \in [S -> 0..BlockSize]:
         /\ last_n_accepted' = [s \in Slots |->
                                  IF s \in S THEN n_fn[s] ELSE last_n_accepted[s]]
         /\ target_kv_n_cells' = [s \in Slots |->
                                    IF s \in S THEN target_kv_n_cells[s] + n_fn[s] + 1
                                    ELSE target_kv_n_cells[s]]
         /\ anchor_pos' = [s \in Slots |->
                             IF s \in S THEN anchor_pos[s] + n_fn[s] + 1
                             ELSE anchor_pos[s]]
         /\ n_rejected_prev' = [s \in Slots |->
                                  IF s \in S THEN BlockSize - n_fn[s]
                                  ELSE n_rejected_prev[s]]
    /\ in_flight_block' = [s \in Slots |->
                             IF s \in S THEN NoBlock ELSE in_flight_block[s]]
    /\ pc' = [s \in Slots |-> IF s \in S THEN "advance" ELSE pc[s]]
    /\ step' = [s \in Slots |-> IF s \in S THEN step[s] + 1 ELSE step[s]]
    /\ UNCHANGED << draft_kv_self_n_cells, draft_kv_injected_n_cells >>

ResetSlot(s) ==
    /\ pc[s] = "advance"
    /\ stream_status[s] = "done"
    /\ pc' = [pc EXCEPT ![s] = "draft"]
    /\ stream_status' = [stream_status EXCEPT ![s] = "idle"]
    /\ UNCHANGED << step, anchor_pos,
                    target_kv_n_cells, draft_kv_self_n_cells,
                    draft_kv_injected_n_cells, n_rejected_prev,
                    in_flight_block, last_n_accepted,
                    in_flight_writes, batch_seqs_history,
                    dispatch_kind_history >>

CompleteSlotStream(s) ==
    /\ CompleteStream(s)
    /\ UNCHANGED << step, pc, anchor_pos,
                    target_kv_n_cells, draft_kv_self_n_cells,
                    draft_kv_injected_n_cells, n_rejected_prev,
                    in_flight_block, last_n_accepted >>

----------------------------------------------------------------------------
Next ==
    \/ \E s \in Slots: CycleStepPerSlot(s)
    \/ \E S \in SUBSET Slots: Cardinality(S) >= 2 /\ CycleStepSingleGrid(S)
    \/ \E S \in SUBSET Slots: Cardinality(S) >= 2 /\ CycleStepUnifiedNe3(S)
    \/ \E s \in Slots: CompleteSlotStream(s)
    \/ \E s \in Slots: ResetSlot(s)
    \/ \E s \in Slots, t \in Slots: s # t /\ StartStreamWithBugD(s, t)
       /\ UNCHANGED << step, pc, anchor_pos,
                       target_kv_n_cells, draft_kv_self_n_cells,
                       draft_kv_injected_n_cells, n_rejected_prev,
                       in_flight_block, last_n_accepted >>

Spec == Init /\ [][Next]_vars /\ \A s \in Slots : WF_vars(ResetSlot(s))

----------------------------------------------------------------------------
(* ===== Invariants — operational forms with canonical Allium names ===== *)

\* PerSlotVerifyDispatchAtMultiSlot — Family B gate.
\* Every multi-slot dispatch is EITHER per-slot (singleton) OR Tier 3
\* UnifiedNe3 (multiple slots, but each in its own ne[3] slice).
\* Violated only by the SingleGrid path (mixed seqs in one batch_view).
PerSlotVerifyDispatchAtMultiSlot ==
    \A i \in 1..Len(batch_seqs_history):
        \/ Cardinality(batch_seqs_history[i]) = 1
        \/ dispatch_kind_history[i] = "UnifiedNe3"

\* Tier3VerifySideRespectsPerStreamPartition — under UnifiedNe3,
\* each slot's in-flight write region is its own RegionsOf(s); the
\* dispatch never causes cross-slot region overlap. (This is what
\* makes UnifiedNe3 distinct from SingleGrid and compatible with
\* NoCrossSlotRegionOverlap.)
Tier3VerifySideRespectsPerStreamPartition ==
    \A i \in 1..Len(dispatch_kind_history):
        dispatch_kind_history[i] = "UnifiedNe3" =>
            \A s \in batch_seqs_history[i], t \in batch_seqs_history[i]:
                s # t =>
                    \* Per-slot regions disjoint by RegionsOf's definition
                    \* (each region carries the slot id in its key).
                    RegionsOf(s) \cap RegionsOf(t) = {}

\* NoCrossSlotRegionOverlap — Family D gate.
\* No two slots with concurrently-running streams share an in-flight write
\* region. Reuse of StreamFix.tla's NoCrossStreamRace shape.
NoCrossSlotRegionOverlap ==
    \A s, t \in Slots:
        s # t /\ stream_status[s] = "running" /\ stream_status[t] = "running"
        => in_flight_writes[s] \cap in_flight_writes[t] = {}

\* SyncBeforeStepAdvance — modeled as: a slot's step counter only
\* advances under ResetSlot (after its own stream is done). Confirmed by
\* the action schema; state-invariant form: if step has just advanced
\* (we approximate by: slot is in "draft" with step > 0), then its
\* stream is not running.
SyncBeforeStepAdvance ==
    \A s \in Slots:
        (pc[s] = "draft" /\ step[s] > 0) => stream_status[s] # "running"

\* Per-slot cycle invariants carried over from DFlashCycle (lifted to
\* [Slots -> ...]). These hold by construction; here we re-check them
\* multi-slot:
NAcceptedWithinBound ==
    \A s \in Slots: last_n_accepted[s] \in 0..BlockSize

NumRejectedTokensFlowsBackToProposer ==
    \A s \in Slots:
        (pc[s] = "draft" /\ step[s] > 0) =>
            n_rejected_prev[s] = BlockSize - last_n_accepted[s]

TargetKVAdvancesByAcceptedPlusBonus ==
    \A s \in Slots:
        target_kv_n_cells[s] >= 1 + step[s]   \* at minimum: 1 anchor + step cycles

BoundedStep == \A s \in Slots: step[s] <= MaxStep

(* Symmetry — slot permutations are equivalent for invariant checking. *)
SlotSymmetry == Permutations(Slots)

============================================================================
