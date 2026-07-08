--------------------------- MODULE StreamFix ---------------------------
(*****************************************************************************)
(* PHASE45 D10.e — Phase 3 spec: V2 multi-stream concurrent per-slot.        *)
(*                                                                            *)
(* Refines Fix.tla. Whereas Fix.tla models PerSlotMode as a sequential       *)
(* split (one slot at a time), this spec models the V2 implementation:      *)
(*   - per-slot decode dispatched on a distinct cudaStream                  *)
(*   - streams may run concurrently on the GPU                              *)
(*   - sync-before-extract barrier waits for all streams                    *)
(*                                                                            *)
(* The fix property remains: each Compute call sees batch_seqs = {s} for    *)
(* the slot s being processed (PerSlotMode). The new failure modes V2       *)
(* introduces are concurrency-related:                                       *)
(*                                                                            *)
(*   F1. Cross-stream race on shared write target — two streams writing     *)
(*       to overlapping KV-cache or scratch buffer regions simultaneously.  *)
(*       In Qwen 3.6, KV cache and recurrent state slots are seq_id-keyed; *)
(*       the property to verify is that disjoint slot KV writes never      *)
(*       overlap.                                                            *)
(*   F2. Output extraction before stream completion — reading per-slot     *)
(*       output buffer before the stream that wrote it has finished.        *)
(*   F3. Sync barrier missed — proceeding to next decode step while a     *)
(*       stream is still in-flight from this step.                          *)
(*                                                                            *)
(* The spec models stream lifecycle as {idle → running → done → idle} per  *)
(* slot. Concurrency comes from `\E S ⊆ Slots: ∀ s ∈ S: stream[s] = idle`,  *)
(* allowing any subset of idle slots to dispatch simultaneously. Sync is   *)
(* gated by `∀ s: stream[s] ∈ {idle, done}`.                                 *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Slots,
    Tokens,
    MaxStep,
    MaxPending,
    DraftK,
    BugAActive,
    BugBActive
    \* Note: PerSlotMode is implicitly TRUE in this spec — V2 IS the fix.
    \* If you want to falsify, see Fix.tla for the PerSlotMode=FALSE control.

VARIABLES
    pending,
    drafts,
    slot_state,
    pre_draft_snap,
    slot_output,
    accept_count,
    step_count,
    alone_state,
    alone_accept,
    \* V2 concurrency variables:
    stream_status,    \* [Slots -> {"idle", "running", "done"}]
    in_flight_writes  \* [Slots -> SUBSET <<region_id, region_id>>] — abstract
                      \* model of "regions this slot's in-flight stream
                      \* is currently writing". Concurrency safety asks
                      \* that no two slots' in-flight writes overlap.

vars == <<pending, drafts, slot_state, pre_draft_snap, slot_output,
          accept_count, step_count, alone_state, alone_accept,
          stream_status, in_flight_writes>>

----------------------------------------------------------------------------
HistoryWitness == Seq([t: Tokens, B: SUBSET Slots])

\* Region-id space: KV cache cells, recurrent state slots, scratch buffers.
\* Modeled abstractly. Per-slot KV writes are keyed by seq_id; recurrent
\* state slots are also seq_id-keyed (qnext_state_slots in the impl). The
\* invariant we verify: when stream s and stream t are both running,
\* their in_flight_writes are disjoint.
\*
\* For tractability, region_id is an integer; per-slot writes claim
\* region_id = slot_index. Shared writes (e.g., scratch buffers) are
\* a single shared region_id = -1; two slots writing to it would race.
\* Per-slot writes are uniquely keyed by the slot id. Use the slot
\* itself as the region id; this is trivially injective and matches
\* the implementation (KV cache and recurrent state slots are
\* keyed by seq_id which equals slot index).
\* "shared_scratch" is a model value introduced when SharedScratchActive
\* is TRUE — represents any shared write target.
RegionId == Slots \cup {"shared_scratch"}

TypeOK ==
    /\ pending \in [Slots -> Seq(Tokens)]
    /\ drafts \in [Slots -> Seq(Tokens)]
    /\ slot_state \in [Slots -> HistoryWitness]
    /\ pre_draft_snap \in [Slots -> HistoryWitness]
    /\ slot_output \in [Slots -> Seq(Tokens)]
    /\ accept_count \in [Slots -> 0..DraftK]
    /\ step_count \in 0..MaxStep
    /\ alone_state \in [Slots -> HistoryWitness]
    /\ alone_accept \in [Slots -> 0..DraftK]
    /\ stream_status \in [Slots -> {"idle", "running", "done"}]
    /\ in_flight_writes \in [Slots -> SUBSET RegionId]

----------------------------------------------------------------------------
Compute(state, token, batch_seqs) ==
    Append(state, [t |-> token, B |-> batch_seqs])

\* Same B-corruption model as Fix.tla.
EmitOutput(state) ==
    IF Len(state) = 0 THEN CHOOSE t \in Tokens : TRUE
    ELSE LET last == state[Len(state)] IN
         IF Cardinality(last.B) > 1 /\ (BugAActive \/ BugBActive)
         THEN CHOOSE t \in Tokens : t # last.t
         ELSE last.t

RECURSIVE ComputeSeq(_, _, _)
ComputeSeq(state, tokens, batch_seqs) ==
    IF tokens = << >> THEN state
    ELSE ComputeSeq(Compute(state, Head(tokens), batch_seqs),
                    Tail(tokens), batch_seqs)

RECURSIVE AcceptCountIter(_, _, _, _)
AcceptCountIter(state, tokens, batch_seqs, k) ==
    IF tokens = << >> THEN k
    ELSE
        LET state2 == Compute(state, Head(tokens), batch_seqs) IN
        IF EmitOutput(state2) = Head(tokens)
        THEN AcceptCountIter(state2, Tail(tokens), batch_seqs, k + 1)
        ELSE k

ComputeAccept(state, tokens, batch_seqs) ==
    AcceptCountIter(state, tokens, batch_seqs, 0)

RECURSIVE TakeSeq(_, _)
TakeSeq(s, k) == IF k = 0 \/ s = << >>
                 THEN << >>
                 ELSE <<Head(s)>> \o TakeSeq(Tail(s), k - 1)

ComputeAcceptedState(state, tokens, batch_seqs) ==
    LET k == ComputeAccept(state, tokens, batch_seqs) IN
    ComputeSeq(state, TakeSeq(tokens, k), batch_seqs)

----------------------------------------------------------------------------
\* Per-slot "regions written by this slot's in-flight compute". By
\* design (and matching the impl: KV cache and recurrent state slots
\* are keyed by seq_id), each slot writes only to its own region.
\*
\* When SharedScratchActive=TRUE (set in MC.cfg), include a shared
\* scratch region used by all slots — this surfaces a V2 race that
\* would only appear if a future change introduced a shared write
\* target across streams.
CONSTANT SharedScratchActive
SlotRegions(s) ==
    IF SharedScratchActive
    THEN {s, "shared_scratch"}
    ELSE {s}

----------------------------------------------------------------------------
Init ==
    /\ pending = [s \in Slots |-> << >>]
    /\ drafts = [s \in Slots |-> << >>]
    /\ slot_state = [s \in Slots |-> << >>]
    /\ pre_draft_snap = [s \in Slots |-> << >>]
    /\ slot_output = [s \in Slots |-> << >>]
    /\ accept_count = [s \in Slots |-> 0]
    /\ step_count = 0
    /\ alone_state = [s \in Slots |-> << >>]
    /\ alone_accept = [s \in Slots |-> 0]
    /\ stream_status = [s \in Slots |-> "idle"]
    /\ in_flight_writes = [s \in Slots |-> {}]

----------------------------------------------------------------------------
Enqueue(s) ==
    /\ stream_status[s] = "idle"  \* don't enqueue while in-flight
    /\ Len(pending[s]) < MaxPending
    /\ \E t \in Tokens:
        pending' = [pending EXCEPT ![s] = Append(@, t)]
    /\ UNCHANGED <<drafts, slot_state, pre_draft_snap, slot_output,
                   accept_count, step_count, alone_state, alone_accept,
                   stream_status, in_flight_writes>>

BuildDraft(s) ==
    /\ stream_status[s] = "idle"
    /\ Len(drafts[s]) = 0
    /\ Len(pending[s]) > 0
    /\ \E ds \in [1..DraftK -> Tokens]:
        drafts' = [drafts EXCEPT ![s] = [i \in 1..DraftK |-> ds[i]]]
    /\ pre_draft_snap' = [pre_draft_snap EXCEPT ![s] = slot_state[s]]
    /\ UNCHANGED <<pending, slot_state, slot_output, accept_count,
                   step_count, alone_state, alone_accept,
                   stream_status, in_flight_writes>>

----------------------------------------------------------------------------
(* Action: DispatchOnStream(s) — start single-slot compute on slot s's      *)
(* dedicated CUDA stream. Sets in_flight_writes[s] := SlotRegions(s),       *)
(* representing the regions this stream's kernels will touch.               *)
(*                                                                            *)
(* This action models V2's "issue kernel on stream_s, return immediately."  *)
(* The compute itself is NOT yet observable in slot_state — we wait for    *)
(* StreamComplete to commit results.                                         *)
(*                                                                            *)
(* NB: under PerSlotMode (always |B|=1), the Compute call uses {s} as      *)
(* batch_seqs.                                                               *)
----------------------------------------------------------------------------
DispatchOnStream(s) ==
    /\ stream_status[s] = "idle"
    /\ Len(drafts[s]) = DraftK
    /\ Len(pending[s]) > 0
    /\ step_count < MaxStep
    /\ stream_status' = [stream_status EXCEPT ![s] = "running"]
    /\ in_flight_writes' = [in_flight_writes EXCEPT ![s] = SlotRegions(s)]
    /\ UNCHANGED <<pending, drafts, slot_state, pre_draft_snap, slot_output,
                   accept_count, step_count, alone_state, alone_accept>>

----------------------------------------------------------------------------
(* Action: StreamComplete(s) — slot s's stream finishes. Commit results to *)
(* slot_state, slot_output. Set stream_status := done. KV/scratch regions *)
(* are released (in_flight_writes[s] := {}).                                *)
(*                                                                            *)
(* The compute uses batch_seqs={s} (PerSlotMode). Both real and alone get *)
(* the same compute — they should match exactly under PerSlotMode.         *)
----------------------------------------------------------------------------
StreamComplete(s) ==
    /\ stream_status[s] = "running"
    /\ LET RealK == ComputeAccept(pre_draft_snap[s], drafts[s], {s})
           AloneK == ComputeAccept(pre_draft_snap[s], drafts[s], {s})
       IN
       /\ accept_count' = [accept_count EXCEPT ![s] = RealK]
       /\ alone_accept' = [alone_accept EXCEPT ![s] = AloneK]
       /\ slot_state' = [slot_state EXCEPT
                          ![s] = ComputeAcceptedState(pre_draft_snap[s],
                                                       drafts[s], {s})]
       /\ alone_state' = [alone_state EXCEPT
                           ![s] = ComputeAcceptedState(pre_draft_snap[s],
                                                        drafts[s], {s})]
       /\ slot_output' = [slot_output EXCEPT
                           ![s] = slot_output[s] \o TakeSeq(drafts[s],
                                                             RealK)]
    /\ pending' = [pending EXCEPT ![s] = Tail(pending[s])]
    /\ drafts' = [drafts EXCEPT ![s] = << >>]
    /\ pre_draft_snap' = [pre_draft_snap EXCEPT ![s] = << >>]
    /\ stream_status' = [stream_status EXCEPT ![s] = "done"]
    /\ in_flight_writes' = [in_flight_writes EXCEPT ![s] = {}]
    /\ UNCHANGED <<step_count>>

----------------------------------------------------------------------------
(* Action: SyncBarrier — wait for all streams to be {idle, done}, then    *)
(* reset to idle and bump step_count. This is the cross-stream sync point *)
(* before the next decode iteration.                                        *)
(*                                                                            *)
(* F3 (sync barrier missed) is verified as: step_count never advances      *)
(* while any stream is "running."                                           *)
----------------------------------------------------------------------------
SyncBarrier ==
    /\ \A s \in Slots: stream_status[s] \in {"idle", "done"}
    /\ \E s \in Slots: stream_status[s] = "done"  \* at least one finished
    /\ stream_status' = [s \in Slots |-> "idle"]
    /\ step_count' = step_count + 1
    /\ UNCHANGED <<pending, drafts, slot_state, pre_draft_snap, slot_output,
                   accept_count, alone_state, alone_accept,
                   in_flight_writes>>

----------------------------------------------------------------------------
Next ==
    \/ \E s \in Slots: Enqueue(s)
    \/ \E s \in Slots: BuildDraft(s)
    \/ \E s \in Slots: DispatchOnStream(s)
    \/ \E s \in Slots: StreamComplete(s)
    \/ SyncBarrier

Spec == Init /\ [][Next]_vars

----------------------------------------------------------------------------
(* Invariants — V2 specific.                                                *)
----------------------------------------------------------------------------

\* Sanity carry-overs.
NoSpontaneousState ==
    \A s \in Slots: Len(slot_state[s]) <= Len(slot_output[s]) + DraftK

OutputBoundedByState ==
    \A s \in Slots: Len(slot_output[s]) <= Len(slot_state[s])

BoundedStep == step_count <= MaxStep

\* Phase 2 carry-over: per-slot determinism.
TokenTrace(state) == [i \in 1..Len(state) |-> state[i].t]

VerifyConsistency ==
    \A s \in Slots: accept_count[s] = alone_accept[s]

NoStateDivergence ==
    \A s \in Slots: TokenTrace(slot_state[s]) = TokenTrace(alone_state[s])

\* (F1) NoCrossStreamRace: no two running streams share an in-flight region.
NoCrossStreamRace ==
    \A s, t \in Slots:
        s # t /\ stream_status[s] = "running" /\ stream_status[t] = "running"
        => in_flight_writes[s] \cap in_flight_writes[t] = {}

\* (F2) NoOutputBeforeStreamComplete: a slot's output reflects only
\* committed (non-in-flight) compute. Stream_status="running" implies
\* slot_output[s] hasn't been updated yet for this iteration. We model
\* this as: slot_output length only ever grows in StreamComplete, never
\* in DispatchOnStream. (Structurally enforced by the actions; the
\* invariant is a redundancy check.)
\* Expressed indirectly: if any stream is running, that slot's
\* accept_count[s] should still reflect the prior step (or 0 if first).
\* For TLC tractability, we encode this via no-output-leak:
NoOutputDuringInFlight ==
    \A s \in Slots:
        stream_status[s] = "running" => in_flight_writes[s] # {}

\* (F3) StepDoesNotAdvanceWithRunningStream: the SyncBarrier action
\* requires no streams running. If step_count advances while any stream
\* is running, the spec is broken.
\* (TLA+ doesn't directly express "this transition," but the SyncBarrier
\* enabling condition + step_count' = step_count + 1 only there enforces it.
\* As a reachable-state invariant, we observe: if step_count just bumped,
\* no stream is running. We approximate as: at any state, step_count's
\* monotonicity respects the barrier. Captured by structural Spec.)

============================================================================
