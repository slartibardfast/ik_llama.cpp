--------------------------- MODULE Fix ---------------------------
(*****************************************************************************)
(* PHASE45 D10.e — Phase 2.5 spec: per-slot dispatch fix.                    *)
(*                                                                            *)
(* Models the proposed architectural fix: opt-in `PerSlotMode` that          *)
(* restricts VerifyBatch to single-slot batches (|B| = 1).                  *)
(*                                                                            *)
(* Council refinements (Olafsson + Patel):                                   *)
(*   - BugAActive / BugBActive parameters: gate the corruption surface for  *)
(*     partial-fix analysis. Bug A = DeltaNet wrapper-level; Bug B = FA     *)
(*     kernel-internal. Empirically both fire; the spec lets us evaluate    *)
(*     fix combinations.                                                     *)
(*   - Honest abstraction note: this spec is a CONTRACT VALIDATOR, not a   *)
(*     bug discoverer. Empirical D10.e.0.L data establishes bug existence; *)
(*     this spec proves PerSlotMode is structurally sufficient to satisfy   *)
(*     the determinism contract.                                             *)
(*                                                                            *)
(* Verification matrix:                                                       *)
(*   PerSlotMode | BugA | BugB | Expected                                   *)
(*   ------------+------+------+------------------------------              *)
(*   FALSE       | F    | F    | Pass (no bug present)                       *)
(*   FALSE       | T    | F    | Violate (Bug A alone is sufficient)         *)
(*   FALSE       | F    | T    | Violate (Bug B alone is sufficient)         *)
(*   FALSE       | T    | T    | Violate (both fire)                         *)
(*   TRUE        | T    | T    | Pass (per-slot dispatch suppresses both)    *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Slots,
    Tokens,
    MaxStep,
    MaxPending,
    DraftK,
    PerSlotMode,
    BugAActive,    \* DeltaNet wrapper batch-shape divergence active?
    BugBActive     \* FA kernel-internal cross-row asymmetry active?

VARIABLES
    pending,
    drafts,
    slot_state,
    pre_draft_snap,
    slot_output,
    accept_count,
    step_count,
    alone_state,
    alone_accept

vars == <<pending, drafts, slot_state, pre_draft_snap, slot_output,
          accept_count, step_count, alone_state, alone_accept>>

----------------------------------------------------------------------------
HistoryWitness == Seq([t: Tokens, B: SUBSET Slots])

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

----------------------------------------------------------------------------
(* Compute and EmitOutput.                                                   *)
(*                                                                           *)
(* The bug surfaces when Cardinality(last.B) > 1 AND a relevant bug knob is *)
(* active. We model "corruption" as 'EmitOutput selects a different token  *)
(* than the alone path would.' This is an over-approximation of the real   *)
(* FP non-determinism (which only sometimes corrupts) — sound for proving  *)
(* sufficiency of the fix, since if the spec PASSES under PerSlotMode=TRUE *)
(* with this strong corruption model, it also passes under any weaker     *)
(* real-world model.                                                         *)
(*****************************************************************************)
Compute(state, token, batch_seqs) ==
    Append(state, [t |-> token, B |-> batch_seqs])

\* Bug active iff at least one of the bug knobs is on AND the batch is multi.
EmitOutput(state) ==
    IF Len(state) = 0 THEN CHOOSE t \in Tokens : TRUE
    ELSE LET last == state[Len(state)] IN
         IF Cardinality(last.B) > 1 /\ (BugAActive \/ BugBActive)
         THEN CHOOSE t \in Tokens : t # last.t  \* corrupts under buggy batched
         ELSE last.t                            \* faithful otherwise

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

Enqueue(s) ==
    /\ Len(pending[s]) < MaxPending
    /\ \E t \in Tokens:
        pending' = [pending EXCEPT ![s] = Append(@, t)]
    /\ UNCHANGED <<drafts, slot_state, pre_draft_snap, slot_output,
                   accept_count, step_count, alone_state, alone_accept>>

BuildDraft(s) ==
    /\ Len(drafts[s]) = 0
    /\ Len(pending[s]) > 0
    /\ \E ds \in [1..DraftK -> Tokens]:
        drafts' = [drafts EXCEPT ![s] = [i \in 1..DraftK |-> ds[i]]]
    /\ pre_draft_snap' = [pre_draft_snap EXCEPT ![s] = slot_state[s]]
    /\ UNCHANGED <<pending, slot_state, slot_output, accept_count,
                   step_count, alone_state, alone_accept>>

ReadyToVerify(B) ==
    /\ B \subseteq Slots
    /\ B # {}
    /\ (~PerSlotMode \/ Cardinality(B) = 1)
    /\ \A s \in B:
        /\ Len(drafts[s]) = DraftK
        /\ Len(pending[s]) > 0

VerifyBatch(B) ==
    /\ ReadyToVerify(B)
    /\ step_count < MaxStep
    /\ LET RealK == [s \in B |-> ComputeAccept(pre_draft_snap[s],
                                                drafts[s], B)]
           AloneK == [s \in B |-> ComputeAccept(pre_draft_snap[s],
                                                 drafts[s], {s})]
       IN
       /\ accept_count' = [s \in Slots |->
                            IF s \in B THEN RealK[s] ELSE accept_count[s]]
       /\ alone_accept' = [s \in Slots |->
                             IF s \in B THEN AloneK[s] ELSE alone_accept[s]]
       /\ slot_state' = [s \in Slots |->
                          IF s \in B
                          THEN ComputeAcceptedState(pre_draft_snap[s],
                                                     drafts[s], B)
                          ELSE slot_state[s]]
       /\ alone_state' = [s \in Slots |->
                           IF s \in B
                           THEN ComputeAcceptedState(pre_draft_snap[s],
                                                      drafts[s], {s})
                           ELSE alone_state[s]]
       /\ slot_output' = [s \in Slots |->
                           IF s \in B
                           THEN slot_output[s] \o TakeSeq(drafts[s],
                                                           RealK[s])
                           ELSE slot_output[s]]
    /\ pending' = [s \in Slots |->
                    IF s \in B THEN Tail(pending[s]) ELSE pending[s]]
    /\ drafts' = [s \in Slots |->
                   IF s \in B THEN << >> ELSE drafts[s]]
    /\ pre_draft_snap' = [s \in Slots |->
                            IF s \in B THEN << >> ELSE pre_draft_snap[s]]
    /\ step_count' = step_count + 1

Next ==
    \/ \E s \in Slots: Enqueue(s)
    \/ \E s \in Slots: BuildDraft(s)
    \/ \E B \in (SUBSET Slots) \ {{}}: VerifyBatch(B)

Spec == Init /\ [][Next]_vars

----------------------------------------------------------------------------
NoSpontaneousState ==
    \A s \in Slots: Len(slot_state[s]) <= Len(slot_output[s]) + DraftK

OutputBoundedByState ==
    \A s \in Slots: Len(slot_output[s]) <= Len(slot_state[s])

BoundedStep == step_count <= MaxStep

VerifyConsistency ==
    \A s \in Slots: accept_count[s] = alone_accept[s]

\* Project state to its observable token sequence; the B-witness field is
\* a modeling artifact (tracks which batch the step was computed under),
\* not a runtime-observable property. Determinism is about tokens, not
\* about which batch they were computed in.
TokenTrace(state) == [i \in 1..Len(state) |-> state[i].t]

NoStateDivergence ==
    \A s \in Slots: TokenTrace(slot_state[s]) = TokenTrace(alone_state[s])

============================================================================
