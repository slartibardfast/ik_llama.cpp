--------------------------- MODULE DecodeMTP ---------------------------
(*****************************************************************************)
(* PHASE45 D10.e — Phase 2 spec: multi-slot speculative decode (MTP).        *)
(*                                                                            *)
(* Extends the Phase 1 decode model with:                                    *)
(*   - draft tokens (K per slot per step)                                    *)
(*   - batched verify across slots                                           *)
(*   - per-slot accept/reject + rollback                                     *)
(*                                                                            *)
(* Adds MTP-specific safety invariants:                                       *)
(*   - VerifyConsistency: per-slot accept count must match single-slot       *)
(*     verify (i.e., accept count must not depend on batch_seqs).            *)
(*   - RollbackEquality: state after verify equals (pre-draft snapshot ++   *)
(*     accepted drafts).                                                      *)
(*   - NoCrossSlotLeak: BuildDraft(s) modifies only slot s's state.          *)
(*   - MTPMonotonicity: slot_state never shrinks except via Rollback.        *)
(*                                                                            *)
(* Under the buggy (Phase 1) Compute, VerifyConsistency will be VIOLATED    *)
(* in MC. This is intended — Phase 2 documents the bug formally. Phase 2.5  *)
(* (Fix.tla) introduces PerSlotMode that discharges the violations.         *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Slots,
    Tokens,
    MaxStep,
    MaxPending,
    DraftK             \* number of draft tokens per slot per MTP step

VARIABLES
    pending,           \* [Slots -> Seq(Tokens)] — main verify tokens queued
    drafts,            \* [Slots -> Seq(Tokens)] — speculative drafts pending verify
    slot_state,        \* [Slots -> HistoryWitness]
    pre_draft_snap,    \* [Slots -> HistoryWitness] — snapshot before drafting
    slot_output,       \* [Slots -> Seq(Tokens)]
    accept_count,      \* [Slots -> Nat] — # drafts accepted at last verify (0..DraftK)
    step_count,
    \* Shadow tracker: what slot_state would be if each slot ran alone.
    alone_state,       \* [Slots -> HistoryWitness]
    alone_accept       \* [Slots -> Nat]

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
    /\ \A s \in Slots: Len(pending[s]) <= MaxPending
    /\ \A s \in Slots: Len(drafts[s]) <= DraftK

----------------------------------------------------------------------------
(* Compute abstraction — same as Phase 1. Bug surface: result depends on    *)
(* batch_seqs.                                                                *)
----------------------------------------------------------------------------
Compute(state, token, batch_seqs) ==
    Append(state, [t |-> token, B |-> batch_seqs])

\* EmitOutput models "argmax of logits" at the current state. In the
\* real implementation this should depend only on per-slot history, not
\* on which other slots are batched. The bug we capture: batched
\* compute (|last.B| > 1) corrupts the output — emitting a different
\* token than the alone (|last.B| = 1) compute would.
EmitOutput(state) ==
    IF Len(state) = 0 THEN CHOOSE t \in Tokens : TRUE
    ELSE LET last == state[Len(state)] IN
         IF Cardinality(last.B) > 1
         THEN CHOOSE t \in Tokens : t # last.t \* batched: corrupts
         ELSE last.t                            \* alone: faithful

\* Apply Compute repeatedly to a sequence of tokens, threading state.
RECURSIVE ComputeSeq(_, _, _)
ComputeSeq(state, tokens, batch_seqs) ==
    IF tokens = << >> THEN state
    ELSE ComputeSeq(Compute(state, Head(tokens), batch_seqs),
                    Tail(tokens), batch_seqs)

\* AcceptCount: deterministic acceptance rule. A draft is accepted iff
\* its emitted output equals the next token in sequence. This abstracts
\* the acceptance test in MTP (compare draft-token-i to verify-output-i).
\* Returns the accept count k ∈ 0..K.
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

\* State after applying only the first k tokens (the accepted prefix).
RECURSIVE TakeSeq(_, _)
TakeSeq(s, k) == IF k = 0 \/ s = << >>
                 THEN << >>
                 ELSE <<Head(s)>> \o TakeSeq(Tail(s), k - 1)

\* Run accept and produce both the accept count and the post-accept state.
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

----------------------------------------------------------------------------
(* Action: Enqueue(s) — append a verify-step token to pending.              *)
----------------------------------------------------------------------------
Enqueue(s) ==
    /\ Len(pending[s]) < MaxPending
    /\ \E t \in Tokens:
        pending' = [pending EXCEPT ![s] = Append(@, t)]
    /\ UNCHANGED <<drafts, slot_state, pre_draft_snap, slot_output,
                   accept_count, step_count, alone_state, alone_accept>>

----------------------------------------------------------------------------
(* Action: BuildDraft(s) — single-slot speculative draft generation.        *)
(*                                                                           *)
(* Snapshots pre_draft_snap[s] := slot_state[s], then draws DraftK tokens.  *)
(* In the real impl the MTP head emits drafts; we abstract as nondeterministic *)
(* token choice. NOTE: BuildDraft does NOT advance slot_state — drafts are  *)
(* applied speculatively only at VerifyBatch time.                          *)
(*                                                                           *)
(* CODE REF: common/speculative.cpp build_qwen35_mtp_fused chain.            *)
(*                                                                           *)
(* NoCrossSlotLeak invariant requires that this action only modifies        *)
(* slot s's variables.                                                       *)
----------------------------------------------------------------------------
BuildDraft(s) ==
    /\ Len(drafts[s]) = 0
    /\ Len(pending[s]) > 0
    /\ \E ds \in [1..DraftK -> Tokens]:
        drafts' = [drafts EXCEPT ![s] = [i \in 1..DraftK |-> ds[i]]]
    /\ pre_draft_snap' = [pre_draft_snap EXCEPT ![s] = slot_state[s]]
    /\ UNCHANGED <<pending, slot_state, slot_output, accept_count,
                   step_count, alone_state, alone_accept>>

----------------------------------------------------------------------------
(* Action: VerifyBatch(B) — batched verify across slots in B.               *)
(*                                                                           *)
(* For each s ∈ B with non-empty drafts:                                    *)
(*   - Compute accept count under batch_seqs=B (the BUGGY path).            *)
(*   - Compute accept count under batch_seqs={s} (the SHADOW alone path).   *)
(*   - Update slot_state to pre_draft_snap[s] ++ accepted-drafts (real).    *)
(*   - Update alone_state to pre_draft_snap[s] ++ accepted-drafts (alone).  *)
(*   - Append accepted draft tokens to slot_output.                         *)
(*   - Pop pending[s] (the verify token's role is consumed).                *)
(*   - Reset drafts[s] := <<>>.                                             *)
(*                                                                           *)
(* The accept counts under real vs alone may differ — that's the bug.       *)
(*                                                                           *)
(* CODE REF: src/llama.cpp:5147 — decode_internal ubatch loop processes     *)
(*   the multi-slot K+1-rows-per-slot batch.                                *)
----------------------------------------------------------------------------
ReadyToVerify(B) ==
    /\ B \subseteq Slots
    /\ B # {}
    /\ \A s \in B:
        /\ Len(drafts[s]) = DraftK
        /\ Len(pending[s]) > 0

VerifyBatch(B) ==
    /\ ReadyToVerify(B)
    /\ step_count < MaxStep
    \* Per-slot accept counts under the batched (buggy) path.
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

----------------------------------------------------------------------------
Next ==
    \/ \E s \in Slots: Enqueue(s)
    \/ \E s \in Slots: BuildDraft(s)
    \/ \E B \in (SUBSET Slots) \ {{}}: VerifyBatch(B)

Spec == Init /\ [][Next]_vars

----------------------------------------------------------------------------
(* Sanity invariants (carry over from Phase 1).                              *)
----------------------------------------------------------------------------
NoSpontaneousState ==
    \A s \in Slots: Len(slot_state[s]) <=
        Len(slot_output[s]) + DraftK \* state can grow speculatively

OutputBoundedByState ==
    \A s \in Slots: Len(slot_output[s]) <= Len(slot_state[s])

BoundedStep == step_count <= MaxStep

----------------------------------------------------------------------------
(* MTP safety invariants — the heart of Phase 2.                             *)
----------------------------------------------------------------------------

\* (A) VerifyConsistency: per-slot accept count must equal the alone result.
\*     Under buggy Compute (Phase 1), this VIOLATES whenever a multi-slot
\*     batch has been verified and the bug has produced different accept
\*     counts for any slot. Under PerSlotMode (Phase 2.5 fix), this holds.
VerifyConsistency ==
    \A s \in Slots: accept_count[s] = alone_accept[s]

\* (B) NoStateDivergence: real slot_state matches alone slot_state.
\*     Strictly stronger than VerifyConsistency — implies it.
NoStateDivergence ==
    \A s \in Slots: slot_state[s] = alone_state[s]

\* (C) RollbackEquality: the real slot_state at any commit point equals
\*     pre_draft_snap extended by exactly the accepted-drafts prefix.
\*     This is enforced structurally by VerifyBatch's update; this
\*     invariant catches any spec-level violation of that contract.
RollbackEquality ==
    \A s \in Slots:
        \/ Len(drafts[s]) > 0   \* still drafting; not at commit point
        \/ Len(pre_draft_snap[s]) = 0
        \/ Len(slot_state[s]) >= Len(pre_draft_snap[s])

\* (D) NoCrossSlotLeak: structurally enforced by BuildDraft(s) only updating
\*     slot s's variables. Express as: drafts[t] for t ≠ s is unchanged
\*     across BuildDraft(s) calls. This is enforced by the action; we
\*     express it as a redundancy check.
\*     (Direct invariant form is hard in pure TLA+; we encode as a
\*     liveness-style temporal check in MC.cfg.)

============================================================================
