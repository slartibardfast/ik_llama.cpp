--------------------------- MODULE Decode ---------------------------
(*****************************************************************************)
(* PHASE45 D10.e — Phase 1 spec: ik_llama.cpp batched plain decode.          *)
(*                                                                            *)
(* Category B (concurrent / runtime). Captures the CURRENT batched           *)
(* multi-slot decode path faithfully — including the bug. The bug is         *)
(* expressed as `Compute(state, token, batch_seqs)` having a non-trivial     *)
(* dependence on `batch_seqs` even when restricted to slot s's view.         *)
(*                                                                            *)
(* This phase has NO safety properties — it is a faithful capture, not       *)
(* a verification. Phase 2 (DecodeMTP.tla) adds MTP draft/verify and the     *)
(* safety invariants. Phase 2.5 (Fix.tla) adds the PerSlotMode fix.          *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Slots,        \* set of slot ids (e.g. {0, 1, 2})
    Tokens,       \* set of distinct token symbols (abstract)
    MaxStep,      \* bound on global step counter for finite MC
    MaxPending    \* bound on |pending[s]| for finite MC

VARIABLES
    pending,      \* [Slots -> Seq(Tokens)] — input tokens queued for decode
    slot_state,   \* [Slots -> Seq(...)] — history witness; opaque growing record
    slot_output,  \* [Slots -> Seq(Tokens)] — emitted output tokens
    step_count    \* nat — global advancement counter

vars == <<pending, slot_state, slot_output, step_count>>

----------------------------------------------------------------------------
(* Types and helpers. *)
----------------------------------------------------------------------------

\* HistoryWitness is an opaque sequence-valued record. We model it as a
\* sequence of records [t |-> token, B |-> batch_seqs_set]. Slots whose
\* history-witness sequences differ are observably non-equivalent.
HistoryWitness == Seq([t: Tokens, B: SUBSET Slots])

TypeOK ==
    /\ pending \in [Slots -> Seq(Tokens)]
    /\ slot_state \in [Slots -> HistoryWitness]
    /\ slot_output \in [Slots -> Seq(Tokens)]
    /\ step_count \in 0..MaxStep
    /\ \A s \in Slots: Len(pending[s]) <= MaxPending

----------------------------------------------------------------------------
(* The Compute abstraction.                                                  *)
(*                                                                           *)
(* Compute(state, token, batch_seqs) models the layer-by-layer forward pass *)
(* through the model for slot s, where:                                      *)
(*   - state: slot s's history witness before this step                      *)
(*   - token: the input token for slot s at this step                        *)
(*   - batch_seqs: the set of slot ids that share this batch with s          *)
(*                                                                           *)
(* In the real implementation, the per-slot output should depend ONLY on    *)
(* (state, token). In the buggy reality of ik_llama batched dispatch, it    *)
(* depends on (state, token, batch_seqs) — this is the bug.                 *)
(*                                                                           *)
(* We capture this by appending a record [t |-> token, B |-> batch_seqs]    *)
(* to the history witness. Two distinct batch_seqs values produce distinct  *)
(* witness sequences; downstream determinism checks see them as different.  *)
(*                                                                           *)
(* CODE REFS:                                                                *)
(*   src/llama-delta-net.cpp:679-733 — DeltaNet wrapper dispatches by       *)
(*     all_same_seq (Bug A). Even forced always-blocks-path does not match  *)
(*     M=1 (D10.e.0.P spike).                                                *)
(*   ggml/src/ggml-cuda/fattn-mma-f16.cu:154-174 — FA tier dispatch hits    *)
(*     same template instance for M=1, M=2 (Bug B). Within-block cross-row *)
(*     reduction makes per-row output position-dependent.                   *)
(*****************************************************************************)
Compute(state, token, batch_seqs) ==
    Append(state, [t |-> token, B |-> batch_seqs])

\* Output emitted from a compute step. Modeled as a function of the
\* full updated history witness. Same bug surface — output depends on
\* batch_seqs via the witness record.
\* EmitOutput models "argmax of logits" at the current state. The bug:
\* batched compute (|last.B| > 1) corrupts the output — emitting a
\* different token than the alone (|last.B| = 1) compute would.
EmitOutput(state) ==
    IF Len(state) = 0 THEN CHOOSE t \in Tokens : TRUE
    ELSE LET last == state[Len(state)] IN
         IF Cardinality(last.B) > 1
         THEN CHOOSE t \in Tokens : t # last.t  \* batched: corrupts
         ELSE last.t                            \* alone: faithful

----------------------------------------------------------------------------
(* Init.                                                                     *)
----------------------------------------------------------------------------
Init ==
    /\ pending = [s \in Slots |-> << >>]
    /\ slot_state = [s \in Slots |-> << >>]
    /\ slot_output = [s \in Slots |-> << >>]
    /\ step_count = 0

----------------------------------------------------------------------------
(* Action: Enqueue(s).                                                       *)
(*                                                                           *)
(* Models the upstream scheduler appending an input token for slot s's      *)
(* next decode step. Bounded by MaxPending.                                  *)
(*                                                                           *)
(* CODE REF: server.cpp slot scheduling enqueues tokens after sampling.    *)
----------------------------------------------------------------------------
Enqueue(s) ==
    /\ Len(pending[s]) < MaxPending
    /\ \E t \in Tokens:
        /\ pending' = [pending EXCEPT ![s] = Append(@, t)]
    /\ UNCHANGED <<slot_state, slot_output, step_count>>

----------------------------------------------------------------------------
(* Action: ProcessBatch(B).                                                  *)
(*                                                                           *)
(* B ⊆ Slots, |B| ≥ 1; every slot in B has a pending token.                *)
(*                                                                           *)
(* For each s ∈ B:                                                          *)
(*   - Pop head(pending[s])                                                 *)
(*   - Update slot_state[s] using Compute(state, token, B). The dependence *)
(*     on B is the bug.                                                      *)
(*   - Append output to slot_output[s].                                     *)
(*                                                                           *)
(* CODE REF: src/llama.cpp:5147 — llama_decode_internal ubatch loop.       *)
(*   Calls into the graph builder once per ubatch with all active slots'   *)
(*   tokens. The graph builder dispatches per-layer; DeltaNet at il>=1 and *)
(*   FA at std-attn layers exhibit the bug.                                *)
(*                                                                           *)
(* ATOMICITY: this action is atomic at our abstraction level. Splitting     *)
(* into load/check/store is unnecessary because we are not modeling cross- *)
(* slot kernel races — the bug is determinism across batch shapes, not     *)
(* between threads.                                                          *)
----------------------------------------------------------------------------
ReadyToProcess(B) ==
    /\ B \subseteq Slots
    /\ B # {}
    /\ \A s \in B: Len(pending[s]) > 0

ProcessBatch(B) ==
    /\ ReadyToProcess(B)
    /\ step_count < MaxStep
    /\ pending' = [s \in Slots |->
                    IF s \in B THEN Tail(pending[s]) ELSE pending[s]]
    /\ slot_state' = [s \in Slots |->
                       IF s \in B
                       THEN Compute(slot_state[s], Head(pending[s]), B)
                       ELSE slot_state[s]]
    /\ slot_output' = [s \in Slots |->
                        IF s \in B
                        THEN Append(slot_output[s],
                                    EmitOutput(Compute(slot_state[s],
                                                        Head(pending[s]),
                                                        B)))
                        ELSE slot_output[s]]
    /\ step_count' = step_count + 1

----------------------------------------------------------------------------
(* Next.                                                                     *)
----------------------------------------------------------------------------
Next ==
    \/ \E s \in Slots: Enqueue(s)
    \/ \E B \in (SUBSET Slots) \ {{}}: ProcessBatch(B)

Spec == Init /\ [][Next]_vars

----------------------------------------------------------------------------
(* Sanity invariants for Phase 1.                                            *)
(*                                                                           *)
(* No safety/determinism invariants here. Phase 1 is a faithful capture of *)
(* the buggy reality; asserting determinism would always violate.           *)
(*****************************************************************************)

\* slot_state grows only via ProcessBatch (never via Enqueue).
NoSpontaneousState ==
    \A s \in Slots: Len(slot_state[s]) <= step_count

\* slot_output length matches slot_state length for each slot — every
\* compute step that updates state also emits an output.
OutputMatchesState ==
    \A s \in Slots: Len(slot_output[s]) = Len(slot_state[s])

\* Bounded counter for finite MC.
BoundedStep ==
    step_count <= MaxStep

\* Bounded pending for finite MC.
BoundedPending ==
    \A s \in Slots: Len(pending[s]) <= MaxPending

============================================================================
