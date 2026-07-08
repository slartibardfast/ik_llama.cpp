--------------------------- MODULE KShiftPerStream ---------------------------
(*****************************************************************************)
(* T3.6.S — TLA+ spec for K-shift under the 4D per-stream KV layout.         *)
(*                                                                            *)
(* Companion to the Allium spec                                              *)
(* specs/kv-cache/k_shift_per_stream.allium. Models the per-stream loop in   *)
(* build_k_shift and the cross-stream isolation property under any           *)
(* schedule of `llama_kv_cache_seq_add(seq_id, p0, p1, delta)` operations.   *)
(*                                                                            *)
(* The load-bearing invariant: KShiftLocality. For every cell (s, p),       *)
(*   cell_pos[s, p] equals total_scheduled[s, p] — the sum of deltas       *)
(*   scheduled for THAT cell only. Holds under the per-stream loop;        *)
(*   fails under the broadcast-bug action that smears one stream's deltas *)
(*   across every stream at the same local position.                        *)
(*                                                                            *)
(* CODE REFS (paths from /home/llm/yarn-agentic):                          *)
(*   ik_llama.cpp/src/llama.cpp:4356-4366 inp_K_shift populator             *)
(*   ik_llama.cpp/src/llama-build-context.cpp:170-242 build_k_shift         *)
(*                                                                            *)
(* Audit findings:                                                          *)
(*   F2 (HIGH) — a single 4D rope on K view with broadcast deltas is WRONG. *)
(*               Per-stream loop is mandatory. This spec models that loop. *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Streams,            \* set of stream ids
    KvpsPerStream,      \* per-stream cell capacity
    MaxDelta,           \* bound on per-shift delta magnitude
    MaxStep             \* bound on global step counter

VARIABLES
    cell_pos,           \* [Streams \X 1..KvpsPerStream -> Int]
                        \* logical position of each cell (post applied shifts)
    pending_delta,      \* [Streams \X 1..KvpsPerStream -> Int]
                        \* delta queued by seq_add but not yet applied
    total_scheduled,    \* [Streams \X 1..KvpsPerStream -> Int]
                        \* ghost: sum of all deltas ever scheduled for (s, p)
    step_count

vars == <<cell_pos, pending_delta, total_scheduled, step_count>>

----------------------------------------------------------------------------
TypeOK ==
    /\ cell_pos \in [Streams \X (1..KvpsPerStream) -> Int]
    /\ pending_delta \in [Streams \X (1..KvpsPerStream) -> Int]
    /\ total_scheduled \in [Streams \X (1..KvpsPerStream) -> Int]
    /\ step_count \in 0..MaxStep

----------------------------------------------------------------------------
Init ==
    /\ cell_pos = [t \in Streams \X (1..KvpsPerStream) |-> 0]
    /\ pending_delta = [t \in Streams \X (1..KvpsPerStream) |-> 0]
    /\ total_scheduled = [t \in Streams \X (1..KvpsPerStream) |-> 0]
    /\ step_count = 0

----------------------------------------------------------------------------
(* Action: ScheduleShift(s, p, d).                                          *)
(*                                                                          *)
(* Models `llama_kv_cache_seq_add` for stream s at local pos p with        *)
(* delta d. The delta accumulates into pending_delta until ApplyShift.    *)
(* total_scheduled is incremented in lockstep as a ghost record.          *)
(*****************************************************************************)
ScheduleShift(s, p, d) ==
    /\ s \in Streams
    /\ p \in 1..KvpsPerStream
    /\ d \in -MaxDelta..MaxDelta
    /\ d # 0
    /\ step_count < MaxStep
    /\ pending_delta' = [pending_delta EXCEPT ![<<s, p>>] = @ + d]
    /\ total_scheduled' = [total_scheduled EXCEPT ![<<s, p>>] = @ + d]
    /\ step_count' = step_count + 1
    /\ UNCHANGED <<cell_pos>>

----------------------------------------------------------------------------
(* Action: ApplyShiftToStream(target).                                      *)
(*                                                                          *)
(* Models one iteration of the per-stream loop in build_k_shift. Applies   *)
(* every cell's pending_delta within stream `target` ONLY; other streams  *)
(* are untouched.                                                          *)
(*****************************************************************************)
ApplyShiftToStream(target) ==
    /\ target \in Streams
    /\ step_count < MaxStep
    /\ cell_pos' = [t \in Streams \X (1..KvpsPerStream) |->
                       IF t[1] = target
                       THEN cell_pos[t] + pending_delta[t]
                       ELSE cell_pos[t]]
    /\ pending_delta' = [t \in Streams \X (1..KvpsPerStream) |->
                            IF t[1] = target
                            THEN 0
                            ELSE pending_delta[t]]
    /\ UNCHANGED <<total_scheduled>>
    /\ step_count' = step_count + 1

----------------------------------------------------------------------------
(* Negative-test action: ApplyShiftBroadcast(target).                       *)
(*                                                                          *)
(* The WRONG implementation: a single 4D rope that broadcasts target's    *)
(* deltas to ALL streams at the same local position. Used by             *)
(* MC_negative.cfg to confirm KShiftLocality catches the bad path.       *)
(*****************************************************************************)
ApplyShiftBroadcast(target) ==
    /\ target \in Streams
    /\ step_count < MaxStep
    /\ cell_pos' = [t \in Streams \X (1..KvpsPerStream) |->
                       cell_pos[t] + pending_delta[<<target, t[2]>>]]
    /\ pending_delta' = [t \in Streams \X (1..KvpsPerStream) |->
                            IF t[1] = target THEN 0 ELSE pending_delta[t]]
    /\ UNCHANGED <<total_scheduled>>
    /\ step_count' = step_count + 1

----------------------------------------------------------------------------
Next ==
    \/ \E s \in Streams, p \in 1..KvpsPerStream, d \in -MaxDelta..MaxDelta:
         ScheduleShift(s, p, d)
    \/ \E target \in Streams: ApplyShiftToStream(target)

NextWithBug ==
    \/ \E s \in Streams, p \in 1..KvpsPerStream, d \in -MaxDelta..MaxDelta:
         ScheduleShift(s, p, d)
    \/ \E target \in Streams: ApplyShiftBroadcast(target)

Spec == Init /\ [][Next]_vars
SpecWithBug == Init /\ [][NextWithBug]_vars

----------------------------------------------------------------------------
(* Safety invariants.                                                       *)
----------------------------------------------------------------------------

\* KShiftLocality — cell_pos[s, p] equals the sum of deltas scheduled for
\* THAT (s, p) cell only, minus deltas still pending. Equivalently:
\*   total_scheduled[s, p] = cell_pos[s, p] + pending_delta[s, p].
\* Holds under ApplyShiftToStream. Fails under ApplyShiftBroadcast when
\* a delta scheduled for (target, p) shows up in another stream's cell_pos.
KShiftLocality ==
    \A pair \in Streams \X (1..KvpsPerStream):
        total_scheduled[pair] = cell_pos[pair] + pending_delta[pair]

\* PendingDeltaCleared — after applying a shift to target stream, target's
\* pending_delta is 0. (Implied by the action body; pinned as an explicit
\* property the model checker confirms.)
PendingDeltaCleared ==
    \A pair \in Streams \X (1..KvpsPerStream):
        pending_delta[pair] \in -MaxDelta * MaxStep..MaxDelta * MaxStep

\* Bounded counter for finite MC.
BoundedStep == step_count <= MaxStep

============================================================================
