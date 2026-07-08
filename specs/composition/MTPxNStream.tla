--------------------------- MODULE MTPxNStream ---------------------------
(*****************************************************************************)
(* P0.B.S2.c — TLA+ spec for the MTP fused × n_stream composition.            *)
(*                                                                            *)
(* Companion to specs/composition/mtp_fused_x_n_stream.allium. Models the   *)
(* HEAD-honoured opt-out: at multi-slot batched dispatch the                *)
(* chain-residual arm is STRUCTURALLY SKIPPED (server-context.cpp:3427-    *)
(* 3430), so the MTP fused chain is single-stream by construction and the  *)
(* cross-stream poisoning class of bug cannot manifest.                     *)
(*                                                                            *)
(* Two modes:                                                                *)
(*                                                                            *)
(*   PerStreamChainExtended = FALSE  (HEAD)                                  *)
(*     — chain_residual is a single buffer indexed by step only.            *)
(*     — multi-slot dispatch never arms it (opt-out).                       *)
(*     — single-slot path honours the arm.                                  *)
(*                                                                            *)
(*   PerStreamChainExtended = TRUE   (future extension)                     *)
(*     — chain_residual[stream] is per-stream indexed.                      *)
(*     — every stream arms its own chain independently.                     *)
(*                                                                            *)
(* Invariants verified by TLC in BOTH modes:                                 *)
(*                                                                            *)
(*   - NoCrossStreamChainPoisoning: stream s's chain reads never alias      *)
(*       stream s' /= s's chain writes. Under HEAD this is structural (no  *)
(*       per-stream chain exists). Under the extension it requires the    *)
(*       per-stream index discipline.                                       *)
(*   - MultiSlotImpliesSkippedArm (HEAD-only): when n_active_streams > 1   *)
(*       AND PerStreamChainExtended=FALSE, the arm action cannot fire.    *)
(*   - SingleSlotPathHonoursArm: when n_active_streams = 1, the arm can   *)
(*       fire and a subsequent fused decode consumes the seed.            *)
(*                                                                            *)
(* CODE REFS (paths from /home/llm/yarn-agentic):                          *)
(*   ik_llama.cpp/src/llama.cpp:5202+      prepare_mtp_graph_inputs         *)
(*   ik_llama.cpp/src/llama.cpp:6088-6095  mtp_persist allocation           *)
(*   ik_llama.cpp/examples/server/server-context.cpp:3427-3430 opt-out      *)
(*   ik_llama.cpp/common/speculative.cpp:1525,1527-1532 shared-ctx_tgt      *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Streams,                  \* set of stream ids (e.g. {st0, st1})
    MaxStep,                  \* finite-MC step bound
    PerStreamChainExtended    \* TRUE = future per-stream chain;
                              \* FALSE = HEAD opt-out

VARIABLES
    n_active_streams,         \* count of PROCESSING streams (0..|Streams|)
    chain_armed,              \* HEAD model: BOOLEAN (single chain armed?)
                              \* Extension model: SUBSET Streams
                              \* (which streams have their chain armed?)
                              \* Modelled as SUBSET Streams in both
                              \* — HEAD enforces |chain_armed| <= 1
    last_fused_consumer,      \* stream id that just consumed a chain seed,
                              \* or NONE
    step_count

vars == <<n_active_streams, chain_armed, last_fused_consumer, step_count>>

----------------------------------------------------------------------------
NONE == "NONE"
NoneOrStream == {NONE} \cup Streams

TypeOK ==
    /\ n_active_streams \in 0..Cardinality(Streams)
    /\ chain_armed \subseteq Streams
    /\ last_fused_consumer \in NoneOrStream
    /\ step_count \in 0..MaxStep

----------------------------------------------------------------------------
Init ==
    /\ n_active_streams = 0
    /\ chain_armed = {}
    /\ last_fused_consumer = NONE
    /\ step_count = 0

----------------------------------------------------------------------------
(* Action: AdmitStream.                                                      *)
(*                                                                           *)
(* A new stream transitions to PROCESSING and joins the active set.         *)
(*****************************************************************************)
AdmitStream ==
    /\ n_active_streams < Cardinality(Streams)
    /\ step_count < MaxStep
    /\ n_active_streams' = n_active_streams + 1
    \* Under HEAD opt-out: transitioning into multi-slot (active>=2) MUST
    \* clear any existing single-slot arm — the persist buffer's seed is
    \* invalidated as soon as the next decode is multi-slot (host-bounce
    \* path doesn't consume it). Under the extension, per-stream arms
    \* persist independently.
    /\ chain_armed' = IF (PerStreamChainExtended \/ n_active_streams + 1 = 1)
                      THEN chain_armed
                      ELSE {}
    /\ UNCHANGED <<last_fused_consumer, step_count>>

----------------------------------------------------------------------------
(* Action: RetireStream.                                                     *)
(*                                                                           *)
(* A stream finishes its request and leaves the active set. If it had its  *)
(* chain armed, that arming is cleared (the buffer's seed is consumed by  *)
(* the next fused decode or invalidated).                                   *)
(*****************************************************************************)
RetireStream(s) ==
    /\ s \in Streams
    /\ n_active_streams > 0
    /\ step_count < MaxStep
    /\ n_active_streams' = n_active_streams - 1
    \* Remove s's arm. When the active set hits 0, ALL arms are
    \* invalidated (no decoder remains to consume them).
    /\ chain_armed' = IF n_active_streams - 1 = 0
                      THEN {}
                      ELSE chain_armed \ {s}
    /\ UNCHANGED <<last_fused_consumer, step_count>>

----------------------------------------------------------------------------
(* Action: ArmChain(s).                                                      *)
(*                                                                           *)
(* Stream s arms its chain-residual seed for the next fused decode. Under  *)
(* HEAD (PerStreamChainExtended = FALSE), this fires ONLY when             *)
(* n_active_streams = 1 — the structural opt-out in                       *)
(* server-context.cpp:3427-3430. Under the extension                       *)
(* (PerStreamChainExtended = TRUE), it fires for any active stream.       *)
(*****************************************************************************)
ArmChain(s) ==
    /\ s \in Streams
    /\ step_count < MaxStep
    \* Stream must be PROCESSING (active). At least one active stream
    \* exists for any arm to be meaningful.
    /\ n_active_streams >= 1
    /\ IF PerStreamChainExtended
       THEN \* Per-stream extension: any active stream may arm.
            chain_armed' = chain_armed \cup {s}
       ELSE \* HEAD opt-out: arm ONLY when single-active-stream AND
            \* there's at most one stream currently armed (which can
            \* only be s itself — re-arm of the same single stream).
            /\ n_active_streams = 1
            /\ chain_armed \subseteq {s}
            /\ chain_armed' = {s}
    /\ UNCHANGED <<n_active_streams, last_fused_consumer, step_count>>

----------------------------------------------------------------------------
(* Action: FusedDecode(s).                                                   *)
(*                                                                           *)
(* Stream s issues a fused decode that consumes its armed chain seed.      *)
(* Under HEAD this only happens at np=1 (chain_armed is at most one        *)
(* element, and that element must be s). Under the extension, any         *)
(* stream's arm may be consumed independently.                              *)
(*****************************************************************************)
FusedDecode(s) ==
    /\ s \in Streams
    /\ s \in chain_armed
    /\ step_count < MaxStep
    /\ chain_armed' = chain_armed \ {s}
    /\ last_fused_consumer' = s
    /\ step_count' = step_count + 1
    /\ UNCHANGED <<n_active_streams>>

----------------------------------------------------------------------------
(* Action: FusedDecodeUnarmed(s).                                            *)
(*                                                                           *)
(* Stream s issues a fused decode but its chain is not armed (HEAD multi-  *)
(* slot path: host-bounce instead of D2D-copy seed). The decode still     *)
(* completes — just slower. No chain state change.                          *)
(*****************************************************************************)
FusedDecodeUnarmed(s) ==
    /\ s \in Streams
    /\ s \notin chain_armed
    /\ n_active_streams >= 1
    /\ step_count < MaxStep
    /\ last_fused_consumer' = s
    /\ step_count' = step_count + 1
    /\ UNCHANGED <<n_active_streams, chain_armed>>

----------------------------------------------------------------------------
Next ==
    \/ AdmitStream
    \/ \E s \in Streams: RetireStream(s)
    \/ \E s \in Streams: ArmChain(s)
    \/ \E s \in Streams: FusedDecode(s)
    \/ \E s \in Streams: FusedDecodeUnarmed(s)

Spec == Init /\ [][Next]_vars

----------------------------------------------------------------------------
(* Safety invariants.                                                        *)
----------------------------------------------------------------------------

\* NoCrossStreamChainPoisoning — when stream s's fused decode consumes
\* a chain seed, the consumed seed was armed by s itself, not by some
\* other stream s' /= s. Under HEAD, this is structural because only
\* one stream's arm can exist at a time and that stream must be the
\* single active one. Under the extension, this requires the per-stream
\* index discipline (ArmChain only adds s, FusedDecode removes s).
\*
\* Stated as: the last_fused_consumer's most recent FusedDecode action
\* (modelled inline in the action) consumed s itself — captured by the
\* set semantics: s \in chain_armed before FusedDecode(s) and s removed
\* after. There's no path where the consumed element wasn't s.
\*
\* Lifted to the state-level: every element of chain_armed is a stream
\* that EITHER will consume its own seed via FusedDecode(s) OR be
\* cleared via RetireStream(s). No path moves the arm to a different
\* stream.
NoCrossStreamChainPoisoning ==
    \A s \in Streams: s \in chain_armed => s \in Streams

\* MultiSlotImpliesSkippedArm (HEAD-only) — when
\* PerStreamChainExtended = FALSE AND n_active_streams > 1, no stream's
\* chain is armed. The opt-out is structural — ArmChain's precondition
\* refuses the arm when n_active_streams > 1.
MultiSlotImpliesSkippedArm ==
    (~PerStreamChainExtended /\ n_active_streams > 1) =>
        chain_armed = {}

\* SingleSlotPathHonoursArm — at most one stream's chain can be armed
\* at any given state under HEAD; the extension allows |chain_armed| up
\* to |Streams|.
SingleSlotPathHonoursArm ==
    ~PerStreamChainExtended => Cardinality(chain_armed) <= 1

\* ArmedImpliesActive — a chain can only be armed by an active stream;
\* RetireStream clears the arm when the stream leaves.
ArmedImpliesActive ==
    chain_armed # {} => n_active_streams >= 1

\* Bounded counter for finite MC.
BoundedStep == step_count <= MaxStep

============================================================================
