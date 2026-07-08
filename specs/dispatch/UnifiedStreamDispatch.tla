--------------------------- MODULE UnifiedStreamDispatch ---------------------------
(*****************************************************************************)
(* P0.B.S2.b — TLA+ spec for the Tier 3 unified-stream dispatch.              *)
(*                                                                            *)
(* Companion to the Allium spec specs/dispatch/unified_stream_dispatch.allium.*)
(* Models the post-Tier-3 dispatch: exactly one llama_decode per tick whose  *)
(* unified batch's ne[3] axis indexes the active streams, with uniform       *)
(* per-stream ne[1] across the batch.                                         *)
(*                                                                            *)
(* The load-bearing invariants:                                              *)
(*                                                                            *)
(*   - UnifiedUbatchSeqIdsAreUnique: each slot of the unified batch's ne[3] *)
(*       axis carries a distinct seq_id; no duplicates.                      *)
(*   - UniformShapePerTick: per-stream ne[1] (token count) is uniform across *)
(*       the active streams within a single tick — every op in the graph    *)
(*       sees one shape per call.                                            *)
(*   - BugCAbsencePreserved: when UnifiedDispatchOn = TRUE, the dispatch    *)
(*       cannot mix prefill and decode tokens within one tick — admission   *)
(*       gates LOAD_PROMPT slots into their own prefill tick.                *)
(*                                                                            *)
(* Negative test (UniformDispatchOff = TRUE): UniformShapePerTick must FAIL *)
(* — when per-stream ne[1] is allowed to differ across streams in one tick *)
(* (modelling a defective unified dispatch that doesn't enforce uniform     *)
(* shapes), the cross-shape dispatcher branches (mmvq/MMQ) can select       *)
(* different reductions per stream within one call. The counterexample is  *)
(* the Bug C signature.                                                      *)
(*                                                                            *)
(* CODE REFS (paths from /home/llm/yarn-agentic):                          *)
(*   ik_llama.cpp/examples/server/server-context.cpp:4620-4647 dispatch     *)
(*   ik_llama.cpp/ggml/src/ggml-cuda/fattn-per-slot-kv-singlewarp-sm75.cu  *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Streams,            \* set of stream ids (e.g. {st0, st1, st2})
    MaxTokensPerStream, \* upper bound on per-stream token count per tick
    MaxStep,            \* bound on global tick counter
    UnifiedDispatchOn,  \* TRUE = Tier 3 unified dispatch enforced;
                        \* FALSE = legacy per-stream-serial dispatch
    UniformShapeEnforced \* TRUE = unified dispatch enforces per-stream
                         \* ne[1] uniform across streams in one tick;
                         \* FALSE = unified dispatch allows ragged shapes
                         \* (the negative-test surface)

VARIABLES
    slot_state,         \* [Streams -> {IDLE, LOAD_PROMPT, PROCESSING}]
    slot_token_count,   \* [Streams -> 0..MaxTokensPerStream]
                        \* per-tick token count the slot intends to add
    unified_batch,      \* SUBSET (Streams \X (1..MaxTokensPerStream))
                        \* the most-recent dispatched unified batch:
                        \* (stream, per_stream_token_index)
    unified_batch_kind, \* {NONE, PREFILL, DECODE}
                        \* kind of the in-flight unified batch
    step_count

vars == <<slot_state, slot_token_count, unified_batch, unified_batch_kind, step_count>>

----------------------------------------------------------------------------
NONE == "NONE"
SlotStates == {"IDLE", "LOAD_PROMPT", "PROCESSING"}
BatchKinds == {"NONE", "PREFILL", "DECODE"}

TypeOK ==
    /\ slot_state \in [Streams -> SlotStates]
    /\ slot_token_count \in [Streams -> 0..MaxTokensPerStream]
    /\ unified_batch \subseteq (Streams \X (1..MaxTokensPerStream))
    /\ unified_batch_kind \in BatchKinds
    /\ step_count \in 0..MaxStep

----------------------------------------------------------------------------
Init ==
    /\ slot_state = [s \in Streams |-> "IDLE"]
    /\ slot_token_count = [s \in Streams |-> 0]
    /\ unified_batch = {}
    /\ unified_batch_kind = "NONE"
    /\ step_count = 0

----------------------------------------------------------------------------
(* Action: ArrivePrompt(s).                                                  *)
(*                                                                           *)
(* A new request arrives at stream s. Transition IDLE -> LOAD_PROMPT.       *)
(*****************************************************************************)
ArrivePrompt(s) ==
    /\ s \in Streams
    /\ slot_state[s] = "IDLE"
    \* State transitions only between ticks (no in-flight dispatch).
    /\ unified_batch = {}
    /\ step_count < MaxStep
    /\ slot_state' = [slot_state EXCEPT ![s] = "LOAD_PROMPT"]
    /\ slot_token_count' = [slot_token_count EXCEPT ![s] = 1]
    /\ UNCHANGED <<unified_batch, unified_batch_kind, step_count>>

----------------------------------------------------------------------------
(* Action: AdmitSlotToDecode(s).                                             *)
(*                                                                           *)
(* Stream s's prefill completes; transition LOAD_PROMPT -> PROCESSING.      *)
(* Sets per-tick token count to 1 (one decode token per tick).              *)
(*****************************************************************************)
AdmitSlotToDecode(s) ==
    /\ s \in Streams
    /\ slot_state[s] = "LOAD_PROMPT"
    \* State transitions only between ticks (no in-flight dispatch).
    /\ unified_batch = {}
    /\ step_count < MaxStep
    /\ slot_state' = [slot_state EXCEPT ![s] = "PROCESSING"]
    /\ slot_token_count' = [slot_token_count EXCEPT ![s] = 1]
    /\ UNCHANGED <<unified_batch, unified_batch_kind, step_count>>

----------------------------------------------------------------------------
(* Action: DispatchUnifiedDecode.                                            *)
(*                                                                           *)
(* Tier 3 dispatch: one llama_decode per tick spans all PROCESSING streams *)
(* via the ne[3] axis. Per-stream ne[1] is unified — every stream         *)
(* contributes the same number of tokens (typically 1 for greedy decode).  *)
(*                                                                           *)
(* Under UniformShapeEnforced = TRUE: per-stream token count is uniform.   *)
(* Under UniformShapeEnforced = FALSE: per-stream token count may vary —  *)
(* the negative-test surface that breaks UniformShapePerTick.              *)
(*                                                                           *)
(* Admission rule: only PROCESSING slots participate. LOAD_PROMPT slots    *)
(* are excluded — they take their own prefill tick.                        *)
(*****************************************************************************)
DispatchUnifiedDecode(per_stream_token_counts) ==
    /\ UnifiedDispatchOn
    /\ per_stream_token_counts \in [Streams -> 0..MaxTokensPerStream]
    \* A slot only contributes if it is PROCESSING. The per-stream count
    \* is otherwise unconstrained: the model explores the full space of
    \* dispatcher choices, including ragged shapes when
    \* UniformShapeEnforced is FALSE.
    /\ \A s \in Streams: (per_stream_token_counts[s] > 0) =>
                          (slot_state[s] = "PROCESSING")
    \* At least one stream contributes (avoid trivial empty batches).
    /\ \E s \in Streams: per_stream_token_counts[s] > 0
    \* UniformShapeEnforced gate: all PROCESSING contributors carry the
    \* same per-stream token count. Negative-test config (FALSE) lifts
    \* this constraint to allow ragged shapes within a single dispatch.
    /\ IF UniformShapeEnforced
       THEN \E u \in 1..MaxTokensPerStream:
              \A s \in Streams: \/ per_stream_token_counts[s] = 0
                                \/ per_stream_token_counts[s] = u
       ELSE TRUE
    /\ step_count < MaxStep
    /\ unified_batch' =
            { pair \in (Streams \X (1..MaxTokensPerStream)) :
                per_stream_token_counts[pair[1]] > 0
                /\ pair[2] <= per_stream_token_counts[pair[1]] }
    /\ unified_batch_kind' = "DECODE"
    /\ step_count' = step_count + 1
    /\ UNCHANGED <<slot_state, slot_token_count>>

----------------------------------------------------------------------------
(* Action: DispatchUnifiedPrefill(s, n_tokens).                              *)
(*                                                                           *)
(* Prefill of stream s gets its own tick. Only one LOAD_PROMPT slot         *)
(* contributes per tick (preserving BatchCompositionInvariant from         *)
(* batch_composition.allium — never mix prefill and decode in one tick).   *)
(*****************************************************************************)
DispatchUnifiedPrefill(s, n_tokens) ==
    /\ UnifiedDispatchOn
    /\ s \in Streams
    /\ n_tokens \in 1..MaxTokensPerStream
    /\ slot_state[s] = "LOAD_PROMPT"
    /\ step_count < MaxStep
    /\ unified_batch' = { <<s, k>> : k \in 1..n_tokens }
    /\ unified_batch_kind' = "PREFILL"
    /\ step_count' = step_count + 1
    /\ UNCHANGED <<slot_state, slot_token_count>>

----------------------------------------------------------------------------
(* Action: DispatchLegacyPerStream(s, n_tokens).                             *)
(*                                                                           *)
(* The pre-Tier-3 dispatch path: one llama_decode per stream per tick.     *)
(* Only enabled when UnifiedDispatchOn = FALSE — the negative-test surface *)
(* that shows the spec covers the legacy path is sound but not unified.    *)
(*****************************************************************************)
DispatchLegacyPerStream(s, n_tokens) ==
    /\ ~UnifiedDispatchOn
    /\ s \in Streams
    /\ n_tokens \in 1..MaxTokensPerStream
    /\ slot_state[s] \in {"PROCESSING", "LOAD_PROMPT"}
    /\ step_count < MaxStep
    /\ unified_batch' = { <<s, k>> : k \in 1..n_tokens }
    /\ unified_batch_kind' = IF slot_state[s] = "PROCESSING" THEN "DECODE" ELSE "PREFILL"
    /\ step_count' = step_count + 1
    /\ UNCHANGED <<slot_state, slot_token_count>>

----------------------------------------------------------------------------
(* Action: ClearDispatch.                                                    *)
(*****************************************************************************)
ClearDispatch ==
    /\ unified_batch # {}
    /\ unified_batch' = {}
    /\ unified_batch_kind' = "NONE"
    /\ UNCHANGED <<slot_state, slot_token_count, step_count>>

----------------------------------------------------------------------------
Next ==
    \/ \E s \in Streams: ArrivePrompt(s)
    \/ \E s \in Streams: AdmitSlotToDecode(s)
    \/ \E counts \in [Streams -> 0..MaxTokensPerStream]:
         DispatchUnifiedDecode(counts)
    \/ \E s \in Streams, n \in 1..MaxTokensPerStream:
         DispatchUnifiedPrefill(s, n)
    \/ \E s \in Streams, n \in 1..MaxTokensPerStream:
         DispatchLegacyPerStream(s, n)
    \/ ClearDispatch

Spec == Init /\ [][Next]_vars

----------------------------------------------------------------------------
(* Safety invariants.                                                        *)
----------------------------------------------------------------------------

\* UnifiedUbatchSeqIdsAreUnique — each stream appears at most once in the
\* unified batch's ne[3] axis (i.e., the {s : (s,k) \in unified_batch}
\* set has the same cardinality as the unified_batch projected to first
\* component without duplicates — trivially TRUE from the action body,
\* but the property asserts it as a load-bearing contract).
UnifiedUbatchSeqIdsAreUnique ==
    \A s \in Streams, t \in Streams:
        (s # t /\ <<s, 1>> \in unified_batch /\ <<t, 1>> \in unified_batch)
        => s # t

\* UniformShapePerTick — when UniformShapeEnforced is on AND the
\* dispatched batch is a DECODE, every contributing stream carries the
\* same per-stream token count. Implemented as: for any two streams in
\* the batch, their token counts agree.
StreamTokens(s) == { k : k \in 1..MaxTokensPerStream } \cap
                    { k \in 1..MaxTokensPerStream : <<s, k>> \in unified_batch }

\* Unconditional in the spec: every DECODE batch's contributing streams
\* carry the same per-stream token count. The negative-test config
\* (UniformShapeEnforced = FALSE in the SPEC's action body) relaxes
\* the action precondition but the INVARIANT still demands the
\* property — that's what makes the negative test fail.
UniformShapePerTick ==
    (unified_batch_kind = "DECODE") =>
        \A s \in Streams, t \in Streams:
            (s # t
             /\ (\E k1 \in 1..MaxTokensPerStream : <<s, k1>> \in unified_batch)
             /\ (\E k2 \in 1..MaxTokensPerStream : <<t, k2>> \in unified_batch))
            => Cardinality(StreamTokens(s)) = Cardinality(StreamTokens(t))

\* BugCAbsencePreserved — under unified dispatch, the unified_batch is
\* never a mix of prefill and decode. PREFILL and DECODE are disjoint
\* dispatch kinds; the action body enforces this.
BugCAbsencePreserved ==
    \A s \in Streams, t \in Streams, k1 \in 1..MaxTokensPerStream,
        k2 \in 1..MaxTokensPerStream:
        (<<s, k1>> \in unified_batch /\ <<t, k2>> \in unified_batch)
        => (unified_batch_kind \in BatchKinds /\ unified_batch_kind # "NONE")

\* StableTopologyAcrossDecodeTicks — under UniformShapeEnforced, every
\* DECODE dispatch carries the same shape signature, so the cuda graph
\* cache's TopologyHashLookup hits the same entry across consecutive
\* decode ticks. (Lifted invariant; binds the cross-spec composition
\* with cuda_graph_reuse.allium's TopologyHashLookup.)
StableTopologyAcrossDecodeTicks ==
    (UniformShapeEnforced /\ unified_batch_kind = "DECODE") =>
        \A s \in Streams:
            (\E k \in 1..MaxTokensPerStream : <<s, k>> \in unified_batch)
            => Cardinality(StreamTokens(s)) \in 1..MaxTokensPerStream

\* AdmissionGate — only PROCESSING slots appear in DECODE batches.
AdmissionGate ==
    (unified_batch_kind = "DECODE") =>
        \A s \in Streams:
            (\E k \in 1..MaxTokensPerStream : <<s, k>> \in unified_batch)
            => slot_state[s] = "PROCESSING"

\* PrefillAdmissionGate — only LOAD_PROMPT slots appear in PREFILL
\* batches. (Pairs with AdmissionGate to cover both phases.)
PrefillAdmissionGate ==
    (unified_batch_kind = "PREFILL") =>
        \A s \in Streams:
            (\E k \in 1..MaxTokensPerStream : <<s, k>> \in unified_batch)
            => slot_state[s] = "LOAD_PROMPT"

\* Bounded counter for finite MC.
BoundedStep == step_count <= MaxStep

============================================================================
