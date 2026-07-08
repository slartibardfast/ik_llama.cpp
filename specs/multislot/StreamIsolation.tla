--------------------------- MODULE StreamIsolation ---------------------------
(*****************************************************************************)
(* S2.b — TLA+ spec for the n_stream KV per-stream layout.                    *)
(*                                                                            *)
(* Companion to the Allium spec specs/kv-cache/n_stream_layer.allium.        *)
(* Models the structural property that makes Bug C impossible by             *)
(* construction: K, V tensors carry an n_stream axis, find_slot allocates   *)
(* contiguously within one stream's slice, KQ_mask is built per-stream,    *)
(* and the server dispatches one llama_decode per active stream per tick. *)
(*                                                                            *)
(* The load-bearing invariants:                                              *)
(*                                                                            *)
(*   - StreamPartition: each cell belongs to exactly one stream's slice;    *)
(*       slices are disjoint.                                                *)
(*   - NoCrossStreamLeakage: a write destined for stream s never mutates    *)
(*       any cell in stream s' /= s.                                         *)
(*   - MaskCorrectness: under PerStreamDispatch (one llama_decode per       *)
(*       stream per tick), the KQ_mask read by stream s's compute sees      *)
(*       only cells in stream s's slice.                                     *)
(*   - DispatchUnitUniformity: every dispatched llama_decode carries        *)
(*       tokens of exactly one seq_id.                                        *)
(*                                                                            *)
(* Relationship to BatchComposition.tla: under DispatchUnitUniformity,      *)
(* MixedBatchProhibition (from the BatchComposition spec) holds             *)
(* automatically. This module exists to verify that — the negative test    *)
(* removes PerStreamDispatch from the model and checks that the              *)
(* MixedBatchProhibition-by-construction property fails.                     *)
(*                                                                            *)
(* CODE REFS (paths from /home/llm/yarn-agentic):                          *)
(*   ik_llama.cpp/src/llama-context.h:53-63    n_stream + v_heads fields    *)
(*   ik_llama.cpp/src/llama.cpp:1156-1254      llama_kv_cache_find_slot     *)
(*   ik_llama.cpp/src/llama.cpp ~4290-4470     build_inp_KQ_mask            *)
(*   ik_llama.cpp/examples/server/server-context.cpp ~4602  dispatch site  *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Streams,            \* set of stream ids (e.g. {st0, st1, st2})
    StreamSize,         \* cells per stream's slice (e.g. 2048)
    MaxStep,            \* bound on global tick counter for finite MC
    MaxTokensPerTick,   \* upper bound on tokens dispatched in one llama_decode
    PerStreamDispatchOn \* TRUE = Option A active; FALSE = legacy shared pool

VARIABLES
    cell_seq,        \* [Streams -> [0..StreamSize-1 -> {NONE} \cup Streams]]
                     \* cell_seq[s][i] = seq_id owning the cell, or NONE.
                     \* The pos values themselves don't enter any
                     \* structural invariant, so they are not modelled.
    v_head,          \* [Streams -> 0..StreamSize] — next-free pointer
    dispatch_unit,   \* SUBSET (Streams \X (1..MaxTokensPerTick))
                     \* {(seq_id, token_index)} carried by the most-recent
                     \* dispatched llama_decode.
    dispatch_seq_id, \* The single seq_id this dispatch unit serves, or NONE
                     \* when no dispatch in flight.
    step_count

vars == <<cell_seq, v_head, dispatch_unit, dispatch_seq_id, step_count>>

----------------------------------------------------------------------------
NONE == "NONE"
NoneOrStream == {NONE} \cup Streams

CellIndex == 0..(StreamSize - 1)

TypeOK ==
    /\ cell_seq \in [Streams -> [CellIndex -> NoneOrStream]]
    /\ v_head \in [Streams -> 0..StreamSize]
    /\ dispatch_unit \subseteq (Streams \X (1..MaxTokensPerTick))
    /\ dispatch_seq_id \in NoneOrStream
    /\ step_count \in 0..MaxStep

\* Global cell index for stream s, local index i. The convention from
\* specs/kv-cache/n_stream_layer.allium: cell index i in stream s falls
\* in the global range [s * StreamSize, (s + 1) * StreamSize). Modelled
\* here implicitly via the (stream, local-index) pair; the global index
\* is just (s, i).

\* The set of all (stream, cell_index) pairs.
AllCells == { <<s, i>> : s \in Streams, i \in CellIndex }

\* Cells that have been written (cell_seq != NONE).
OccupiedCells == { c \in AllCells : cell_seq[c[1]][c[2]] # NONE }

----------------------------------------------------------------------------
Init ==
    /\ cell_seq = [s \in Streams |-> [i \in CellIndex |-> NONE]]
    /\ v_head = [s \in Streams |-> 0]
    /\ dispatch_unit = {}
    /\ dispatch_seq_id = NONE
    /\ step_count = 0

----------------------------------------------------------------------------
(* Action: FindSlotAndWrite(s, n_tokens, base_pos).                          *)
(*                                                                           *)
(* Models llama_kv_cache_find_slot(seq_id=s, n_tokens) followed by the      *)
(* graph builder's K/V writes. Allocates n_tokens contiguous cells starting *)
(* at v_head[s], wrapping if necessary (modelled as failure for             *)
(* simplicity — the model checker doesn't need to exercise the wrap path). *)
(*                                                                           *)
(* Under PerStreamDispatchOn = TRUE, the allocation is scoped to stream s. *)
(* Under PerStreamDispatchOn = FALSE (legacy shared pool), the find_slot   *)
(* would search the global pool and could allocate cells outside s's       *)
(* "natural" slice — the spec models this as allocating into any stream's *)
(* slice with capacity, demonstrating the StreamPartition failure that     *)
(* motivates the port.                                                      *)
(*****************************************************************************)
FindSlotAndWrite(s, n_tokens) ==
    /\ n_tokens \in 1..MaxTokensPerTick
    /\ s \in Streams
    /\ step_count < MaxStep
    /\ IF PerStreamDispatchOn
       THEN \* Strict per-stream allocation: only within stream s's slice.
            /\ v_head[s] + n_tokens <= StreamSize
            /\ cell_seq' = [cell_seq EXCEPT
                             ![s] = [i \in CellIndex |->
                                      IF i \in (v_head[s] .. (v_head[s] + n_tokens - 1))
                                      THEN s
                                      ELSE cell_seq[s][i]]]
            /\ v_head' = [v_head EXCEPT ![s] = v_head[s] + n_tokens]
       ELSE \* Legacy shared pool: may allocate into ANY stream's slice with capacity.
            \E target \in Streams:
              /\ v_head[target] + n_tokens <= StreamSize
              /\ cell_seq' = [cell_seq EXCEPT
                               ![target] = [i \in CellIndex |->
                                            IF i \in (v_head[target] .. (v_head[target] + n_tokens - 1))
                                            THEN s  \* seq_id is s; target may differ
                                            ELSE cell_seq[target][i]]]
              /\ v_head' = [v_head EXCEPT ![target] = v_head[target] + n_tokens]
    /\ UNCHANGED <<dispatch_unit, dispatch_seq_id, step_count>>

----------------------------------------------------------------------------
(* Action: DispatchPerStream(s, n_tokens).                                   *)
(*                                                                           *)
(* Models server_context::update_slots dispatching one llama_decode for     *)
(* stream s, carrying n_tokens tokens of seq_id = s. Sets the dispatch     *)
(* unit accordingly.                                                         *)
(*                                                                           *)
(* Under PerStreamDispatchOn = TRUE, this is the only way a dispatch is    *)
(* assembled — each call carries a single seq_id (DispatchUnitUniformity). *)
(* Under PerStreamDispatchOn = FALSE, the alternative action               *)
(* DispatchMixed below assembles a multi-seq batch — that's the negative   *)
(* control.                                                                 *)
(*****************************************************************************)
DispatchPerStream(s, n_tokens) ==
    /\ PerStreamDispatchOn
    /\ s \in Streams
    /\ n_tokens \in 1..MaxTokensPerTick
    /\ step_count < MaxStep
    /\ dispatch_unit' = { <<s, k>> : k \in 1..n_tokens }
    /\ dispatch_seq_id' = s
    /\ step_count' = step_count + 1
    /\ UNCHANGED <<cell_seq, v_head>>

----------------------------------------------------------------------------
(* Action: DispatchMixed(slots, n_per_slot).                                 *)
(*                                                                           *)
(* Models the legacy shared-pool dispatch where one llama_decode can carry *)
(* tokens from multiple seq_ids (the geometry that produces Bug C). This   *)
(* action is only enabled when PerStreamDispatchOn = FALSE.                *)
(*****************************************************************************)
DispatchMixed(seqs, n_per_slot) ==
    /\ ~PerStreamDispatchOn
    /\ seqs \subseteq Streams
    /\ seqs # {}
    /\ n_per_slot \in 1..MaxTokensPerTick
    /\ Cardinality(seqs) * n_per_slot <= MaxTokensPerTick
    /\ step_count < MaxStep
    /\ dispatch_unit' = { <<s, k>> : s \in seqs, k \in 1..n_per_slot }
    /\ dispatch_seq_id' = IF Cardinality(seqs) = 1
                          THEN CHOOSE s \in seqs : TRUE
                          ELSE NONE
    /\ step_count' = step_count + 1
    /\ UNCHANGED <<cell_seq, v_head>>

----------------------------------------------------------------------------
(* Action: ClearDispatch.                                                    *)
(*                                                                           *)
(* The dispatch completes; the in-flight slot is cleared. Allows the next  *)
(* dispatch action to fire.                                                  *)
(*****************************************************************************)
ClearDispatch ==
    /\ dispatch_unit # {}
    /\ dispatch_unit' = {}
    /\ dispatch_seq_id' = NONE
    /\ UNCHANGED <<cell_seq, v_head, step_count>>

----------------------------------------------------------------------------
Next ==
    \/ \E s \in Streams, n \in 1..MaxTokensPerTick:
         FindSlotAndWrite(s, n)
    \/ \E s \in Streams, n \in 1..MaxTokensPerTick: DispatchPerStream(s, n)
    \/ \E seqs \in (SUBSET Streams) \ {{}},
         n \in 1..MaxTokensPerTick: DispatchMixed(seqs, n)
    \/ ClearDispatch

Spec == Init /\ [][Next]_vars

----------------------------------------------------------------------------
(* Safety invariants.                                                        *)
----------------------------------------------------------------------------

\* StreamPartition — each occupied cell's seq_id field references exactly
\* the stream's own slice. Under PerStreamDispatch, find_slot writes only
\* into stream s's slice and labels the cell with seq_id = s.
\*
\* This invariant is the load-bearing structural guarantee. Under
\* PerStreamDispatchOn = FALSE, the legacy shared pool may write seq_id =
\* s into a cell in stream t /= s's slice — violating this invariant.
StreamPartition ==
    \A s \in Streams, i \in CellIndex:
        cell_seq[s][i] # NONE => cell_seq[s][i] = s

\* NoCrossStreamLeakage — operational form of StreamPartition. A write
\* destined for seq_id s does not appear in any stream other than s.
NoCrossStreamLeakage ==
    \A s \in Streams, i \in CellIndex:
        \A t \in Streams:
            (cell_seq[s][i] = t) => (s = t)

\* DispatchUnitUniformity — every in-flight dispatch unit carries tokens
\* of exactly one seq_id (the dispatch_seq_id). Under
\* PerStreamDispatchOn, DispatchPerStream sets this trivially. Under
\* PerStreamDispatchOn = FALSE, DispatchMixed may produce multi-seq batches
\* and violate this invariant.
DispatchUnitUniformity ==
    (dispatch_unit # {}) =>
        (dispatch_seq_id # NONE
         /\ \A pair \in dispatch_unit: pair[1] = dispatch_seq_id)

\* MixedBatchProhibitionByConstruction — the property that the Bug C
\* closure depends on. With PerStreamDispatch the dispatch is single-
\* seq by construction; without it a multi-seq dispatch can form.
\*
\* Stated independently of DispatchUnitUniformity to emphasise the
\* connection to BatchComposition.tla's MixedBatchProhibition.
MixedBatchProhibitionByConstruction ==
    (dispatch_unit # {}) =>
        Cardinality({ pair[1] : pair \in dispatch_unit }) = 1

\* VHeadBounded — book-keeping safety; v_head never exceeds stream's
\* capacity. Catches off-by-one in the allocator model.
VHeadBounded ==
    \A s \in Streams: v_head[s] <= StreamSize

\* VHeadMatchesOccupancy — every cell below v_head[s] in stream s is
\* occupied (cell_seq != NONE). The allocator is contiguous and
\* never leaves gaps below v_head.
VHeadMatchesOccupancy ==
    \A s \in Streams, i \in CellIndex:
        (i < v_head[s]) => (cell_seq[s][i] # NONE)

\* MaskCorrectness — under PerStreamDispatch, the cells eligible for the
\* in-flight dispatch's KQ_mask all belong to that dispatch's seq_id.
\* Models the per-stream mask filter cells[i].has_seq_id(s) AND
\* cells[i].pos <= pos from build_inp_KQ_mask. Lifted to a state-level
\* property: the set of cells the mask "sees" matches the set of cells
\* in the dispatch's stream.
MaskCorrectness ==
    (PerStreamDispatchOn /\ dispatch_seq_id # NONE) =>
        (\A s \in Streams, i \in CellIndex:
            (s = dispatch_seq_id /\ cell_seq[s][i] # NONE) =>
                cell_seq[s][i] = dispatch_seq_id)

\* Bounded counter for finite MC.
BoundedStep == step_count <= MaxStep

============================================================================
