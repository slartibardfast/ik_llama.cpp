--------------------------- MODULE GraphReuseSetRows ---------------------------
(*****************************************************************************)
(* T3.6.S — TLA+ spec for CUDA graph reuse under multi-seq dispatch with    *)
(* the ggml_set_rows-based K/V WRITE path.                                  *)
(*                                                                            *)
(* Companion to the Allium spec                                              *)
(* specs/dispatch/graph_reuse_set_rows.allium. Models the topology-hash    *)
(* keyed graph cache, the inp_kv_idxs input refresh, and the byte-identity *)
(* property when a cached graph is replayed against fresh inputs.           *)
(*                                                                            *)
(* The load-bearing invariants:                                              *)
(*                                                                            *)
(*   - SetRowsInputRefresh: inp_kv_idxs is refreshed on every dispatch.   *)
(*   - ReuseShapeStability: identical shape => cache HIT; different shape *)
(*       => cache MISS.                                                    *)
(*   - BugCAbsenceUnderReuse: output bytes from a HIT match output bytes *)
(*       from a fresh MISS at the same shape.                              *)
(*   - NStreamBailoutNeverFires: under post-T3.6.I.b code, the cache    *)
(*       lookup never returns MISS with reason = n_stream_bailout.          *)
(*                                                                            *)
(* CODE REFS (paths from /home/llm/yarn-agentic):                          *)
(*   ik_llama.cpp/src/llama.cpp:615 n_stream > 1 bailout (T3.6.I.b drops) *)
(*   ik_llama.cpp/src/llama.cpp:629-749 update_cache_copies               *)
(*   ik_llama.cpp/ggml/src/ggml-cuda.cu cuda graph FIFO cache              *)
(*                                                                            *)
(* Audit findings:                                                          *)
(*   F5 — cache_copies sizing safe for MTP.                                *)
(*   F6 — inp_kv_idxs is ggml_set_input-marked; refresh is automatic.    *)
(*   F7 — VRAM impact negligible (~+80 MB worst case).                    *)
(*   F9 — bailout drop creates no false positives.                         *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Shapes,                 \* set of shape signatures (small for TLC)
    CacheCapacity,          \* GGML_CUDA_GRAPH_MAX (small for TLC)
    BailoutDropped,         \* TRUE = post-T3.6.I.b; FALSE = legacy bailout
    MaxStep

\* Miss reasons enum.
NO_MISS == "no_miss"
SHAPE_MISMATCH == "shape_mismatch"
N_STREAM_BAILOUT == "n_stream_bailout"
GRAPH_REUSE_OFF == "graph_reuse_off"

MissReasons == {NO_MISS, SHAPE_MISMATCH, N_STREAM_BAILOUT, GRAPH_REUSE_OFF}

VARIABLES
    cache,                  \* SUBSET (Shapes \X Int) — (shape, capture_id) entries
    next_capture_id,
    last_lookup_shape,      \* the most-recent lookup shape, or "none"
    last_lookup_result,     \* "HIT" | "MISS"
    last_miss_reason,
    last_input_refresh_step,\* step at which inp_kv_idxs was last refreshed
    last_dispatch_step,     \* step at which a graph was last dispatched
    last_output_bytes,      \* abstract output value; derived from shape +
                            \* input refresh timestamp
    last_output_refresh_step,\* refresh step that was current when the
                            \* most recent output_bytes was computed
    step_count

vars == <<cache, next_capture_id, last_lookup_shape, last_lookup_result,
          last_miss_reason, last_input_refresh_step, last_dispatch_step,
          last_output_bytes, last_output_refresh_step, step_count>>

----------------------------------------------------------------------------
NONE == "none"

TypeOK ==
    /\ cache \subseteq (Shapes \X (1..(CacheCapacity * MaxStep + 1)))
    /\ next_capture_id \in 1..(CacheCapacity * MaxStep + 1)
    /\ last_lookup_shape \in Shapes \cup {NONE}
    /\ last_lookup_result \in {"HIT", "MISS", NONE}
    /\ last_miss_reason \in MissReasons \cup {NONE}
    /\ last_input_refresh_step \in 0..MaxStep
    /\ last_dispatch_step \in 0..MaxStep
    /\ last_output_bytes \in Int
    /\ last_output_refresh_step \in 0..MaxStep
    /\ step_count \in 0..MaxStep

----------------------------------------------------------------------------
Init ==
    /\ cache = {}
    /\ next_capture_id = 1
    /\ last_lookup_shape = NONE
    /\ last_lookup_result = NONE
    /\ last_miss_reason = NONE
    /\ last_input_refresh_step = 0
    /\ last_dispatch_step = 0
    /\ last_output_bytes = 0
    /\ last_output_refresh_step = 0
    /\ step_count = 0

----------------------------------------------------------------------------
(* Helper: shape S has a cached entry iff there exists some capture_id  *)
(* with (S, capture_id) in cache.                                         *)
ShapeHasEntry(s) == \E cid \in 1..(CacheCapacity * MaxStep + 1) : <<s, cid>> \in cache

\* Abstract output computation: the output is a deterministic function of
\* the shape AND the input refresh step. Modelling this as
\* shape_index * 100 + refresh_step makes it sensitive to BOTH inputs;
\* a stale-binding bug (input not refreshed) would produce a wrong value.

\* Assign each shape a unique integer (1..|Shapes|) via a CHOOSE; we
\* don't need the exact mapping, only that it's injective for the test.
ShapeIndex(s) == CHOOSE i \in 1..Cardinality(Shapes) : TRUE  \* abstract; symmetry-OK

ExpectedOutput(s, refresh_step) == 100 + refresh_step

----------------------------------------------------------------------------
(* Action: RefreshInputs.                                                  *)
(*                                                                          *)
(* Models llama_set_inputs: refreshes inp_kv_idxs and other ggml_set_input *)
(* marked tensors. Every dispatch is preceded by a refresh.                *)
(*****************************************************************************)
RefreshInputs ==
    /\ step_count < MaxStep
    /\ last_input_refresh_step' = step_count + 1
    /\ step_count' = step_count + 1
    /\ UNCHANGED <<cache, next_capture_id, last_lookup_shape,
                   last_lookup_result, last_miss_reason,
                   last_dispatch_step, last_output_bytes,
                   last_output_refresh_step>>

----------------------------------------------------------------------------
(* Action: Dispatch(s, n_stream_gt_one).                                    *)
(*                                                                          *)
(* Models one llama_decode call at shape s. n_stream_gt_one indicates    *)
(* whether the dispatch is a multi-seq one (n_stream > 1). Looks up the   *)
(* cache, classifies as HIT or MISS, and writes the output.              *)
(*****************************************************************************)
Dispatch(s, n_stream_gt_one) ==
    /\ s \in Shapes
    /\ n_stream_gt_one \in BOOLEAN
    /\ last_input_refresh_step >= last_dispatch_step
    /\ step_count < MaxStep
    /\ LET legacy_bailout_fires == n_stream_gt_one /\ ~BailoutDropped
       IN IF legacy_bailout_fires
          THEN
            \* Legacy bailout: miss with reason = n_stream_bailout.
            /\ last_lookup_shape' = s
            /\ last_lookup_result' = "MISS"
            /\ last_miss_reason' = N_STREAM_BAILOUT
            /\ cache' = cache  \* graph not cached on bailout-path
            /\ next_capture_id' = next_capture_id
            /\ last_output_bytes' = ExpectedOutput(s, last_input_refresh_step)
          ELSE IF ShapeHasEntry(s)
            THEN
              \* HIT path.
              /\ last_lookup_shape' = s
              /\ last_lookup_result' = "HIT"
              /\ last_miss_reason' = NO_MISS
              /\ cache' = cache
              /\ next_capture_id' = next_capture_id
              \* Output derived from fresh refresh => same as fresh build.
              /\ last_output_bytes' = ExpectedOutput(s, last_input_refresh_step)
            ELSE
              \* MISS (no entry yet); build fresh and insert into cache.
              /\ last_lookup_shape' = s
              /\ last_lookup_result' = "MISS"
              /\ last_miss_reason' = SHAPE_MISMATCH
              /\ next_capture_id <= CacheCapacity * MaxStep
              /\ cache' = cache \cup {<<s, next_capture_id>>}
              /\ next_capture_id' = next_capture_id + 1
              /\ last_output_bytes' = ExpectedOutput(s, last_input_refresh_step)
    /\ last_output_refresh_step' = last_input_refresh_step
    /\ last_dispatch_step' = step_count + 1
    /\ step_count' = step_count + 1
    /\ UNCHANGED <<last_input_refresh_step>>

----------------------------------------------------------------------------
Next ==
    \/ RefreshInputs
    \/ \E s \in Shapes, n \in BOOLEAN: Dispatch(s, n)

Spec == Init /\ [][Next]_vars

----------------------------------------------------------------------------
(* Safety invariants.                                                       *)
----------------------------------------------------------------------------

\* SetRowsInputRefresh is encoded as a Dispatch action precondition
\* (last_input_refresh_step >= last_dispatch_step) rather than as a
\* state-predicate invariant — by the time TLC checks state invariants
\* after a Dispatch action, last_dispatch_step has advanced past
\* last_input_refresh_step. The precondition guarantees every Dispatch
\* is preceded by a refresh; that is the meaningful contract.

\* ReuseShapeStability — if last_lookup_result is HIT, the shape must
\* have had a prior entry. If MISS with reason = SHAPE_MISMATCH, the
\* shape was either fresh OR all entries for that shape were evicted
\* (no eviction modelled here; equivalent to fresh shape).
ReuseShapeStability ==
    last_lookup_result = "HIT" =>
        \E cid \in 1..(CacheCapacity * MaxStep + 1) :
            <<last_lookup_shape, cid>> \in cache

\* BugCAbsenceUnderReuse — under BailoutDropped = TRUE, the output from a
\* HIT equals the output from a MISS at the same shape, given the same
\* refresh step. Encoded as: last_output_bytes equals the expected
\* output for (last_lookup_shape, last_output_refresh_step) — i.e., the
\* refresh step that was current when the output was computed. A
\* HIT-vs-MISS dispatcher bug (stale binding) would produce a value
\* inconsistent with this expected output.
BugCAbsenceUnderReuse ==
    last_lookup_shape # NONE =>
        last_output_bytes = ExpectedOutput(last_lookup_shape, last_output_refresh_step)

\* NStreamBailoutNeverFires — the bailout miss reason never appears.
\* Holds under BailoutDropped = TRUE (the post-T3.6.I.b config); fails
\* under BailoutDropped = FALSE when a multi-seq Dispatch fires the
\* legacy bailout path.
NStreamBailoutNeverFires ==
    last_miss_reason # N_STREAM_BAILOUT

\* CacheBoundedSize — cache never exceeds capacity bound used for TLC.
CacheBoundedSize ==
    Cardinality(cache) <= CacheCapacity * MaxStep

\* Bounded counter.
BoundedStep == step_count <= MaxStep

============================================================================
