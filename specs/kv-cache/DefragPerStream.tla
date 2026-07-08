--------------------------- MODULE DefragPerStream ---------------------------
(*****************************************************************************)
(* T3.6.S — TLA+ spec for defragmentation under the 4D per-stream KV layout.*)
(*                                                                            *)
(* Companion to the Allium spec                                              *)
(* specs/kv-cache/defrag_per_stream.allium. Models                            *)
(* `llama_kv_cache_defrag_internal` (cell metadata moves) and                *)
(* `build_defrag` (K/V byte cpy via 3D-per-stream views) under the           *)
(* per-stream outer loop.                                                    *)
(*                                                                            *)
(* The load-bearing invariant: DefragNoCrossStream. Every cell tag stays  *)
(* in the stream where it was originally placed. Cross-stream pulls are   *)
(* impossible under the per-stream outer loop; the negative-test action  *)
(* DefragFlat models the WRONG implementation that walks cells[] flat.   *)
(*                                                                            *)
(* CODE REFS (paths from /home/llm/yarn-agentic):                          *)
(*   ik_llama.cpp/src/llama.cpp:6661+ llama_kv_cache_defrag_internal      *)
(*   ik_llama.cpp/src/llama-build-context.cpp:280-364 build_defrag         *)
(*                                                                            *)
(* Audit findings:                                                          *)
(*   F3 (HIGH) — flat hole-fill walks cells[] across stream boundaries.   *)
(*               Per-stream outer loop is mandatory.                       *)
(*   F4 (HIGH) — 2D view incompatible with 4D layout. 3D-view-per-stream *)
(*               rewrite required.                                          *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Streams,            \* set of stream ids
    KvpsPerStream,      \* per-stream cell capacity (small for TLC)
    MaxStep             \* bound on global step counter

\* Tag domain: 0 means HOLE; positive integers are unique identity tags.
\* Tag identity discriminates "moved byte" from "stale byte".

HOLE == 0
MaxTag == MaxStep + 1
TagDomain == 0..MaxTag

VARIABLES
    cell_content,       \* [Streams \X 1..KvpsPerStream -> TagDomain]
    tag_origin,         \* [1..MaxTag -> Streams \cup {"unused"}]
                        \* origin stream of each tag at allocation time
    next_tag,
    step_count

vars == <<cell_content, tag_origin, next_tag, step_count>>

----------------------------------------------------------------------------
TypeOK ==
    /\ cell_content \in [Streams \X (1..KvpsPerStream) -> TagDomain]
    /\ tag_origin \in [1..MaxTag -> Streams \cup {"unused"}]
    /\ next_tag \in 1..(MaxTag + 1)
    /\ step_count \in 0..MaxStep

----------------------------------------------------------------------------
Init ==
    /\ cell_content = [t \in Streams \X (1..KvpsPerStream) |-> HOLE]
    /\ tag_origin = [tg \in 1..MaxTag |-> "unused"]
    /\ next_tag = 1
    /\ step_count = 0

----------------------------------------------------------------------------
OccupyCell(s, p) ==
    /\ s \in Streams
    /\ p \in 1..KvpsPerStream
    /\ cell_content[<<s, p>>] = HOLE
    /\ next_tag <= MaxTag
    /\ step_count < MaxStep
    /\ cell_content' = [cell_content EXCEPT ![<<s, p>>] = next_tag]
    /\ tag_origin' = [tag_origin EXCEPT ![next_tag] = s]
    /\ next_tag' = next_tag + 1
    /\ step_count' = step_count + 1

ReleaseCell(s, p) ==
    /\ s \in Streams
    /\ p \in 1..KvpsPerStream
    /\ cell_content[<<s, p>>] # HOLE
    /\ step_count < MaxStep
    /\ cell_content' = [cell_content EXCEPT ![<<s, p>>] = HOLE]
    /\ UNCHANGED <<tag_origin, next_tag>>
    /\ step_count' = step_count + 1

----------------------------------------------------------------------------
OccupiedTagsInStream(s) ==
    { cell_content[<<s, p>>] :
        p \in { q \in 1..KvpsPerStream : cell_content[<<s, q>>] # HOLE } }

OccupiedCountInStream(s) ==
    Cardinality({ p \in 1..KvpsPerStream : cell_content[<<s, p>>] # HOLE })

\* DefragStream(target) — compact target's occupied cells into the prefix
\* [1..k] of its slice. The set of tags is preserved; their order may
\* change (TLC explores both stable and reordered packings).
DefragStream(target) ==
    /\ target \in Streams
    /\ step_count < MaxStep
    /\ LET tags == OccupiedTagsInStream(target)
           k == Cardinality(tags)
       IN \E permutation \in [(1..k) -> tags]:
            /\ { permutation[i] : i \in 1..k } = tags
            /\ cell_content' = [t \in Streams \X (1..KvpsPerStream) |->
                                   IF t[1] # target
                                   THEN cell_content[t]
                                   ELSE IF t[2] <= k
                                        THEN permutation[t[2]]
                                        ELSE HOLE]
    /\ UNCHANGED <<tag_origin, next_tag>>
    /\ step_count' = step_count + 1

----------------------------------------------------------------------------
(* Negative-test action: DefragFlat.                                        *)
(*                                                                          *)
(* The WRONG implementation: flat walk across cells[] without stream     *)
(* awareness. A cell from stream s may land in stream s' != s. The         *)
(* DefragNoCrossStream invariant catches this.                             *)
(*****************************************************************************)
DefragFlat ==
    /\ step_count < MaxStep
    /\ LET all_occupied_pairs ==
              { pair \in Streams \X (1..KvpsPerStream) :
                  cell_content[pair] # HOLE }
           all_tags == { cell_content[pair] : pair \in all_occupied_pairs }
           total == Cardinality(all_tags)
       IN \E packing \in [Streams \X (1..KvpsPerStream) -> TagDomain]:
            /\ { p \in Streams \X (1..KvpsPerStream) :
                 packing[p] # HOLE } \subseteq all_occupied_pairs
            /\ { packing[p] :
                 p \in { q \in Streams \X (1..KvpsPerStream) :
                          packing[q] # HOLE } } = all_tags
            /\ Cardinality({ p \in Streams \X (1..KvpsPerStream) :
                              packing[p] # HOLE }) = total
            /\ cell_content' = packing
    /\ UNCHANGED <<tag_origin, next_tag>>
    /\ step_count' = step_count + 1

----------------------------------------------------------------------------
Next ==
    \/ \E s \in Streams, p \in 1..KvpsPerStream: OccupyCell(s, p)
    \/ \E s \in Streams, p \in 1..KvpsPerStream: ReleaseCell(s, p)
    \/ \E target \in Streams: DefragStream(target)

NextWithBug ==
    \/ \E s \in Streams, p \in 1..KvpsPerStream: OccupyCell(s, p)
    \/ \E s \in Streams, p \in 1..KvpsPerStream: ReleaseCell(s, p)
    \/ DefragFlat

Spec == Init /\ [][Next]_vars
SpecWithBug == Init /\ [][NextWithBug]_vars

----------------------------------------------------------------------------
(* Safety invariants.                                                       *)
----------------------------------------------------------------------------

\* DefragNoCrossStream — for every (s, p) with content != HOLE,
\* tag_origin[cell_content[s, p]] = s. Tags stay in their home stream.
\* Holds under DefragStream (which permutes within target's slice).
\* Fails under DefragFlat when tags get mixed across streams.
DefragNoCrossStream ==
    \A pair \in Streams \X (1..KvpsPerStream):
        (cell_content[pair] # HOLE) =>
            (tag_origin[cell_content[pair]] = pair[1])

\* TagsUnique — at any state, each tag appears at most once in
\* cell_content. The action bodies preserve this; pinned for safety.
TagsUnique ==
    \A pair1, pair2 \in Streams \X (1..KvpsPerStream):
        (pair1 # pair2 /\ cell_content[pair1] # HOLE
         /\ cell_content[pair1] = cell_content[pair2])
        => FALSE

\* Bounded counter.
BoundedStep == step_count <= MaxStep

============================================================================
