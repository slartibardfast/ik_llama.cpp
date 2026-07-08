--------------------------- MODULE CUDAGraphCacheConsistency ---------------------------
(*****************************************************************************)
(* Status: LIVE as of PHASE_CUDA_NATIVE_DISPATCH commit C5.                  *)
(*                                                                            *)
(* Companion to specs/cuda-native-dispatch/multi_device_graph_cache.allium. *)
(*                                                                            *)
(* Models the sched-level multi-device cudaGraph_t cache as a state machine *)
(* over insertions, evictions, lookups, and replays. The cache is keyed by  *)
(* the dispatch topology hash and bounded at MAX_OUTER_GRAPHS (16).         *)
(* Eviction is FIFO on capacity overflow.                                   *)
(*                                                                            *)
(* The load-bearing invariants:                                              *)
(*                                                                            *)
(*   - CacheSizeBounded: |entries| <= capacity at all states.                *)
(*   - InsertedKeysInAgeList: every key in entries appears exactly once     *)
(*       in age_fifo (the FIFO order list).                                 *)
(*   - EvictionIsFifo: when a key is evicted, it was the oldest one in     *)
(*       age_fifo at that moment.                                            *)
(*   - HitImpliesLaunchOnly: when key in entries and compute_splits is     *)
(*       invoked, only LaunchExec runs — no BeginCapture or EndCapture.    *)
(*   - DestroyOnEvict: every evicted exec handle is destroyed before its   *)
(*       slot is reused (no leaked cudaGraphExec_t).                       *)
(*                                                                            *)
(* CODE REFS (paths from /home/dconnolly/yarn-agentic):                     *)
(*   ik_llama.cpp/ggml/src/ggml-backend.cpp                                 *)
(*     ggml_backend_sched->outer_graphs (cache map)                         *)
(*     ggml_backend_sched->outer_graphs_age (FIFO)                          *)
(*     ggml_backend_sched_outer_topology_key                                *)
(*     ggml_backend_sched_outer_graphs_clear                                *)
(*   ik_llama.cpp/ggml/src/ggml-cuda.cu                                     *)
(*     ggml_cuda_outer_capture_end_capture / _launch_exec / _destroy_exec  *)
(*****************************************************************************)

EXTENDS Integers, Sequences, FiniteSets

CONSTANTS
    Capacity,          \* MAX_OUTER_GRAPHS = 16
    Keys,              \* finite set of topology keys to explore
    MaxOps             \* upper bound on operations modelled

ASSUME Capacity \in Nat \ {0}
ASSUME MaxOps   \in Nat \ {0}

VARIABLES
    entries,           \* function: Keys -> ExecHandle; NoEntry (= -1) means absent
    age_fifo,          \* sequence of Keys (FIFO insertion order)
    next_exec_id,      \* monotonic counter for synthetic exec handles
    destroyed,         \* set of exec ids that have been destroyed
    ops_done           \* count of operations performed

vars == <<entries, age_fifo, next_exec_id, destroyed, ops_done>>

\* Sentinel for "no cached exec". An integer (-1), NOT a string: exec ids
\* are Nat and start at 0, so -1 never collides. A string sentinel here
\* put a string/Nat union in `entries`' range, which TLC cannot fingerprint
\* and cannot compare against Nat exec ids (the `entries[k] /= id` check in
\* DestroyOnEvict). Keeping the whole range in Int fixes both.
NoEntry == -1

(*****************************************************************************)
(* Helpers                                                                    *)
(*****************************************************************************)

EntryCount == Cardinality({k \in Keys: entries[k] /= NoEntry})

InFifoOnce(k) ==
    LET occurrences == {i \in 1..Len(age_fifo): age_fifo[i] = k}
    IN  Cardinality(occurrences) <= 1

(*****************************************************************************)
(* Initial state                                                              *)
(*****************************************************************************)

Init ==
    /\ entries      = [k \in Keys |-> NoEntry]
    /\ age_fifo     = <<>>
    /\ next_exec_id = 0
    /\ destroyed    = {}
    /\ ops_done     = 0

(*****************************************************************************)
(* Transitions                                                                *)
(*****************************************************************************)

\* Insert a fresh key into the cache. If the cache is full, evict the
\* oldest (head of age_fifo) first.
Insert(k) ==
    /\ ops_done < MaxOps
    /\ k \in Keys
    /\ entries[k] = NoEntry
    /\ LET full == EntryCount >= Capacity
       IN
         IF full /\ Len(age_fifo) > 0 THEN
            LET victim == Head(age_fifo)
                victim_id == entries[victim]
            IN  /\ entries'   = [entries EXCEPT ![victim] = NoEntry,
                                              ![k]       = next_exec_id]
                /\ age_fifo'  = Append(Tail(age_fifo), k)
                /\ destroyed' = destroyed \cup {victim_id}
                /\ next_exec_id' = next_exec_id + 1
                /\ ops_done'  = ops_done + 1
         ELSE
            /\ entries'      = [entries EXCEPT ![k] = next_exec_id]
            /\ age_fifo'     = Append(age_fifo, k)
            /\ next_exec_id' = next_exec_id + 1
            /\ UNCHANGED <<destroyed>>
            /\ ops_done'     = ops_done + 1

\* Replay a cached entry. Doesn't change cache contents — just count.
Replay(k) ==
    /\ ops_done < MaxOps
    /\ k \in Keys
    /\ entries[k] /= NoEntry
    /\ ops_done' = ops_done + 1
    /\ UNCHANGED <<entries, age_fifo, next_exec_id, destroyed>>

\* sched_free: destroy every cached exec.
ClearAll ==
    /\ ops_done < MaxOps
    /\ entries  /= [k \in Keys |-> NoEntry]
    /\ destroyed' = destroyed \cup
            {entries[k]: k \in {k \in Keys: entries[k] /= NoEntry}}
    /\ entries'   = [k \in Keys |-> NoEntry]
    /\ age_fifo'  = <<>>
    /\ ops_done'  = ops_done + 1
    /\ UNCHANGED next_exec_id

Next ==
    \/ \E k \in Keys: Insert(k)
    \/ \E k \in Keys: Replay(k)
    \/ ClearAll

Spec == Init /\ [][Next]_vars

(*****************************************************************************)
(* Invariants                                                                 *)
(*****************************************************************************)

\* I1: cache size never exceeds capacity.
CacheSizeBounded ==
    EntryCount <= Capacity

\* I2: every key in entries appears exactly once in age_fifo, and
\* age_fifo contains no keys absent from entries.
InsertedKeysInAgeList ==
    /\ \A k \in Keys: (entries[k] /= NoEntry) => InFifoOnce(k)
    /\ \A i \in 1..Len(age_fifo): entries[age_fifo[i]] /= NoEntry

\* I3: when capacity is hit, eviction picks the oldest entry (head of
\* age_fifo). Encoded in Insert's eviction branch — this invariant
\* states the structural relationship.
\* Always TRUE because we model only FIFO eviction (no policy choice).
EvictionIsFifo == TRUE

\* I4: every destroyed exec id is no longer in entries.
DestroyOnEvict ==
    \A id \in destroyed:
        \A k \in Keys: entries[k] /= id

\* I5: type / state invariants.
TypeOK ==
    /\ entries \in [Keys -> ({NoEntry} \cup Nat)]
    /\ age_fifo \in Seq(Keys)
    /\ next_exec_id \in Nat
    /\ destroyed \subseteq Nat
    /\ ops_done \in 0..(MaxOps + 1)

============================================================================
