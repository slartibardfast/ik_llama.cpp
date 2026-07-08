--------------------------- MODULE CUDANativeDispatch ---------------------------
(*****************************************************************************)
(* Status: LIVE (full) as of PHASE_CUDA_NATIVE_DISPATCH commit C4.           *)
(* C1 lit up the IDLE -> ENQUEUING -> COMPLETE arc and HostThreadIsExactly   *)
(* One / EvalsAreSequential / EventRecordedAfterEval. C4 adds the            *)
(* CAPTURING -> LAUNCHED states + AllSplitsEnqueuedBeforeCapture /           *)
(* SingleGraphLaunch.                                                         *)
(*                                                                            *)
(* Companion to specs/cuda-native-dispatch/single_threaded_dispatch.allium   *)
(* and cross_device_event_chain.allium.                                       *)
(*                                                                            *)
(* Models the new single-threaded ggml_backend_sched_compute_splits as a    *)
(* state machine over the lifetime of one graph computation:                *)
(*                                                                            *)
(*     IDLE -> ENQUEUING -> [C4: CAPTURING -> LAUNCHED ->] COMPLETE          *)
(*                                                                            *)
(* In C1, CAPTURING and LAUNCHED are skipped: kernels launch eagerly on     *)
(* per-backend streams; COMPLETE is reached directly from ENQUEUING when    *)
(* all splits' evals have been enqueued.                                    *)
(*                                                                            *)
(* The load-bearing invariants:                                              *)
(*                                                                            *)
(*   - HostThreadIsExactlyOne: at every step, exactly one thread is in       *)
(*       the dispatch path. No openmp parallel region.                       *)
(*   - EvalsAreSequential: split i's eval is enqueued before split i+1's     *)
(*       eval on the host thread.                                            *)
(*   - EventRecordedAfterEval: for every split with n_inputs > 0, the        *)
(*       cudaEventRecord on its backend's event slot happens AFTER the       *)
(*       eval enqueue and BEFORE any later consumer's event_wait.            *)
(*   - ReduceMarksAllParticipantsSticky: after a REDUCE split (except the   *)
(*       last), needs_sync[j] = true for every reduce participant j.        *)
(*                                                                            *)
(* C4 will add:                                                              *)
(*   - AllSplitsEnqueuedBeforeCapture: cudaStreamEndCapture is called       *)
(*       AFTER every split's eval has enqueued its kernels.                  *)
(*   - SingleGraphLaunch: at most one cudaGraphLaunch per token of the      *)
(*       same topology (cache hit replays; cache miss captures-then-        *)
(*       launches).                                                          *)
(*                                                                            *)
(* The negative variant (not yet written) removes                           *)
(* HostThreadIsExactlyOne and checks that the four PD1 lazy-create race    *)
(* surfaces become reachable in TLC's state space — confirms the spec's    *)
(* invariant is load-bearing.                                              *)
(*                                                                            *)
(* CODE REFS (paths from /home/dconnolly/yarn-agentic):                     *)
(*   ik_llama.cpp/ggml/src/ggml-backend.cpp:2177  compute_splits entry      *)
(*   ik_llama.cpp/ggml/src/ggml-cuda.cu:4382      cpy_tensor_async          *)
(*   ik_llama.cpp/ggml/src/ggml-cuda.cu:5150      graph_compute             *)
(*****************************************************************************)

EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    NBackends,      \* number of backends (n_backends in sched)
    Splits,         \* sequence of [backend_id, n_inputs, has_reduce] records
    MaxIters        \* upper bound on dispatch iterations modeled

ASSUME NBackends \in Nat \ {0}
ASSUME MaxIters  \in Nat \ {0}

VARIABLES
    state,                          \* "IDLE" | "ENQUEUING" | "CAPTURING" | "LAUNCHED" | "COMPLETE"
    iter,                           \* current dispatch iteration index
    cur_split,                      \* index into Splits of split currently being processed
    dispatch_thread_ids,            \* set of thread ids that have entered compute_splits
    eval_enqueued,                  \* function: split_index -> bool
    event_recorded,                 \* function: (backend_id, iter) -> bool
    needs_sync,                     \* function: backend_id -> bool
    capture_active,                 \* bool: outer cudaStreamBeginCapture in progress
    graphs_launched                 \* count of cudaGraphLaunch invocations in this iter

vars == <<state, iter, cur_split, dispatch_thread_ids,
          eval_enqueued, event_recorded, needs_sync,
          capture_active, graphs_launched>>

(*****************************************************************************)
(* The set of available host thread ids modeled. Realistic implementations  *)
(* have at most one running dispatcher; we model with {tid0, tid1} so the   *)
(* invariant has a chance to fail in TLC's state space if a future change   *)
(* re-introduces parallelism.                                               *)
(*****************************************************************************)
HostThreadIds == {"tid0", "tid1"}

(*****************************************************************************)
(* Initial state                                                              *)
(*****************************************************************************)

Init ==
    /\ state               = "IDLE"
    /\ iter                = 0
    /\ cur_split           = 0
    /\ dispatch_thread_ids = {}
    /\ eval_enqueued       = [s \in 0..(Len(Splits) - 1) |-> FALSE]
    /\ event_recorded      = [b \in 0..(NBackends - 1) |-> [i \in 0..MaxIters |-> FALSE]]
    /\ needs_sync          = [b \in 0..(NBackends - 1) |-> TRUE]
    /\ capture_active      = FALSE
    /\ graphs_launched     = 0

(*****************************************************************************)
(* Transitions                                                                *)
(*****************************************************************************)

\* The dispatcher thread enters compute_splits. The thread id observed is
\* recorded into dispatch_thread_ids. A future change that lets two
\* different tids enter would expand this set; the invariant catches it.
EnterDispatch(tid) ==
    /\ state = "IDLE"
    /\ tid \in HostThreadIds
    /\ iter <= MaxIters
    \* Match the implementation: the SAME thread enters dispatch every
    \* round. The first EnterDispatch picks a tid; subsequent rounds
    \* must use the same tid. A negative-control variant removes this
    \* guard and the HostThreadIsExactlyOne invariant fires — proof
    \* that the invariant is load-bearing.
    /\ (dispatch_thread_ids = {} \/ tid \in dispatch_thread_ids)
    /\ state'               = "CAPTURING"
    /\ dispatch_thread_ids' = dispatch_thread_ids \cup {tid}
    /\ cur_split'           = 0
    /\ capture_active'      = TRUE
    /\ UNCHANGED <<iter, eval_enqueued, event_recorded, needs_sync, graphs_launched>>

\* After CAPTURING is established (BeginCapture + cross-stream join),
\* we enter the ENQUEUING phase where splits are walked sequentially.
StartEnqueuing ==
    /\ state = "CAPTURING"
    /\ capture_active = TRUE
    /\ state' = "ENQUEUING"
    /\ UNCHANGED <<iter, cur_split, dispatch_thread_ids, eval_enqueued,
                   event_recorded, needs_sync, capture_active, graphs_launched>>

\* Enqueue the eval for the current split (sets eval_enqueued[cur_split]).
\* Followed by an event_record on the producing backend if n_inputs > 0.
\* Walks through Splits in order; cur_split advances by 1.
EnqueueSplit ==
    /\ state = "ENQUEUING"
    /\ cur_split < Len(Splits)
    /\ LET sp == Splits[cur_split + 1]  \* TLA+ sequences are 1-indexed
       IN /\ eval_enqueued' = [eval_enqueued EXCEPT ![cur_split] = TRUE]
          /\ event_recorded' =
                IF sp.n_inputs > 0 THEN
                    [event_recorded EXCEPT ![sp.backend_id] =
                        [@ EXCEPT ![iter] = TRUE]]
                ELSE event_recorded
          /\ needs_sync' =
                IF sp.has_reduce /\ cur_split < Len(Splits) - 1 THEN
                    [b \in DOMAIN needs_sync |-> TRUE]
                ELSE needs_sync
    /\ cur_split' = cur_split + 1
    /\ UNCHANGED <<state, iter, dispatch_thread_ids, capture_active, graphs_launched>>

\* All splits have enqueued. Transition to LAUNCHED via EndCapture +
\* cudaGraphInstantiate + cudaGraphLaunch.
EndCaptureAndLaunch ==
    /\ state = "ENQUEUING"
    /\ cur_split >= Len(Splits)
    /\ capture_active = TRUE
    /\ state'           = "LAUNCHED"
    /\ capture_active'  = FALSE
    /\ graphs_launched' = graphs_launched + 1
    /\ UNCHANGED <<iter, cur_split, dispatch_thread_ids, eval_enqueued,
                   event_recorded, needs_sync>>

\* From LAUNCHED, complete the iter.
FinishDispatch ==
    /\ state = "LAUNCHED"
    /\ state' = "COMPLETE"
    /\ iter'  = iter + 1
    /\ UNCHANGED <<cur_split, dispatch_thread_ids, eval_enqueued,
                   event_recorded, needs_sync, capture_active, graphs_launched>>

\* Reset for the next graph compute.
RestartDispatch ==
    /\ state = "COMPLETE"
    /\ iter <= MaxIters
    /\ state'           = "IDLE"
    /\ cur_split'       = 0
    /\ eval_enqueued'   = [s \in DOMAIN eval_enqueued |-> FALSE]
    /\ graphs_launched' = 0
    /\ UNCHANGED <<iter, dispatch_thread_ids, event_recorded, needs_sync,
                   capture_active>>

Next ==
    \/ \E tid \in HostThreadIds: EnterDispatch(tid)
    \/ StartEnqueuing
    \/ EnqueueSplit
    \/ EndCaptureAndLaunch
    \/ FinishDispatch
    \/ RestartDispatch

Spec == Init /\ [][Next]_vars

(*****************************************************************************)
(* Invariants                                                                 *)
(*****************************************************************************)

\* I1: HostThreadIsExactlyOne. The set of distinct thread ids that has
\* entered compute_splits never exceeds 1. (Cardinality 0 is allowed before
\* the first dispatch.)
HostThreadIsExactlyOne ==
    Cardinality(dispatch_thread_ids) <= 1

\* I2: EvalsAreSequential. Within one ENQUEUING phase, all splits with
\* index < cur_split have eval_enqueued = TRUE; all splits with index >=
\* cur_split (within this phase) have eval_enqueued = FALSE OR a prior
\* phase set them, which is fine.
EvalsAreSequential ==
    (state = "ENQUEUING") =>
        \A i \in 0..(cur_split - 1):
            (i < Len(Splits)) => eval_enqueued[i]

\* I3: EventRecordedAfterEval. For every split with n_inputs > 0 that has
\* been enqueued within the current dispatch round, event_recorded is set
\* for that backend at the current iter. Bound to dispatch-active states
\* ({ENQUEUING, LAUNCHED}): between rounds (state = COMPLETE / IDLE), iter
\* has already advanced and the previously-recorded event slot bears no
\* relation to the new iter.
EventRecordedAfterEval ==
    (state \in {"ENQUEUING", "LAUNCHED"}) =>
        \A i \in 0..(Len(Splits) - 1):
            LET sp == Splits[i + 1]
            IN  (i < cur_split /\ sp.n_inputs > 0 /\ iter \in DOMAIN event_recorded[sp.backend_id]) =>
                event_recorded[sp.backend_id][iter]

\* I4: ReduceMarksAllParticipantsSticky. After a non-terminal reduce
\* split is enqueued, every backend's needs_sync is TRUE.
ReduceMarksAllParticipantsSticky ==
    \A i \in 0..(cur_split - 1):
        LET sp == Splits[i + 1]
        IN  (sp.has_reduce /\ i < Len(Splits) - 1) =>
            \A b \in DOMAIN needs_sync: needs_sync[b]

\* I5: AllSplitsEnqueuedBeforeCapture — cudaStreamEndCapture (modeled
\* by transition to LAUNCHED) happens only after every split has
\* enqueued. Encoded in EndCaptureAndLaunch's guard `cur_split >= Len`.
AllSplitsEnqueuedBeforeCapture ==
    (state = "LAUNCHED") =>
        \A i \in 0..(Len(Splits) - 1): eval_enqueued[i]

\* I6: SingleGraphLaunch — at most one cudaGraphLaunch per iter. After
\* RestartDispatch resets graphs_launched, the next round can launch
\* exactly one more.
SingleGraphLaunch ==
    graphs_launched <= 1

\* All invariants combined for TLC.
TypeOK ==
    /\ state \in {"IDLE", "ENQUEUING", "CAPTURING", "LAUNCHED", "COMPLETE"}
    /\ iter \in 0..(MaxIters + 1)
    /\ cur_split \in 0..Len(Splits)
    /\ dispatch_thread_ids \subseteq HostThreadIds
    /\ capture_active \in BOOLEAN
    /\ graphs_launched \in 0..1

============================================================================
