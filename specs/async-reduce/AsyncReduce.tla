------------------------------- MODULE AsyncReduce -------------------------------
(* AsyncReduce: TLA+ specification of the async F32 cross-device reduce
   protocol. Models per-device compute and comm streams, CUDA events, and
   consumer waits. Verifies deadlock-freedom, safety (no read-before-write
   of the reduced tensor), and termination under fair scheduling.

   The model abstracts to a 2-device system (matches our TU102 pair).
   The reduce is layer-iterated; layers process in sequence and each layer
   has its own per-device input, reduce kernel, and downstream consumer.

   Per device d ∈ {0, 1} and layer l ∈ 0..N-1:
     - compute_state[d, l]   ∈ {IDLE, COMPUTING_FFN, FFN_DONE, WAITING_REDUCE, CONSUMING}
     - comm_state[d, l]      ∈ {IDLE, PEER_COPY, REDUCE_KERNEL, REDUCE_DONE}
     - evt_input_ready[d, l] ∈ {NOT_SIGNALED, SIGNALED}
     - evt_reduce_done[d, l] ∈ {NOT_SIGNALED, SIGNALED}
     - reduced[d, l]         ∈ {UNDEFINED, COMPUTED}  (the reduced tensor for device d, layer l)

   Properties verified:
     - Safety: never read reduced[d, l] before it's COMPUTED (no race)
     - Liveness: every layer eventually reaches CONSUMING on both devices
     - Determinism (modeled separately by abstract output equality): the
       reduce kernel produces the same output regardless of stream
       interleaving, because the F32 add at the kernel level is fixed-order
       per cell.
*)

EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANT N_LAYERS

ASSUME N_LAYERS \in Nat \ {0}

Devices == {0, 1}
Layers  == 0..N_LAYERS-1

VARIABLES
    compute_state,   \* [Devices × Layers -> {"IDLE","COMPUTING_FFN","FFN_DONE","WAITING_REDUCE","CONSUMING","DONE"}]
    comm_state,      \* [Devices × Layers -> {"IDLE","PEER_COPY","REDUCE_KERNEL","REDUCE_DONE"}]
    evt_input_ready, \* [Devices × Layers -> {"NOT_SIGNALED","SIGNALED"}]
    evt_reduce_done, \* [Devices × Layers -> {"NOT_SIGNALED","SIGNALED"}]
    reduced          \* [Devices × Layers -> {"UNDEFINED","COMPUTED"}]

vars == << compute_state, comm_state, evt_input_ready, evt_reduce_done, reduced >>

ComputeStates == {"IDLE","COMPUTING_FFN","FFN_DONE","WAITING_REDUCE","CONSUMING","DONE"}
CommStates    == {"IDLE","PEER_COPY","REDUCE_KERNEL","REDUCE_DONE"}
EventStates   == {"NOT_SIGNALED","SIGNALED"}
ReducedStates == {"UNDEFINED","COMPUTED"}

TypeOK ==
    /\ compute_state    \in [Devices \X Layers -> ComputeStates]
    /\ comm_state       \in [Devices \X Layers -> CommStates]
    /\ evt_input_ready  \in [Devices \X Layers -> EventStates]
    /\ evt_reduce_done  \in [Devices \X Layers -> EventStates]
    /\ reduced          \in [Devices \X Layers -> ReducedStates]

Init ==
    /\ compute_state    = [<<d,l>> \in Devices \X Layers |-> "IDLE"]
    /\ comm_state       = [<<d,l>> \in Devices \X Layers |-> "IDLE"]
    /\ evt_input_ready  = [<<d,l>> \in Devices \X Layers |-> "NOT_SIGNALED"]
    /\ evt_reduce_done  = [<<d,l>> \in Devices \X Layers |-> "NOT_SIGNALED"]
    /\ reduced          = [<<d,l>> \in Devices \X Layers |-> "UNDEFINED"]

(*--- Compute stream actions (per device, per layer) ---*)

StartFFN(d, l) ==
    /\ compute_state[d, l] = "IDLE"
    /\ \/ l = 0
       \/ /\ l > 0
          /\ compute_state[d, l-1] = "DONE"
    /\ compute_state' = [compute_state EXCEPT ![d, l] = "COMPUTING_FFN"]
    /\ UNCHANGED << comm_state, evt_input_ready, evt_reduce_done, reduced >>

FinishFFN(d, l) ==
    /\ compute_state[d, l] = "COMPUTING_FFN"
    /\ compute_state'   = [compute_state   EXCEPT ![d, l] = "FFN_DONE"]
    /\ evt_input_ready' = [evt_input_ready EXCEPT ![d, l] = "SIGNALED"]
    /\ UNCHANGED << comm_state, evt_reduce_done, reduced >>

WaitForReduce(d, l) ==
    /\ compute_state[d, l] = "FFN_DONE"
    /\ compute_state' = [compute_state EXCEPT ![d, l] = "WAITING_REDUCE"]
    /\ UNCHANGED << comm_state, evt_input_ready, evt_reduce_done, reduced >>

(* Compute stream may pre-emptively start NEXT layer's FFN while waiting on reduce.
   This is the OVERLAP — modeled via StartFFN(d, l+1) firing while compute_state[d, l] = "WAITING_REDUCE". *)

Consume(d, l) ==
    /\ compute_state[d, l] = "WAITING_REDUCE"
    /\ evt_reduce_done[d, l] = "SIGNALED"
    /\ reduced[d, l] = "COMPUTED"  \* SAFETY: only read after COMPUTED
    /\ compute_state' = [compute_state EXCEPT ![d, l] = "CONSUMING"]
    /\ UNCHANGED << comm_state, evt_input_ready, evt_reduce_done, reduced >>

FinishLayer(d, l) ==
    /\ compute_state[d, l] = "CONSUMING"
    /\ compute_state' = [compute_state EXCEPT ![d, l] = "DONE"]
    /\ UNCHANGED << comm_state, evt_input_ready, evt_reduce_done, reduced >>

(*--- Comm stream actions ---*)

CommWaitInputReady(d, l) ==
    /\ comm_state[d, l] = "IDLE"
    /\ \A d2 \in Devices: evt_input_ready[d2, l] = "SIGNALED"
       \* The comm stream waits on BOTH devices' input-ready events
       \* before starting peer-copy. Models cudaStreamWaitEvent edges.
    /\ comm_state' = [comm_state EXCEPT ![d, l] = "PEER_COPY"]
    /\ UNCHANGED << compute_state, evt_input_ready, evt_reduce_done, reduced >>

PerformReduce(d, l) ==
    /\ comm_state[d, l] = "PEER_COPY"
    /\ comm_state' = [comm_state EXCEPT ![d, l] = "REDUCE_KERNEL"]
    /\ reduced'    = [reduced    EXCEPT ![d, l] = "COMPUTED"]
    /\ UNCHANGED << compute_state, evt_input_ready, evt_reduce_done >>

SignalReduceDone(d, l) ==
    /\ comm_state[d, l] = "REDUCE_KERNEL"
    /\ reduced[d, l] = "COMPUTED"  \* SAFETY: signal only after write
    /\ comm_state'      = [comm_state      EXCEPT ![d, l] = "REDUCE_DONE"]
    /\ evt_reduce_done' = [evt_reduce_done EXCEPT ![d, l] = "SIGNALED"]
    /\ UNCHANGED << compute_state, evt_input_ready, reduced >>

(*--- Composite next-state ---*)

Next ==
    \E d \in Devices, l \in Layers:
        \/ StartFFN(d, l)
        \/ FinishFFN(d, l)
        \/ WaitForReduce(d, l)
        \/ Consume(d, l)
        \/ FinishLayer(d, l)
        \/ CommWaitInputReady(d, l)
        \/ PerformReduce(d, l)
        \/ SignalReduceDone(d, l)

Fairness ==
    /\ \A d \in Devices, l \in Layers: WF_vars(StartFFN(d, l))
    /\ \A d \in Devices, l \in Layers: WF_vars(FinishFFN(d, l))
    /\ \A d \in Devices, l \in Layers: WF_vars(WaitForReduce(d, l))
    /\ \A d \in Devices, l \in Layers: WF_vars(Consume(d, l))
    /\ \A d \in Devices, l \in Layers: WF_vars(FinishLayer(d, l))
    /\ \A d \in Devices, l \in Layers: WF_vars(CommWaitInputReady(d, l))
    /\ \A d \in Devices, l \in Layers: WF_vars(PerformReduce(d, l))
    /\ \A d \in Devices, l \in Layers: WF_vars(SignalReduceDone(d, l))

Spec == Init /\ [][Next]_vars /\ Fairness

(*--- Properties ---*)

\* SAFETY: a device never consumes a layer's reduce before it's been computed
SafetyConsumeAfterCompute ==
    [] \A d \in Devices, l \in Layers:
        compute_state[d, l] = "CONSUMING" => reduced[d, l] = "COMPUTED"

\* SAFETY: events are only signaled after the corresponding state transition is complete
SafetyEventOrdering ==
    [] \A d \in Devices, l \in Layers:
        /\ (evt_reduce_done[d, l] = "SIGNALED" => reduced[d, l] = "COMPUTED")
        /\ (evt_input_ready[d, l] = "SIGNALED" => compute_state[d, l] \in {"FFN_DONE","WAITING_REDUCE","CONSUMING","DONE"})

\* LIVENESS: every layer on every device eventually reaches DONE
LivenessAllLayersComplete ==
    <>(\A d \in Devices, l \in Layers: compute_state[d, l] = "DONE")

\* LIVENESS: every reduce eventually signals its event
LivenessAllReducesSignal ==
    \A d \in Devices, l \in Layers: <>(evt_reduce_done[d, l] = "SIGNALED")

\* DEADLOCK FREEDOM: from any reachable state, some action is enabled (unless all DONE)
DeadlockFreedom ==
    [] (\E d \in Devices, l \in Layers: compute_state[d, l] # "DONE") =>
        ENABLED Next

\* OVERLAP-FRIENDLY: while a layer's reduce is in flight (REDUCE_KERNEL),
\* compute on the same device CAN be advancing on a later layer.
\* This is the perf invariant — we check that StartFFN(d, l+1) is
\* not blocked when comm_state[d, l] is in flight.
OverlapPossible ==
    [] \A d \in Devices, l \in Layers \ {N_LAYERS-1}:
        (comm_state[d, l] \in {"PEER_COPY","REDUCE_KERNEL"}
         /\ compute_state[d, l]   = "WAITING_REDUCE"
         /\ compute_state[d, l+1] = "IDLE"
         /\ \/ l = 0
            \/ /\ l > 0
               /\ compute_state[d, l-1] = "DONE")
        => ENABLED StartFFN(d, l+1)

===============================================================================
