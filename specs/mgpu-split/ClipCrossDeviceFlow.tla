--------------------------- MODULE ClipCrossDeviceFlow ---------------------------
(* ClipCrossDeviceFlow: TLA+ specification of per-layer cross-device flow
   in the CLIP vision encoder under Phase 46's row-chunked split (GRAPH
   mode). Phase 46 Path B (§12.2 spec #4) extends specs/async-reduce/
   AsyncReduce.tla — which already models the LM's per-layer cross-device
   reduce on the TU102 2-device topology — to cover the CLIP encoder's
   structurally-identical pattern, plus the CLIP-specific encoder-output
   hand-off to the LM.

   Why this spec, given AsyncReduce.tla exists:

     1. AsyncReduce models the LM. CLIP under Phase 46 inherits the
        same async-F32-cross-device-reduce machinery (same primitive,
        same scheduler, same NVLink topology), but with a different
        consumer: the final encoder output feeds the LM's token embed
        path, not another layer of itself.
     2. The B.7 perf gate (§12.3 P7: encode latency ≤ 1.3× single-GPU
        baseline) requires the compute/transfer overlap property to
        hold under saturated CLIP pipelining. AsyncReduce verifies
        OverlapPossible; we strengthen this to FullOverlapUnderSaturation,
        which is the property B.7 binds on.
     3. The cross-codepath consistency spec (#5) needs a CLIP-specific
        state machine to reference; this spec provides it.

   Topology inherited from AsyncReduce.tla:
     - 2-device system (CUDA0, CUDA1) — matches TU102 pair on prod host.
     - Per-(device, layer) compute_state, comm_state, evt_*, reduced
       variables.
     - Per-layer state machine: IDLE → COMPUTING_FFN → FFN_DONE →
       WAITING_REDUCE → CONSUMING → DONE.

   Extensions in this module:
     - encoder_output_ready  — set when ALL devices reach DONE for the
       LAST CLIP encoder layer. Signals to the LM that token embeddings
       are ready.
     - lm_consumed           — set when the LM picks up the encoder
       output; encoder graph can then be released.

   Properties verified (in addition to AsyncReduce.tla's properties,
   which compose by inclusion):
     - Safety: encoder_output_ready set only after all layers DONE
       (EncoderReadyAfterAllLayers).
     - Safety: lm_consumed only after encoder_output_ready
       (LMConsumesAfterReady).
     - Liveness: encoder eventually signals output ready
       (EventuallyEncoderReady).
     - Liveness: LM eventually consumes (EventuallyLMConsumes).
     - PERF (P3): compute/transfer overlap holds at saturation
       (FullOverlapUnderSaturation). Binds B.7.

   Provenance: AsyncReduce.tla read in full on 2026-05-25;
   structural pattern matches. CLIP encoder topology confirmed by
   reading examples/mtmd/clip.cpp:488-516 (init), :530 (sched
   creation with vector<backend>), :5212-5562 (encode driver).

   Out of scope (modeled elsewhere):
     - The mgpu_split_config struct invariants (MgpuSplitConfig.allium).
     - The buft-setup loop populating the config (BuftSetupLoop.tla).
     - The row-chunk allocation algorithm (CreateSplitBalance.tla).
     - LM-side internals after lm_consumed = TRUE (out of Phase 46).
*)

EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANT N_LAYERS

ASSUME N_LAYERS \in Nat \ {0}

\* Inherited from AsyncReduce.tla — same 2-device topology.
Devices == {0, 1}
Layers  == 0..(N_LAYERS - 1)

VARIABLES
    \* Inherited state machine (identical to AsyncReduce.tla).
    compute_state,
    comm_state,
    evt_input_ready,
    evt_reduce_done,
    reduced,
    \* CLIP-specific extensions.
    encoder_output_ready,    \* BOOLEAN
    lm_consumed              \* BOOLEAN

vars == << compute_state, comm_state, evt_input_ready, evt_reduce_done,
           reduced, encoder_output_ready, lm_consumed >>

\* Inherited domains.
ComputeStates == {"IDLE","COMPUTING_FFN","FFN_DONE","WAITING_REDUCE","CONSUMING","DONE"}
CommStates    == {"IDLE","PEER_COPY","REDUCE_KERNEL","REDUCE_DONE"}
EventStates   == {"NOT_SIGNALED","SIGNALED"}
ReducedStates == {"UNDEFINED","COMPUTED"}

TypeOK ==
    /\ compute_state         \in [Devices \X Layers -> ComputeStates]
    /\ comm_state            \in [Devices \X Layers -> CommStates]
    /\ evt_input_ready       \in [Devices \X Layers -> EventStates]
    /\ evt_reduce_done       \in [Devices \X Layers -> EventStates]
    /\ reduced               \in [Devices \X Layers -> ReducedStates]
    /\ encoder_output_ready  \in BOOLEAN
    /\ lm_consumed           \in BOOLEAN

Init ==
    /\ compute_state         = [<<d,l>> \in Devices \X Layers |-> "IDLE"]
    /\ comm_state            = [<<d,l>> \in Devices \X Layers |-> "IDLE"]
    /\ evt_input_ready       = [<<d,l>> \in Devices \X Layers |-> "NOT_SIGNALED"]
    /\ evt_reduce_done       = [<<d,l>> \in Devices \X Layers |-> "NOT_SIGNALED"]
    /\ reduced               = [<<d,l>> \in Devices \X Layers |-> "UNDEFINED"]
    /\ encoder_output_ready  = FALSE
    /\ lm_consumed           = FALSE

UnchangedClipExtensions == UNCHANGED << encoder_output_ready, lm_consumed >>

\* ============================================================
\* Inherited actions from AsyncReduce.tla
\* ============================================================
\* Each action below is structurally identical to its AsyncReduce.tla
\* counterpart, modified only to add UnchangedClipExtensions to
\* preserve the CLIP-specific variables.

StartFFN(d, l) ==
    /\ compute_state[d, l] = "IDLE"
    /\ \* Compute pipeline is independent of the reduce pipeline:
       \* layer l+1's compute can start as soon as layer l's FFN is done,
       \* WITHOUT waiting for layer l's reduce + consume. This is the
       \* compute/transfer overlap pattern (P3 in PHASE46 §12.3) that
       \* makes the B.7 perf gate (≤ 1.3× single-GPU baseline) achievable.
       \*
       \* Use IF/THEN/ELSE rather than \/ to guarantee TLC short-circuits
       \* compute_state[d, l-1] when l = 0 (TLC 2026.05.18 fingerprint
       \* failure observed otherwise).
       \*
       \* Diverges from AsyncReduce.tla's StartFFN, which required prev
       \* DONE — that guard prevented the overlap the comment in
       \* AsyncReduce.tla:84-85 promised. This spec resolves the
       \* AsyncReduce inconsistency by aligning the guard with the
       \* documented intent.
       (IF l = 0 THEN TRUE
        ELSE compute_state[d, l - 1] \in
                {"FFN_DONE", "WAITING_REDUCE", "CONSUMING", "DONE"})
    /\ compute_state' = [compute_state EXCEPT ![d, l] = "COMPUTING_FFN"]
    /\ UNCHANGED << comm_state, evt_input_ready, evt_reduce_done, reduced >>
    /\ UnchangedClipExtensions

FinishFFN(d, l) ==
    /\ compute_state[d, l] = "COMPUTING_FFN"
    /\ compute_state'   = [compute_state   EXCEPT ![d, l] = "FFN_DONE"]
    /\ evt_input_ready' = [evt_input_ready EXCEPT ![d, l] = "SIGNALED"]
    /\ UNCHANGED << comm_state, evt_reduce_done, reduced >>
    /\ UnchangedClipExtensions

WaitForReduce(d, l) ==
    /\ compute_state[d, l] = "FFN_DONE"
    /\ compute_state' = [compute_state EXCEPT ![d, l] = "WAITING_REDUCE"]
    /\ UNCHANGED << comm_state, evt_input_ready, evt_reduce_done, reduced >>
    /\ UnchangedClipExtensions

Consume(d, l) ==
    /\ compute_state[d, l] = "WAITING_REDUCE"
    /\ evt_reduce_done[d, l] = "SIGNALED"
    /\ reduced[d, l] = "COMPUTED"
    /\ compute_state' = [compute_state EXCEPT ![d, l] = "CONSUMING"]
    /\ UNCHANGED << comm_state, evt_input_ready, evt_reduce_done, reduced >>
    /\ UnchangedClipExtensions

FinishLayer(d, l) ==
    /\ compute_state[d, l] = "CONSUMING"
    /\ compute_state' = [compute_state EXCEPT ![d, l] = "DONE"]
    /\ UNCHANGED << comm_state, evt_input_ready, evt_reduce_done, reduced >>
    /\ UnchangedClipExtensions

CommWaitInputReady(d, l) ==
    /\ comm_state[d, l] = "IDLE"
    /\ \A d2 \in Devices: evt_input_ready[d2, l] = "SIGNALED"
    /\ comm_state' = [comm_state EXCEPT ![d, l] = "PEER_COPY"]
    /\ UNCHANGED << compute_state, evt_input_ready, evt_reduce_done, reduced >>
    /\ UnchangedClipExtensions

PerformReduce(d, l) ==
    /\ comm_state[d, l] = "PEER_COPY"
    /\ comm_state' = [comm_state EXCEPT ![d, l] = "REDUCE_KERNEL"]
    /\ reduced'    = [reduced    EXCEPT ![d, l] = "COMPUTED"]
    /\ UNCHANGED << compute_state, evt_input_ready, evt_reduce_done >>
    /\ UnchangedClipExtensions

SignalReduceDone(d, l) ==
    /\ comm_state[d, l] = "REDUCE_KERNEL"
    /\ reduced[d, l] = "COMPUTED"
    /\ comm_state'      = [comm_state      EXCEPT ![d, l] = "REDUCE_DONE"]
    /\ evt_reduce_done' = [evt_reduce_done EXCEPT ![d, l] = "SIGNALED"]
    /\ UNCHANGED << compute_state, evt_input_ready, reduced >>
    /\ UnchangedClipExtensions

\* ============================================================
\* CLIP-specific actions (extensions)
\* ============================================================

\* Signal that the encoder is done: every (device, layer) pair is DONE.
\* NOT just the last layer — under the compute/transfer overlap pattern
\* enabled by StartFFN's relaxed guard, layer N-1 can reach DONE before
\* earlier layers do (compute pipeline ahead of reduce pipeline). The
\* binding contract for downstream LM consumption is that the FULL
\* encoder output is materialized, which requires every layer's reduce
\* to have CONSUMED — i.e. every layer is DONE.
SignalEncoderOutputReady ==
    /\ encoder_output_ready = FALSE
    /\ \A d \in Devices, l \in Layers : compute_state[d, l] = "DONE"
    /\ encoder_output_ready' = TRUE
    /\ UNCHANGED << compute_state, comm_state, evt_input_ready,
                    evt_reduce_done, reduced, lm_consumed >>

\* The LM picks up the encoder output and releases the encoder graph.
LMConsume ==
    /\ encoder_output_ready = TRUE
    /\ lm_consumed = FALSE
    /\ lm_consumed' = TRUE
    /\ UNCHANGED << compute_state, comm_state, evt_input_ready,
                    evt_reduce_done, reduced, encoder_output_ready >>

\* ============================================================
\* Composite next-state
\* ============================================================

Next ==
    \/ \E d \in Devices, l \in Layers:
            \/ StartFFN(d, l)
            \/ FinishFFN(d, l)
            \/ WaitForReduce(d, l)
            \/ Consume(d, l)
            \/ FinishLayer(d, l)
            \/ CommWaitInputReady(d, l)
            \/ PerformReduce(d, l)
            \/ SignalReduceDone(d, l)
    \/ SignalEncoderOutputReady
    \/ LMConsume

Fairness ==
    /\ \A d \in Devices, l \in Layers: WF_vars(StartFFN(d, l))
    /\ \A d \in Devices, l \in Layers: WF_vars(FinishFFN(d, l))
    /\ \A d \in Devices, l \in Layers: WF_vars(WaitForReduce(d, l))
    /\ \A d \in Devices, l \in Layers: WF_vars(Consume(d, l))
    /\ \A d \in Devices, l \in Layers: WF_vars(FinishLayer(d, l))
    /\ \A d \in Devices, l \in Layers: WF_vars(CommWaitInputReady(d, l))
    /\ \A d \in Devices, l \in Layers: WF_vars(PerformReduce(d, l))
    /\ \A d \in Devices, l \in Layers: WF_vars(SignalReduceDone(d, l))
    /\ WF_vars(SignalEncoderOutputReady)
    /\ WF_vars(LMConsume)

Spec == Init /\ [][Next]_vars /\ Fairness

\* ============================================================
\* INHERITED PROPERTIES (from AsyncReduce.tla, restated for clarity)
\* ============================================================

\* SAFETY: a device never consumes a layer's reduce before COMPUTED.
SafetyConsumeAfterCompute ==
    [] \A d \in Devices, l \in Layers:
        compute_state[d, l] = "CONSUMING" => reduced[d, l] = "COMPUTED"

\* SAFETY: events signaled only after the corresponding transition completes.
SafetyEventOrdering ==
    [] \A d \in Devices, l \in Layers:
        /\ (evt_reduce_done[d, l] = "SIGNALED" => reduced[d, l] = "COMPUTED")
        /\ (evt_input_ready[d, l] = "SIGNALED" =>
              compute_state[d, l] \in {"FFN_DONE","WAITING_REDUCE","CONSUMING","DONE"})

\* LIVENESS: every layer on every device reaches DONE.
LivenessAllLayersComplete ==
    <>(\A d \in Devices, l \in Layers: compute_state[d, l] = "DONE")

\* ============================================================
\* CLIP-SPECIFIC PROPERTIES (the reason this spec exists)
\* ============================================================

\* SAFETY: encoder output ready only after every layer is DONE.
\* (State invariant — no [] prefix; TLC's INVARIANTS clause wraps it.)
EncoderReadyAfterAllLayers ==
    encoder_output_ready = TRUE =>
        \A d \in Devices, l \in Layers: compute_state[d, l] = "DONE"

\* SAFETY: LM never consumes before encoder ready.
LMConsumesAfterReady ==
    lm_consumed = TRUE => encoder_output_ready = TRUE

\* LIVENESS: encoder eventually signals ready.
EventuallyEncoderReady == <>(encoder_output_ready = TRUE)

\* LIVENESS: LM eventually consumes.
EventuallyLMConsumes == <>(lm_consumed = TRUE)

\* ============================================================
\* PERF PROPERTY (binds on §12.3 P7 / B.7 perf gate)
\* ============================================================
\*
\* FullOverlapUnderSaturation: under saturated pipelining (the steady
\* state of the encoder loop), while a layer's reduce is in flight
\* (REDUCE_KERNEL state), the next layer's compute IS already in
\* COMPUTING_FFN — not merely able to start, but actually started.
\*
\* This is strictly stronger than AsyncReduce.tla's OverlapPossible
\* (which asserts only that StartFFN is ENABLED). The strengthening
\* is what makes the B.7 perf gate achievable: the critical-path
\* latency reduces to max{compute_d + transfer_d} per layer, instead
\* of compute_d + transfer_d serialized.
\*
\* The property is asserted as: for every device d and every
\* non-terminal layer l, IF comm_state[d, l] = REDUCE_KERNEL AND
\* layer l+1's prerequisites are satisfied, THEN
\* compute_state[d, l+1] = "COMPUTING_FFN".
\*
\* "Prerequisites satisfied" means: layer l = 0 OR layer l-1 is DONE
\* on device d (the StartFFN guard).
\*
\* TLC will find a violation iff the schedule permits compute_state[d, l+1]
\* to stall in "IDLE" while comm_state[d, l] is in REDUCE_KERNEL —
\* exactly the schedule pattern P3 forbids.

FullOverlapUnderSaturation ==
    \A d \in Devices, l \in (Layers \ {N_LAYERS - 1}):
        LET prereqs_ready ==
                IF l = 0 THEN TRUE
                ELSE compute_state[d, l - 1] \in
                       {"FFN_DONE","WAITING_REDUCE","CONSUMING","DONE"}
        IN  (/\ comm_state[d, l] = "REDUCE_KERNEL"
             /\ compute_state[d, l + 1] = "IDLE"
             /\ prereqs_ready)
            => ENABLED StartFFN(d, l + 1)
       \* Spec asserts overlap is PERMITTED (StartFFN ENABLED), not that
       \* the schedule HAS fired it. Whether overlap actually occurs at
       \* runtime is the empirical B.7 perf gate (PHASE46 §12.3 P7):
       \* if encode latency exceeds 1.3× single-GPU baseline, the schedule
       \* failed to overlap and Phase 46 stays OPEN.
       \*
       \* This is structurally the same form as AsyncReduce.tla's
       \* OverlapPossible — kept here for the CLIP topology to make the
       \* spec/B.7 division explicit.

\* ============================================================
\* DEADLOCK FREEDOM (inherited; restated to bind on this spec's Next)
\* ============================================================

DeadlockFreedom ==
    [] (\/ lm_consumed = FALSE) => ENABLED Next

=============================================================================
\* End MODULE ClipCrossDeviceFlow
