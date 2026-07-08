--------------------------- MODULE CbEvalObservational ---------------------------
(*****************************************************************************)
(* TLA+ spec for the observational equivalence of llama_decode under         *)
(* residual extraction.                                                       *)
(*                                                                            *)
(* Companion to the Allium spec                                              *)
(*   specs/dflash/cb_eval_residual_capture.allium                            *)
(*                                                                            *)
(* P0.A.3 root cause (2026-05-20 falsification matrix): the cb_eval hook    *)
(* installed by llama_set_dflash_extract_layers (src/llama.cpp:10072)        *)
(* perturbs the target's forward output because ggml_backend_sched_eval     *)
(* (ggml-backend.cpp:2126-2173) iterates the per-split cgraph node-by-node *)
(* whenever a callback is installed, breaking the split into N sub-graphs  *)
(* separated by ggml_backend_synchronize. Each sub-invocation lets the      *)
(* CUDA backend re-plan fusion / cudaGraph capture / cuBLAS algo selection *)
(* on a different node range than the fast path; the resulting argmax       *)
(* drift cascades through 65 layers and breaks DFlash verify.               *)
(*                                                                            *)
(* The spec models two decoder contexts side by side: one with extract     *)
(* disarmed (baseline), one with extract armed (test). The state space     *)
(* includes the implementation MECHANISM (scheduler_callback vs            *)
(* graph_tap_node). The load-bearing invariant is                          *)
(* ObservationalEquivalence: at every step where Decode has fired on both *)
(* contexts, their argmax token streams agree.                              *)
(*                                                                            *)
(* The negative-test config (CbEvalObservationalMC_callback.cfg)           *)
(* installs the scheduler_callback mechanism on the test context. TLC      *)
(* must find a counterexample showing the invariant violated. The positive *)
(* config (CbEvalObservationalMC.cfg) installs graph_tap_node and verifies *)
(* the invariant holds.                                                     *)
(*                                                                            *)
(* The model is intentionally abstract on the actual transformer math:    *)
(* "argmax token" is a value drawn from a fixed Tokens set, parameterised *)
(* by (prompt, mechanism, extract_armed). The Perturbed function captures *)
(* the observational fact that scheduler_callback + extract_armed produces *)
(* a different value than the baseline; nothing else perturbs.             *)
(*                                                                            *)
(* CODE REFS (paths from /home/llm/yarn-agentic):                          *)
(*   ik_llama.cpp/src/llama.cpp:9961-10049    llama_dflash_extract_cb_eval *)
(*   ik_llama.cpp/src/llama.cpp:10051-10082   llama_set_dflash_extract_layers *)
(*   ik_llama.cpp/ggml/src/ggml-backend.cpp:1173-1174  callback_eval field *)
(*   ik_llama.cpp/ggml/src/ggml-backend.cpp:2126-2173  ggml_backend_sched_eval *)
(*   ik_llama.cpp/src/graphs/build_qwen35.cpp:96-99   MTP h_pre_norm tap   *)
(*   ik_llama.cpp/src/graphs/build_qwen35.cpp:205-208 dense h_pre_norm tap *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Prompts,            \* set of opaque prompt identifiers
    BaselineToken,      \* token value emitted by the un-perturbed decoder
    PerturbedToken,     \* token value emitted by the cb_eval-perturbed decoder
    MaxStep,            \* bound on global tick counter for finite MC
    Mechanism           \* in {"NONE", "GRAPH_TAP_NODE", "SCHEDULER_CALLBACK"}
                        \* — the mechanism used in the TEST context only;
                        \* the BASELINE context always uses "NONE".

VARIABLES
    test_armed,         \* BOOLEAN — extract enabled on the test context
    test_tokens,        \* Seq(tokens emitted so far) by the test context
    base_tokens,        \* Seq(tokens emitted so far) by the baseline context
    sched_path,         \* "FAST" | "SLOW" — which scheduler branch the test
                        \* context's last Decode took
    sched_has_cb,       \* BOOLEAN — does the test context's scheduler have a
                        \* callback installed right now?
    step_count

vars == <<test_armed, test_tokens, base_tokens, sched_path, sched_has_cb,
          step_count>>

----------------------------------------------------------------------------
(* Constants and helpers.                                                    *)
----------------------------------------------------------------------------

Mechanisms == {"NONE", "GRAPH_TAP_NODE", "SCHEDULER_CALLBACK"}

\* Predicate: the mechanism keeps the scheduler on its fast path.
\* Only SCHEDULER_CALLBACK forces the slow per-node iteration path
\* (ggml-backend.cpp:2136-2173). NONE and GRAPH_TAP_NODE both keep the
\* scheduler on the single-graph_compute_async fast path at lines 2127-2135.
KeepsFastPath(m) ==
    \/ m = "NONE"
    \/ m = "GRAPH_TAP_NODE"

\* The argmax token a context emits this step, given the active mechanism
\* and whether extract is armed. The only case that emits PerturbedToken
\* is (armed = TRUE /\ mechanism = SCHEDULER_CALLBACK). Every other case
\* emits BaselineToken. This is the empirical fact P0.A.3 confirmed:
\*   spec-none  (armed=FALSE)              -> baseline
\*   ngram-simple (armed=FALSE)            -> baseline
\*   dflash (armed=TRUE, cb_eval installed) -> perturbed
\* and the architectural fact we want to assert post-fix:
\*   dflash (armed=TRUE, tap-node mechanism) -> baseline
EmitsToken(armed, mechanism) ==
    IF armed /\ mechanism = "SCHEDULER_CALLBACK"
    THEN PerturbedToken
    ELSE BaselineToken

----------------------------------------------------------------------------
(* Init.                                                                     *)
----------------------------------------------------------------------------
Init ==
    /\ test_armed = FALSE
    /\ test_tokens = <<>>
    /\ base_tokens = <<>>
    /\ sched_path = "FAST"
    /\ sched_has_cb = FALSE
    /\ step_count = 0

----------------------------------------------------------------------------
(* Action: ArmExtract                                                        *)
(*                                                                           *)
(* The test context calls llama_set_dflash_extract_layers with N>0. Whether *)
(* this installs a scheduler callback depends on the active Mechanism:      *)
(*   - SCHEDULER_CALLBACK: sched_has_cb' = TRUE                              *)
(*   - GRAPH_TAP_NODE:     sched_has_cb' = FALSE (tap nodes are emitted by  *)
(*                         the graph builder, not via callback)             *)
(*   - NONE:               armed stays FALSE — this branch only fires when *)
(*                         the mechanism is set; we model the no-op.        *)
(*****************************************************************************)
ArmExtract ==
    /\ ~test_armed
    /\ Mechanism /= "NONE"
    /\ test_armed' = TRUE
    /\ sched_has_cb' = (Mechanism = "SCHEDULER_CALLBACK")
    /\ UNCHANGED <<test_tokens, base_tokens, sched_path, step_count>>

----------------------------------------------------------------------------
(* Action: Decode                                                            *)
(*                                                                           *)
(* Both contexts run one llama_decode on the same input. The base context  *)
(* always uses mechanism NONE; the test context uses the constant          *)
(* Mechanism (if armed) or NONE (if disarmed). The scheduler path is        *)
(* updated according to whether sched_has_cb is set right now.              *)
(*****************************************************************************)
Decode ==
    /\ step_count < MaxStep
    /\ LET active_mechanism == IF test_armed THEN Mechanism ELSE "NONE"
           tok_test == EmitsToken(test_armed, active_mechanism)
           tok_base == BaselineToken
       IN
       /\ test_tokens' = Append(test_tokens, tok_test)
       /\ base_tokens' = Append(base_tokens, tok_base)
       /\ sched_path'  = IF sched_has_cb THEN "SLOW" ELSE "FAST"
    /\ step_count' = step_count + 1
    /\ UNCHANGED <<test_armed, sched_has_cb>>

----------------------------------------------------------------------------
(* Action: DisarmExtract                                                     *)
(*                                                                           *)
(* The test context calls llama_set_dflash_extract_layers with N=0. The     *)
(* callback (if any) is uninstalled (per                                    *)
(* llama_set_dflash_extract_layers src/llama.cpp:10074-10081, the field     *)
(* is cleared IFF the callback pointer equals the dflash-installed one).   *)
(* Models the DisarmReleasesScheduler invariant.                            *)
(*****************************************************************************)
DisarmExtract ==
    /\ test_armed
    /\ test_armed' = FALSE
    /\ sched_has_cb' = FALSE
    /\ UNCHANGED <<test_tokens, base_tokens, sched_path, step_count>>

----------------------------------------------------------------------------
(* Next-state.                                                               *)
----------------------------------------------------------------------------
Next ==
    \/ ArmExtract
    \/ Decode
    \/ DisarmExtract

Spec == Init /\ [][Next]_vars /\ WF_vars(Decode)

----------------------------------------------------------------------------
(* Invariants.                                                               *)
----------------------------------------------------------------------------

TypeOK ==
    /\ test_armed \in BOOLEAN
    /\ test_tokens \in Seq({BaselineToken, PerturbedToken})
    /\ base_tokens \in Seq({BaselineToken})
    /\ sched_path \in {"FAST", "SLOW"}
    /\ sched_has_cb \in BOOLEAN
    /\ step_count \in 0..MaxStep
    /\ Mechanism \in Mechanisms

\* The load-bearing safety property. At every reachable state, the two   *
\* contexts' emitted token streams are equal. This is the spec-level     *
\* binding of ResidualExtractObservationallyEquivalent.                  *
ObservationalEquivalence ==
    test_tokens = base_tokens

\* The mechanism-level invariant. Whenever the test context is currently *
\* armed, the scheduler must not have a callback installed. Together     *
\* with KeepsFastPath this captures the                                  *
\* CaptureMechanismIsTapNode + SchedulerFastPath pair from the Allium    *
\* spec. Property-based tests propagated off this invariant can inspect  *
\* the underlying llama_context.cparams.cb_eval pointer.                 *
SchedulerStaysFastPath ==
    test_armed => ~sched_has_cb

\* Composite: the spec layer's full statement. If a future implementation*
\* picks a different mechanism that ALSO keeps the fast path, the spec   *
\* still holds; the negative test is what binds on the cb_eval-specific  *
\* failure mode.                                                          *
ExtractIsObservational ==
    /\ ObservationalEquivalence
    /\ SchedulerStaysFastPath

----------------------------------------------------------------------------
(* Liveness — every armed context eventually fires Decode at least once.    *)
(*                                                                           *)
(* This rules out a degenerate model where the spec passes by virtue of    *)
(* Decode never being taken. The fairness on Decode in Spec, combined with *)
(* the EventuallyDecodes property, asserts that the implementation is      *)
(* exercised after arming.                                                  *)
----------------------------------------------------------------------------
EventuallyDecodes ==
    (test_armed => <>(Len(test_tokens) > 0))

==============================================================================
