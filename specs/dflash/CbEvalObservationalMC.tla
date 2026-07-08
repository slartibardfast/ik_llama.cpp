--------------------------- MODULE CbEvalObservationalMC ---------------------------
(*****************************************************************************)
(* MC wrapper for CbEvalObservational.tla.                                   *)
(*                                                                            *)
(* Two cfg files use this wrapper:                                           *)
(*   CbEvalObservationalMC.cfg          — POSITIVE: Mechanism = GRAPH_TAP_NODE *)
(*   CbEvalObservationalMC_callback.cfg — NEGATIVE: Mechanism =              *)
(*                                         SCHEDULER_CALLBACK; counterexample *)
(*                                         expected on ObservationalEquivalence *)
(*****************************************************************************)
EXTENDS CbEvalObservational, TLC

==============================================================================
