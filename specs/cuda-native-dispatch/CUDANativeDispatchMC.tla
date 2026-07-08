---------------------------- MODULE CUDANativeDispatchMC ----------------------------
(*****************************************************************************)
(* Model-checking shim for CUDANativeDispatch.tla. TLC's config-file parser  *)
(* cannot handle structured constants inline; this module defines them as    *)
(* operators that CUDANativeDispatchMC.cfg references via the `<-` arrow.    *)
(*                                                                           *)
(* The split shape exercises:                                                *)
(*   - split 0: a CPU-style backend, n_inputs = 0, no reduce                 *)
(*   - split 1: a CUDA backend, n_inputs = 1, REDUCE (sticky-needs_sync      *)
(*     trigger; binds ReduceMarksAllParticipantsSticky on a non-terminal     *)
(*     reduce split)                                                         *)
(*   - split 2: a CUDA backend, n_inputs = 1, no reduce (downstream          *)
(*     consumer)                                                             *)
(*****************************************************************************)

EXTENDS CUDANativeDispatch

NBackends_def == 2
MaxIters_def  == 2
Splits_def    == <<
    [ backend_id |-> 0, n_inputs |-> 0, has_reduce |-> FALSE ],
    [ backend_id |-> 1, n_inputs |-> 1, has_reduce |-> TRUE  ],
    [ backend_id |-> 0, n_inputs |-> 1, has_reduce |-> FALSE ]
>>

============================================================================
