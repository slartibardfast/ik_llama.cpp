---------------------------- MODULE CalibratedOpEquivalenceMC ----------------------------
(*****************************************************************************)
(* Model-checking shim for CalibratedOpEquivalence.tla.                      *)
(*                                                                           *)
(* Buckets / PayloadSamples are ordinal stand-ins for the five real payload  *)
(* sizes plus the SIZE_MAX sentinel (= 5, defined in the parent module):     *)
(*   0 -> "no payload"  1 -> 1MB  2 -> 10MB  3 -> 100MB  4 -> 1GB  5 -> MAX   *)
(* Hashes is a two-element set so the equivalence invariant is non-trivially *)
(* exercised (default and alt slots can each take 0 or 1, and the contract   *)
(* requires they agree).                                                     *)
(*****************************************************************************)

EXTENDS CalibratedOpEquivalence

Ops_def            == {"reduce", "matmul"}
Buckets_def        == {0, 1, 2, 3, 4, 5}
PayloadSamples_def == {0, 1, 2, 3, 4, 5}
Hashes_def         == {0, 1}

============================================================================
