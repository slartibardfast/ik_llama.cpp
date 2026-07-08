---------------------------- MODULE CalibrationFrameworkMC ----------------------------
(*****************************************************************************)
(* Model-checking shim for CalibrationFramework.tla.                         *)
(*                                                                           *)
(* Buckets are ordinal stand-ins for the five real payload sizes plus the    *)
(* SIZE_MAX sentinel (= 5, defined in the parent module):                    *)
(*   0 -> "no payload"   1 -> 1MB   2 -> 10MB   3 -> 100MB   4 -> 1GB         *)
(*   5 -> SIZE_MAX ("never use alt")                                         *)
(*                                                                           *)
(* Two ops, no env overrides, cache enabled. on_disk_cache is left free in   *)
(* Init (NotExists or any full table) so both the CacheHit and CacheMiss     *)
(* arcs are explored.                                                        *)
(*****************************************************************************)

EXTENDS CalibrationFramework

Ops_def          == {"reduce", "matmul"}
Buckets_def      == {0, 1, 2, 3, 4, 5}
HardwareKey_def  == "xeon_rtx6000_v0"
EnvOverrides_def == {}
DisableCache_def == FALSE

============================================================================
