---------------------------- MODULE CUDAGraphCacheConsistencyMC ----------------------------
(*****************************************************************************)
(* Model-checking shim for CUDAGraphCacheConsistency.tla. Capacity = 2 with  *)
(* three distinct keys forces the FIFO eviction branch; MaxOps = 4 bounds    *)
(* the state space while still reaching a full-then-evict-then-clear cycle.  *)
(*****************************************************************************)

EXTENDS CUDAGraphCacheConsistency

Capacity_def == 2
Keys_def     == {"k1", "k2", "k3"}
MaxOps_def   == 4

============================================================================
