--------------------------- MODULE CUDAGraphReuseMC ---------------------------
(*****************************************************************************)
(* MC wrapper for CUDAGraphReuse.tla. Provides Symmetry over Topologies      *)
(* and Dtypes for tractable BFS.                                              *)
(*****************************************************************************)
EXTENDS CUDAGraphReuse, TLC

Symmetry == Permutations(Topologies) \cup Permutations(Dtypes)

============================================================================
