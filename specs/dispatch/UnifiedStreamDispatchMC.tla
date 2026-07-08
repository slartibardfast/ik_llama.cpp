--------------------------- MODULE UnifiedStreamDispatchMC ---------------------------
(*****************************************************************************)
(* MC wrapper for UnifiedStreamDispatch.tla. Provides Symmetry over Streams *)
(* for tractable BFS.                                                         *)
(*****************************************************************************)
EXTENDS UnifiedStreamDispatch, TLC

Symmetry == Permutations(Streams)

============================================================================
