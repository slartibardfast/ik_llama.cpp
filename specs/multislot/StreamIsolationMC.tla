--------------------------- MODULE StreamIsolationMC ---------------------------
(*****************************************************************************)
(* MC wrapper for StreamIsolation.tla. Provides Symmetry over Streams.       *)
(*****************************************************************************)
EXTENDS StreamIsolation, TLC

Symmetry == Permutations(Streams)

============================================================================
