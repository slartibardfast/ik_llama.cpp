--------------------------- MODULE DefragPerStreamMC ---------------------------
(*****************************************************************************)
(* MC wrapper for DefragPerStream.tla. Provides Symmetry over Streams.     *)
(*****************************************************************************)
EXTENDS DefragPerStream, TLC

Symmetry == Permutations(Streams)

============================================================================
