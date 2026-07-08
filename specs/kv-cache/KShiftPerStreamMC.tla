--------------------------- MODULE KShiftPerStreamMC ---------------------------
(*****************************************************************************)
(* MC wrapper for KShiftPerStream.tla. Provides Symmetry over Streams.      *)
(*****************************************************************************)
EXTENDS KShiftPerStream, TLC

Symmetry == Permutations(Streams)

============================================================================
