--------------------------- MODULE BatchCompositionMC ---------------------------
(*****************************************************************************)
(* MC wrapper for BatchComposition.tla. Provides Symmetry over Slots.        *)
(*****************************************************************************)
EXTENDS BatchComposition, TLC

Symmetry == Permutations(Slots)

============================================================================
