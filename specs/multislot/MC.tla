--------------------------- MODULE MC ---------------------------
(*****************************************************************************)
(* MC wrapper for Fix.tla. Provides Symmetry definition for SYMMETRY        *)
(* reduction over the symmetric Slots set.                                   *)
(*****************************************************************************)
EXTENDS Fix, TLC

\* Permutations over Slots; SYMMETRY in cfg refers to this.
Symmetry == Permutations(Slots)

============================================================================
