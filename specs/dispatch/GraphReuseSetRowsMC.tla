--------------------------- MODULE GraphReuseSetRowsMC ---------------------------
(*****************************************************************************)
(* MC wrapper for GraphReuseSetRows.tla. Provides Symmetry over Shapes.   *)
(*****************************************************************************)
EXTENDS GraphReuseSetRows, TLC

Symmetry == Permutations(Shapes)

============================================================================
