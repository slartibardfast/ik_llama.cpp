--------------------------- MODULE PagedKVByteIdentityMC ---------------------------
(*****************************************************************************)
(* T5.0 — TLC model-check config for PagedKVByteIdentity.tla.                *)
(*                                                                            *)
(* Bounded constants chosen to:                                              *)
(*   - Exercise both modes ("Contiguous", "Paged").                         *)
(*   - Cover multiple blocks per seq (BlocksPerStreamSlab >= 2).            *)
(*   - Verify the invariant holds at every (seq, k) reachable by the action *)
(*     graph.                                                                *)
(*                                                                            *)
(* The invariant is mostly static (depends on block_table = TrivialBlockTable*)
(* which is the Init value and never changes in this model), so TLC catches *)
(* a violation only if the address formulae diverge. That's exactly the    *)
(* property under test.                                                     *)
(*****************************************************************************)
EXTENDS PagedKVByteIdentity

=============================================================================
