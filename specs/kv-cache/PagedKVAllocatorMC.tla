--------------------------- MODULE PagedKVAllocatorMC ---------------------------
(*****************************************************************************)
(* T5.0 — TLC model-check config for PagedKVAllocator.tla.                   *)
(*                                                                            *)
(* Bounded constants chosen to:                                              *)
(*   - Surface fragmentation cases (3 seqs alloc/free interleaved).        *)
(*   - Cover the OOM signal at NBlocks = 8 with up to 24 writes per seq    *)
(*     forcing pool exhaustion.                                              *)
(*   - Verify AllocLazy under writes that span 1, 2, 3 blocks.             *)
(*   - Exercise the LIFO free-list ordering (DeterministicAtFixedSequence  *)
(*     refinement at the action level).                                     *)
(*****************************************************************************)
EXTENDS PagedKVAllocator

\* Symmetry over the seq set — TLC can collapse symmetric subgraphs.
SeqSymmetry == Permutations(Seqs)

=============================================================================
