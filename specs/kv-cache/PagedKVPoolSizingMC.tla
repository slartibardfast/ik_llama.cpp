--------------------------- MODULE PagedKVPoolSizingMC ---------------------------
(*****************************************************************************)
(* T5.9 — sibling MC module for PagedKVAllocator.tla that under-sizes the    *)
(* pool to force OOB and bind PoolBoundsRespected + PoolExhaustionRecorded.  *)
(*                                                                            *)
(* Bounded constants in PagedKVPoolSizingMC.cfg:                             *)
(*   NBlocks = 2, BlockSizeTokens = 2, MaxWritesPerSeq = 6, two seqs.        *)
(* Together this guarantees the OOB path in WriteTokens is REACHED in       *)
(* state-space exploration; PagedKVAllocatorMC.cfg sizes things to fit     *)
(* exactly so OOB is barely reachable.                                       *)
(*****************************************************************************)
EXTENDS PagedKVAllocator

\* Symmetry over the seq set — TLC can collapse symmetric subgraphs.
SeqSymmetry == Permutations(Seqs)

=============================================================================
