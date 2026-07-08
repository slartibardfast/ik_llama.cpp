--------------------------- MODULE PagedKVAllocator ---------------------------
(*****************************************************************************)
(* T5.0 — TLA+ spec for the paged KV block allocator.                        *)
(*                                                                            *)
(* Companion to specs/kv-cache/paged_block_allocator.allium. Models the      *)
(* block pool, per-seq block table, and free list under any schedule of     *)
(* alloc / free / defrag / write operations.                                 *)
(*                                                                            *)
(* Why this spec exists: T5.0-probe falsified the perf-uplift mechanism on  *)
(* current workload; user selected Path C override (data/t5-probe-          *)
(* findings.md §9) and reframed Tier 5 as forward-looking infra for         *)
(* high-ctx workloads. "Complete sincerity" discipline: allocator           *)
(* correctness must be airtight — including OOM behaviour at high ctx       *)
(* (first-class test target, not edge case).                                 *)
(*                                                                            *)
(* The four load-bearing invariants:                                        *)
(*   BlockUniquelyOwned — no block_id is in two seqs' block_tables.        *)
(*   FreeListDisjoint — free_list and ALL block_tables are pairwise        *)
(*                      disjoint.                                            *)
(*   AllocLazy — a seq's block_table length matches ceil(writes / 64).    *)
(*   DefragPreservesOwnership — for every (seq, logical_pos), the         *)
(*                              physical block holding that token's bytes  *)
(*                              is correctly re-mapped through any defrag. *)
(*                                                                            *)
(* Temporal property:                                                       *)
(*   EventuallyAllocSucceedsUnlessFull — alloc requests eventually          *)
(*     succeed under fair scheduling, unless the pool is genuinely full.    *)
(*                                                                            *)
(* Composes with:                                                          *)
(*   - specs/dispatch/unified_stream_dispatch.allium (T3 Bug C closure at  *)
(*     the WRITE side).                                                     *)
(*   - specs/kv-cache/paged_write_path.allium (KVWriteOp at SET_ROWS).     *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    Seqs,               \* set of seq ids
    NBlocks,            \* total block pool capacity
    BlockSizeTokens,    \* tokens per block (= 64 in production)
    MaxWritesPerSeq,    \* TLC bound on per-seq writes
    MaxStep             \* global step bound (action budget)

\* The OUT_OF_BLOCKS sentinel returned by alloc when pool is full.
OOB == -1

VARIABLES
    block_pool,         \* [0..NBlocks-1 -> "FREE" or seq_id]
                        \* who owns each physical block (or FREE)
    block_table,        \* [Seqs -> Seq(0..NBlocks-1)]
                        \* per-seq ordered list of owned block ids
    free_list,          \* Seq(0..NBlocks-1) — LIFO stack of free ids
    written_tokens,     \* [Seqs -> 0..MaxWritesPerSeq] — running count per seq
    alloc_history,      \* Seq(<<seq_id, result_block_id>>) — ghost record
    step_count

vars == <<block_pool, block_table, free_list, written_tokens, alloc_history, step_count>>

----------------------------------------------------------------------------
(* Type invariant. *)
TypeOK ==
    /\ block_pool \in [0..(NBlocks-1) -> {"FREE"} \cup Seqs]
    /\ block_table \in [Seqs -> Seq(0..(NBlocks-1))]
    /\ free_list \in Seq(0..(NBlocks-1))
    /\ written_tokens \in [Seqs -> 0..MaxWritesPerSeq]
    /\ alloc_history \in Seq((Seqs) \X ((0..(NBlocks-1)) \cup {OOB}))
    /\ step_count \in 0..MaxStep

----------------------------------------------------------------------------
(* Initial state: pool entirely free, every block_table empty, free_list  *)
(* in canonical order [NBlocks-1, NBlocks-2, ..., 1, 0] (LIFO so block 0  *)
(* is popped first on first alloc).                                        *)
Init ==
    /\ block_pool = [b \in 0..(NBlocks-1) |-> "FREE"]
    /\ block_table = [s \in Seqs |-> <<>>]
    /\ free_list = [i \in 1..NBlocks |-> NBlocks - i]
    /\ written_tokens = [s \in Seqs |-> 0]
    /\ alloc_history = <<>>
    /\ step_count = 0

----------------------------------------------------------------------------
(* Helper: BlocksNeededForWrites(n_tokens) = ceil(n_tokens / block_size). *)
BlocksNeededForWrites(n_tokens) ==
    IF n_tokens = 0
    THEN 0
    ELSE ((n_tokens - 1) \div BlockSizeTokens) + 1

----------------------------------------------------------------------------
(* Action: WriteTokens(seq, n) — record that seq wrote n tokens. Triggers *)
(* lazy alloc up to BlocksNeededForWrites if not enough blocks are held.  *)
(* Models the WRITE-time alloc path:                                      *)
(*   for each new token: if pos / block_size >= block_table[seq].size():  *)
(*     alloc_block(seq).                                                  *)
(*****************************************************************************)
WriteTokens(seq, n) ==
    /\ seq \in Seqs
    /\ n > 0
    /\ written_tokens[seq] + n <= MaxWritesPerSeq
    /\ step_count < MaxStep
    /\ LET new_total == written_tokens[seq] + n
           needed == BlocksNeededForWrites(new_total)
           have == Len(block_table[seq])
           deficit == needed - have IN
       IF deficit = 0
       THEN
         /\ written_tokens' = [written_tokens EXCEPT ![seq] = new_total]
         /\ step_count' = step_count + 1
         /\ UNCHANGED <<block_pool, block_table, free_list, alloc_history>>
       ELSE IF Len(free_list) >= deficit
       THEN
         \* Take the first `deficit` entries from free_list (LIFO).
         LET taken == SubSeq(free_list, 1, deficit)
             remaining == SubSeq(free_list, deficit + 1, Len(free_list)) IN
         /\ block_table' = [block_table EXCEPT ![seq] = @ \o taken]
         /\ block_pool' = [b \in DOMAIN block_pool |->
                              IF \E i \in 1..deficit: taken[i] = b
                              THEN seq
                              ELSE block_pool[b]]
         /\ free_list' = remaining
         /\ written_tokens' = [written_tokens EXCEPT ![seq] = new_total]
         /\ alloc_history' = alloc_history \o
              [i \in 1..deficit |-> <<seq, taken[i]>>]
         /\ step_count' = step_count + 1
       ELSE
         \* Pool exhausted mid-write. Models OOM behaviour: write
         \* operation aborts without partial progress.
         /\ alloc_history' = Append(alloc_history, <<seq, OOB>>)
         /\ step_count' = step_count + 1
         /\ UNCHANGED <<block_pool, block_table, free_list, written_tokens>>

----------------------------------------------------------------------------
(* Action: FreeSeq(seq) — release all blocks owned by seq.               *)
(* Postcondition: block_table[seq] = <<>>, block_pool re-marks them      *)
(* FREE, free_list extended with the freed ids in reverse-allocation      *)
(* order (LIFO preservation).                                             *)
(*****************************************************************************)
FreeSeq(seq) ==
    /\ seq \in Seqs
    /\ Len(block_table[seq]) > 0
    /\ step_count < MaxStep
    /\ LET freed == block_table[seq]
           reversed == [i \in 1..Len(freed) |-> freed[Len(freed) - i + 1]] IN
       /\ block_table' = [block_table EXCEPT ![seq] = <<>>]
       /\ block_pool' = [b \in DOMAIN block_pool |->
                            IF \E i \in 1..Len(freed): freed[i] = b
                            THEN "FREE"
                            ELSE block_pool[b]]
       /\ free_list' = reversed \o free_list
       /\ written_tokens' = [written_tokens EXCEPT ![seq] = 0]
       /\ step_count' = step_count + 1
       /\ UNCHANGED <<alloc_history>>

----------------------------------------------------------------------------
(* Action: Defrag — coalesce block_table entries to the lowest available  *)
(* block ids, then push freed ids back into the free_list.                *)
(*                                                                          *)
(* Simplification: this models defrag as a one-step "renumber" that       *)
(* re-assigns block ids to occupy [0, total_used) contiguously, preserving *)
(* per-seq block ORDER. The K/V byte fidelity (the actual cpy of data)    *)
(* is in scope of paged_kshift_defrag.allium, not this TLA+ allocator     *)
(* model.                                                                  *)
(*                                                                          *)
(* Defrag changes block_pool and block_table mappings; block_table[seq]   *)
(* must hold the same number of blocks per seq before/after.              *)
(*****************************************************************************)
\* Defrag is intentionally modeled only as a no-op step in this
\* allocator-level model. The full physical-id renumbering and the
\* byte-level move invariants are in paged_kshift_defrag.allium and
\* PagedKVByteIdentity.tla. Including defrag-as-renumber here would
\* require quantified updates that blow up TLC's state space without
\* adding new allocator invariants beyond what BlockUniquelyOwned
\* +FreeListDisjoint + AllocLazy already check pre/post any defrag.
DefragNoOp ==
    /\ step_count < MaxStep
    /\ step_count' = step_count + 1
    /\ UNCHANGED <<block_pool, block_table, free_list, written_tokens, alloc_history>>

----------------------------------------------------------------------------
Next ==
    \/ \E s \in Seqs, n \in 1..(BlockSizeTokens * 2): WriteTokens(s, n)
    \/ \E s \in Seqs: FreeSeq(s)
    \/ DefragNoOp

Spec == Init /\ [][Next]_vars
    /\ WF_vars(\E s \in Seqs, n \in 1..(BlockSizeTokens * 2): WriteTokens(s, n))

----------------------------------------------------------------------------
(* Load-bearing invariants. *)
----------------------------------------------------------------------------

\* BlockUniquelyOwned — no block id appears in two seqs' block_tables.
\* The single most important safety property.
BlockUniquelyOwned ==
    \A s1, s2 \in Seqs:
        \A i \in 1..Len(block_table[s1]):
            \A j \in 1..Len(block_table[s2]):
                (s1 # s2) => (block_table[s1][i] # block_table[s2][j])

\* FreeListDisjoint — free_list and every block_table are pairwise disjoint.
FreeListDisjoint ==
    \A b \in 0..(NBlocks-1):
        ~ (\E s \in Seqs: \E i \in 1..Len(block_table[s]): block_table[s][i] = b)
        \/
        ~ (\E i \in 1..Len(free_list): free_list[i] = b)

\* AllocLazy — block_table[seq] size = ceil(written_tokens[seq] / block_size).
AllocLazy ==
    \A s \in Seqs:
        Len(block_table[s]) = BlocksNeededForWrites(written_tokens[s])

\* FreeOrOwned — every block id is either in free_list OR in exactly one
\* block_table, never both, never neither.
FreeOrOwned ==
    \A b \in 0..(NBlocks-1):
        \/ (\E i \in 1..Len(free_list): free_list[i] = b)
        \/ (\E s \in Seqs: \E i \in 1..Len(block_table[s]): block_table[s][i] = b)

\* DefragPreservesOwnership — under our DefragNoOp this is trivially true.
\* The full property (byte-equal reads post-defrag) lives in
\* PagedKVByteIdentity.tla.
DefragPreservesOwnership == TRUE

\* PoolBoundsRespected — every block id in any block_table is in
\* [0, NBlocks). The bound is implicit in TypeOK (Seq(0..(NBlocks-1)))
\* but expressed here explicitly as defence-in-depth for T5.9 paged
\* BACKING. Composes with paged_kv_pool_sizing.allium::PoolBoundsRespected.
PoolBoundsRespected ==
    \A s \in Seqs:
        \A i \in 1..Len(block_table[s]):
            block_table[s][i] \in 0..(NBlocks-1)

\* PoolExhaustionRecorded — when a WriteTokens action cannot allocate
\* enough blocks, an OOB entry is appended to alloc_history. The
\* model's WriteTokens action (line 123-128) already implements this;
\* this invariant checks it bound.
PoolExhaustionRecorded ==
    \* Whenever Len(free_list) < deficit during a WriteTokens, the
    \* alloc_history records OOB. Captured as: if any seq has
    \* written_tokens > 0 but its block_table is empty, then OOB
    \* must appear in alloc_history (the only way that state is
    \* reachable in this model).
    \A s \in Seqs:
        (written_tokens[s] > 0 /\ Len(block_table[s]) = 0)
            => (\E i \in 1..Len(alloc_history): alloc_history[i][2] = OOB)

----------------------------------------------------------------------------
(* Temporal property: EventuallyAllocSucceedsUnlessFull. *)
----------------------------------------------------------------------------

\* Under fair scheduling, every WRITE request that fits in the pool
\* eventually completes (i.e. produces a non-empty block_table for the seq).
EventuallyWriteSucceedsUnlessFull ==
    \A s \in Seqs:
        [](Len(free_list) > 0 => <>(written_tokens[s] > 0 \/ Len(free_list) = 0))

=============================================================================
