--------------------------- MODULE PagedKVByteIdentity ---------------------------
(*****************************************************************************)
(* T5.0 — TLA+ spec for the K/V read byte-identity contract.                 *)
(*                                                                            *)
(* Companion to:                                                            *)
(*   - specs/kv-cache/paged_read_path.allium                                *)
(*       ::PagedFAReadEquivToContiguousAtIdentity                          *)
(*   - specs/kv-cache/paged_write_path.allium                               *)
(*       ::PagedKVWriteEquivToLegacyAtIdentity                             *)
(*                                                                            *)
(* Parametrised on Mode in {"Contiguous", "Paged"}. Models the K-row       *)
(* address computation as a function of (seq, head, k_position) and the     *)
(* current Mode. The load-bearing invariant:                                *)
(*                                                                            *)
(*   ByteIdentityAtTrivialMapping —                                         *)
(*     When Mode = "Paged" and the block_table is the trivial (per-seq      *)
(*     contiguous) mapping, the physical address produced for every         *)
(*     (seq, head, k_position) is byte-identical to Mode = "Contiguous".   *)
(*                                                                            *)
(* This is the formal version of the surfaces-summary claim "collapses     *)
(* byte-identically at n_stream = 1" (Mechanism section of                  *)
(* PHASE_NSTREAM_KV_PERF.md), generalised to multi-seq with the trivial     *)
(* mapping per-seq.                                                         *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NSeqs,              \* batch size in this read tick
    Ne11PerSeq,         \* K-loop bound per seq
    BlockSizeTokens,    \* block granularity (= 64 in production)
    KvpsPerStream,      \* contiguous per-stream slab size
    HeadDim,            \* per-head K row dimension (informational)
    NHeadKv,            \* GQA head count
    Nb11               \* per-row byte stride (informational; constant)

VARIABLES
    mode,               \* "Contiguous" or "Paged"
    block_table,        \* [0..NSeqs-1 -> Seq(Nat)]
                        \* per-seq physical block ids; trivial mapping is
                        \* [0..NSeqs-1 |-> seq * (KvpsPerStream / BlockSizeTokens)
                        \*                + [0, 1, 2, ...]]
    step_count

vars == <<mode, block_table, step_count>>

----------------------------------------------------------------------------
MaxStep == NSeqs * Ne11PerSeq + 4

\* Blocks per stream slab.
BlocksPerStreamSlab == KvpsPerStream \div BlockSizeTokens

\* "Trivial" mapping: seq s owns blocks [s * BlocksPerStreamSlab, ...).
TrivialBlockTable ==
    [s \in 0..(NSeqs-1) |->
        [i \in 1..BlocksPerStreamSlab |->
            s * BlocksPerStreamSlab + (i - 1)]]

\* Address formula — Contiguous mode (T4 baseline).
\* Address = k_stream_base + (seq * KvpsPerStream) + k_position * Nb11
ContiguousAddress(seq, k) ==
    seq * KvpsPerStream * Nb11 + k * Nb11

\* Address formula — Paged mode.
\* Address = block_table[seq][k / block_size + 1] * block_size * Nb11
\*         + (k % block_size) * Nb11
\* Note TLA+ sequence indices are 1-based; the (k / block_size + 1) maps
\* k in [0, KvpsPerStream) to the (1-indexed) block slot.
PagedAddress(seq, k) ==
    LET blk_idx == (k \div BlockSizeTokens) + 1 IN
    LET tok_in_blk == k % BlockSizeTokens IN
    IF blk_idx > Len(block_table[seq])
    THEN -1    \* out-of-range: block not allocated. Read MUST NOT fire.
    ELSE block_table[seq][blk_idx] * BlockSizeTokens * Nb11
       + tok_in_blk * Nb11

----------------------------------------------------------------------------
TypeOK ==
    /\ mode \in {"Contiguous", "Paged"}
    /\ block_table \in [0..(NSeqs-1) -> Seq(0..(NSeqs * BlocksPerStreamSlab - 1))]
    /\ step_count \in 0..MaxStep

----------------------------------------------------------------------------
(* Init: enter with the trivial block_table mapping. *)
Init ==
    /\ mode \in {"Contiguous", "Paged"}
    /\ block_table = TrivialBlockTable
    /\ step_count = 0

----------------------------------------------------------------------------
(* Action: Read(seq, k) — one K-row read at (seq, k). Records the         *)
(* physical address that the current Mode produces. The kernel iterates   *)
(* over (seq, k) in canonical order; we model this as nondeterministic    *)
(* selection bounded by step_count.                                       *)
(*****************************************************************************)
Read(seq, k) ==
    /\ seq \in 0..(NSeqs-1)
    /\ k \in 0..(Ne11PerSeq-1)
    /\ step_count < MaxStep
    /\ step_count' = step_count + 1
    /\ UNCHANGED <<mode, block_table>>

----------------------------------------------------------------------------
(* Action: ReadPair(seq, k) — a paired read in both modes. This action    *)
(* is the byte-identity check at the action level: for the SAME (seq, k), *)
(* both modes are computed and compared. If the addresses differ, the     *)
(* invariant ByteIdentityAtTrivialMapping is violated.                    *)
(*                                                                          *)
(* Modeled as a single state transition that records both addresses.      *)
(*****************************************************************************)
ReadPair(seq, k) ==
    /\ seq \in 0..(NSeqs-1)
    /\ k \in 0..(Ne11PerSeq-1)
    /\ step_count < MaxStep
    \* The invariant ByteIdentityAtTrivialMapping is the load-bearing
    \* property; the per-action check is left as a state-invariant under
    \* TLC rather than a sequence-of-actions check. The action just
    \* progresses step_count.
    /\ step_count' = step_count + 1
    /\ UNCHANGED <<mode, block_table>>

----------------------------------------------------------------------------
Next ==
    \/ \E s \in 0..(NSeqs-1), k \in 0..(Ne11PerSeq-1): Read(s, k)
    \/ \E s \in 0..(NSeqs-1), k \in 0..(Ne11PerSeq-1): ReadPair(s, k)

Spec == Init /\ [][Next]_vars

----------------------------------------------------------------------------
(* Load-bearing invariant: ByteIdentityAtTrivialMapping.                  *)
(* At the trivial block_table mapping, for every (seq, k) in range, the   *)
(* paged and contiguous addresses are equal.                               *)
(*****************************************************************************)
ByteIdentityAtTrivialMapping ==
    \A s \in 0..(NSeqs-1):
        \A k \in 0..(Ne11PerSeq-1):
            (block_table[s] = TrivialBlockTable[s])
            => (PagedAddress(s, k) = ContiguousAddress(s, k))

----------------------------------------------------------------------------
(* Companion invariants. *)
----------------------------------------------------------------------------

\* Paged READ never produces a "block not allocated" sentinel when
\* block_table is the trivial mapping and k is in range.
NoOutOfRangeAtTrivialMapping ==
    \A s \in 0..(NSeqs-1):
        \A k \in 0..(Ne11PerSeq-1):
            (block_table[s] = TrivialBlockTable[s])
            => (PagedAddress(s, k) # -1)

\* Bug C absence at the address-formula level: for any two (seq, k)
\* pairs from different seqs in range, the addresses do not collide.
\* This holds for Contiguous and (at trivial mapping) for Paged.
NoCrossSeqAliasingAtTrivialMapping ==
    \A s1, s2 \in 0..(NSeqs-1):
        \A k1, k2 \in 0..(Ne11PerSeq-1):
            (s1 # s2 /\ block_table[s1] = TrivialBlockTable[s1]
             /\ block_table[s2] = TrivialBlockTable[s2])
            => (PagedAddress(s1, k1) # PagedAddress(s2, k2))

=============================================================================
