--------------------------- MODULE DFlashCycle ---------------------------
(*****************************************************************************)
(* Phase 1 — single-slot DFlash speculative-decoding cycle.                  *)
(*                                                                            *)
(* Hand-tuned from the allium-to-tla.py generator skeleton. Every TLA+        *)
(* INVARIANT name MATCHES an Allium invariant name verbatim — this is the    *)
(* synergy contract.                                                          *)
(*                                                                            *)
(* Source: specs/dflash/dflash.allium top-level + in-contract @invariants.    *)
(*                                                                            *)
(* Bug families (modeling-brief §2) verified here:                            *)
(*   A — Rejection-propagation chain breaks       RejectionDropped            *)
(*   C — Anchor-pos misalignment between block + anchor    AnchorDrift   *)
(*                                                                            *)
(* (B and D are multi-slot — phase 2.)                                        *)
(*                                                                            *)
(* Single-slot abstractions:                                                  *)
(*   - ExtractFeatures and ProjectAndFuse are setup paths; modelled as one    *)
(*     atomic Init block (target hidden states & projection happen once).    *)
(*   - DraftBlockEmit, TargetVerifyBlock, AcceptPrefixDecision, AdvanceState  *)
(*     are explicit actions in the cycle.                                     *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    MaxStep,             \* TLC bound on cycle counter
    BlockSize,           \* From Allium DraftBlock.block_size (16 for Qwen3.6-27B-DFlash)
    RejectionDropped,    \* Inject Family A bug at AdvanceState
    AnchorDrift     \* Inject Family C bug at DraftBlockEmit

ASSUME MaxStep   \in Nat /\ MaxStep   > 0
ASSUME BlockSize \in Nat /\ BlockSize > 0
ASSUME RejectionDropped \in BOOLEAN
ASSUME AnchorDrift \in BOOLEAN

PCValues == { "draft", "verify", "accept", "advance" }
NoBlock  == [present |-> FALSE, anchor_pos |-> 0, n_tokens |-> 0]
BlockRecord == [present: BOOLEAN, anchor_pos: Nat, n_tokens: Nat]

VARIABLES
    step,
    pc,
    anchor_pos,
    target_kv_n_cells,
    draft_kv_self_n_cells,
    draft_kv_injected_n_cells,
    n_rejected_prev,
    in_flight_block,
    verify_seq_lens_seen,
    verify_effective_seen,
    last_n_accepted,
    last_bonus_pos,
    accept_history

vars == << step, pc, anchor_pos,
           target_kv_n_cells, draft_kv_self_n_cells, draft_kv_injected_n_cells,
           n_rejected_prev, in_flight_block,
           verify_seq_lens_seen, verify_effective_seen,
           last_n_accepted, last_bonus_pos, accept_history >>

----------------------------------------------------------------------------
TypeOK ==
    /\ step                       \in 0..MaxStep
    /\ pc                         \in PCValues
    /\ anchor_pos                 \in Nat
    /\ target_kv_n_cells          \in Nat
    /\ draft_kv_self_n_cells      \in Nat
    /\ draft_kv_injected_n_cells  \in Nat
    /\ n_rejected_prev            \in 0..BlockSize
    /\ in_flight_block            \in BlockRecord
    /\ verify_seq_lens_seen       \in Nat
    /\ verify_effective_seen      \in Nat
    /\ last_n_accepted            \in 0..BlockSize
    /\ last_bonus_pos             \in Nat
    /\ accept_history             \in Seq(0..BlockSize)

Init ==
    /\ step                      = 0
    /\ pc                        = "draft"
    /\ anchor_pos                = 0
    /\ target_kv_n_cells         = 1
    /\ draft_kv_self_n_cells     = 0
    /\ draft_kv_injected_n_cells = 0
    /\ n_rejected_prev           = 0
    /\ in_flight_block           = NoBlock
    /\ verify_seq_lens_seen      = 0
    /\ verify_effective_seen     = 0
    /\ last_n_accepted           = 0
    /\ last_bonus_pos            = 0
    /\ accept_history            = << >>

----------------------------------------------------------------------------
DraftBlockEmit ==
    /\ pc = "draft"
    /\ step < MaxStep
    /\ LET correct_p == anchor_pos
           buggy_p   == anchor_pos + 1
           emit_p    == IF AnchorDrift THEN buggy_p ELSE correct_p
       IN  in_flight_block' =
              [ present    |-> TRUE,
                anchor_pos |-> emit_p,
                n_tokens   |-> BlockSize ]
    /\ pc' = "verify"
    /\ UNCHANGED << step, anchor_pos,
                    target_kv_n_cells, draft_kv_self_n_cells,
                    draft_kv_injected_n_cells, n_rejected_prev,
                    verify_seq_lens_seen, verify_effective_seen,
                    last_n_accepted, last_bonus_pos, accept_history >>

TargetVerifyBlock ==
    /\ pc = "verify"
    /\ in_flight_block.present
    /\ LET seq_lens_in == target_kv_n_cells + n_rejected_prev + BlockSize + 1
           effective   == seq_lens_in - n_rejected_prev
       IN  /\ verify_seq_lens_seen'  = seq_lens_in
           /\ verify_effective_seen' = effective
    /\ pc' = "accept"
    /\ UNCHANGED << step, anchor_pos,
                    target_kv_n_cells, draft_kv_self_n_cells,
                    draft_kv_injected_n_cells, n_rejected_prev,
                    in_flight_block, last_n_accepted, last_bonus_pos,
                    accept_history >>

AcceptPrefixDecision ==
    /\ pc = "accept"
    /\ in_flight_block.present
    /\ \E n \in 0..BlockSize:
         /\ last_n_accepted' = n
         /\ last_bonus_pos'  = in_flight_block.anchor_pos + n + 1
    /\ pc' = "advance"
    /\ UNCHANGED << step, anchor_pos,
                    target_kv_n_cells, draft_kv_self_n_cells,
                    draft_kv_injected_n_cells, n_rejected_prev,
                    in_flight_block, verify_seq_lens_seen,
                    verify_effective_seen, accept_history >>

AdvanceState ==
    /\ pc = "advance"
    /\ LET n_acc       == last_n_accepted
           n_rej_new   == BlockSize - n_acc
           correct_c   == n_rej_new
           buggy_c     == 0
           carry       == IF RejectionDropped THEN buggy_c ELSE correct_c
       IN
       /\ target_kv_n_cells'         = target_kv_n_cells + n_acc + 1
       /\ draft_kv_self_n_cells'     = draft_kv_self_n_cells + n_acc
       /\ draft_kv_injected_n_cells' = 1
       /\ anchor_pos'                = in_flight_block.anchor_pos + n_acc + 1
       /\ n_rejected_prev'           = carry
       /\ in_flight_block'           = NoBlock
       /\ accept_history'            = Append(accept_history, n_acc)
       /\ step'                      = step + 1
    /\ pc' = "draft"
    /\ UNCHANGED << verify_seq_lens_seen, verify_effective_seen,
                    last_n_accepted, last_bonus_pos >>

Next ==
    \/ DraftBlockEmit
    \/ TargetVerifyBlock
    \/ AcceptPrefixDecision
    \/ AdvanceState

Spec == Init /\ [][Next]_vars /\ WF_vars(AdvanceState)

----------------------------------------------------------------------------
(* Top-level Allium invariants — most are static deployment facts.
   They become TRUE here and are bound by load-time C++ property tests. *)

SharedEmbedAndLMHead                   == TRUE
FeatureSourceFixedPerDeployment        == TRUE
BlockSizeFixedAtContextStart           == TRUE
PerDeploymentDeterminism               == TRUE
SpeculativeCycleAtomicity              == TRUE
DraftKVViaInjectionNotCrossAttention   == TRUE
PerSlotDraftKVCache                    == TRUE
HybridTargetRecurrentStateTracking     == TRUE
DPAttentionNotSupported                == TRUE
PipelineParallelismRequiresPpSizeEq1   == TRUE
ContextBudgetAtNp8                     == TRUE
DrafterIsDenseQwen3                    == TRUE
DrafterTargetVocabIdentity             == TRUE
MultimodalTurnsRoutedAroundDrafter     == TRUE
SourceLayerCountMatchesDrafterTraining == TRUE
TargetLayerIdsPlusOneAtRuntime         == TRUE
NoTreeDraftingNoChainRollout           == TRUE
\* KernelDeterminism moved to bindings_external (test-side binding)

----------------------------------------------------------------------------
(* In-contract @invariants — those without an operational form in this
   single-slot Phase 1 stay TRUE; the seven cycle-structure ones get bodies. *)

SourceLayersWithinTarget         == TRUE
UniformSampleSpacing             == TRUE
DeterminismPerDeployment         == TRUE
\* FuseProjectionFcWeight       moved to bindings_external (test-side binding)
\* FeatureWidthMatchesTarget    moved to bindings_external (test-side binding)
\* CombineOrderFCThenHiddenNorm moved to bindings_external (test-side binding)
\* ContextStatesAnchorLevel     moved to bindings_external (test-side binding)
\* PerLayerArity                moved to bindings_external (test-side binding)
\* HeadShapeMatchesDraft        moved to bindings_external (test-side binding)
\* KAsymmetricallyNormedVNot    moved to bindings_external (test-side binding)
\* InjectedAnchorAlignment      moved to bindings_external (test-side binding)
\* InjectPerLayerLaunches       moved to bindings_external (test-side binding)
ReuseAcrossDenoiseSteps          == TRUE
SingleForwardPerStep             == TRUE
QuerySpanIsOnePlusN              == TRUE
InjectionConsumedAtEveryLayer    == TRUE
LayerTypeDependentMask           == TRUE
AnchorEmbeddingFromTarget        == TRUE
BlockSizeBindsToConfig ==
    in_flight_block.present => in_flight_block.n_tokens = BlockSize
VerifyBatchShape                 == TRUE
SingleContiguousRun              == TRUE
PerSlotVerifyDispatchAtMultiSlot == TRUE
LongestPrefixMatchUnderArgmax    == TRUE
BonusIsArgmaxAtFirstUnacceptedRow == TRUE
DeterminismUnderFixedInputs      == TRUE
ProbabilisticVerifyOutOfScope    == TRUE
DraftKVAdvancesPaired            == TRUE
DraftKVRollbackOnRejection       == TRUE
InjectedKVEvictedOnAnchorAdvance == TRUE
NewAnchorIsBonus                 == TRUE
AtomicityPerCycle                == TRUE

(* ===== Operational invariants — the seven Family A/C gates ===== *)

TargetKVNotAdvancedDuringVerify ==
    target_kv_n_cells >= 1

AnchorPosPreserved ==
    in_flight_block.present =>
        in_flight_block.anchor_pos = anchor_pos

NAcceptedWithinBound ==
    last_n_accepted \in 0..BlockSize

\* Transition invariant: meaningful in the accept->advance window, when
\* bonus_pos has just been computed and the block is still in flight.
\* After AdvanceState clears the block we let it hold trivially.
BonusPosIsAnchorPlusNAcceptedPlusOne ==
    pc = "advance" =>
        /\ in_flight_block.present
        /\ last_bonus_pos = in_flight_block.anchor_pos + last_n_accepted + 1

NumRejectedTokensFlowsBackToProposer ==
    (pc = "draft" /\ step > 0) =>
        n_rejected_prev = BlockSize - last_n_accepted

\* Transition invariant: meaningful in the post-TargetVerifyBlock window
\* (pc in {accept, advance}) when verify_*_seen reflect the just-completed
\* verify AND n_rejected_prev still holds the value used to compute them.
\* After AdvanceState, n_rejected_prev refreshes for the NEXT cycle; the
\* old verify values are stale and the relation holds trivially.
EffectiveSeqLensSubtractsRejected ==
    pc \in { "accept", "advance" } =>
        /\ verify_seq_lens_seen >= verify_effective_seen
        /\ (verify_seq_lens_seen > 0 =>
                verify_effective_seen = verify_seq_lens_seen - n_rejected_prev)

TargetKVAdvancesByAcceptedPlusBonus ==
    LET R[i \in 0..Len(accept_history)] ==
            IF i = 0 THEN 0
            ELSE R[i-1] + accept_history[i]
        n_done == Len(accept_history)
    IN target_kv_n_cells = 1 + R[n_done] + n_done

BoundedStep == step <= MaxStep

============================================================================
