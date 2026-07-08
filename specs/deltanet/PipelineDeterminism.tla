--------------------------- MODULE PipelineDeterminism ---------------------------
(*****************************************************************************)
(* Phase 1 — vanilla decode pipeline batch-invariance (or lack thereof)      *)
(* for the hybrid Qwen 3.5/3.6 target on dual TU102 sm_75.                   *)
(*                                                                            *)
(* This spec models the CURRENT state of the production pipeline. Per the    *)
(* `batch-invariance.allium` companion in this directory, byte-identity      *)
(* between batch widths (NP=2 vs NP=4 vs NP=8) is NOT held — the FA and     *)
(* MMQ kernels both have data-dependent batch-shape sensitivity that breaks  *)
(* invariance at specific layers depending on input data.                    *)
(*                                                                            *)
(* The model captures:                                                        *)
(*   - The pipeline as a sequence of LayerCompute actions, layer by layer.   *)
(*   - For each layer, the per-slot input residual + kernel template +      *)
(*     batch width determines the per-slot output residual.                  *)
(*   - Two parallel runs at different NP values (run0 at NP_a, run1 at      *)
(*     NP_b, with the same prompt p in the same slot K).                     *)
(*   - The invariant that "if both runs have identical inputs at slot K     *)
(*     entering layer L, AND the kernel template at L is identical, AND    *)
(*     the kernel is batch-invariant, then both runs produce identical      *)
(*     slot-K output at layer L."                                            *)
(*   - The empirical fact that FA at SOME inputs IS batch-shape sensitive   *)
(*     (FA_SHAPE_SENSITIVE config) — modelling a non-deterministic action   *)
(*     where the same kernel can drift even with identical inputs.          *)
(*   - The empirical fact that MMQ amplifies upstream noise differently at  *)
(*     different batch widths (MMQ_AMPLIFIES_NOISE config).                  *)
(*   - The cascade property: once any layer L diverges at slot K, all       *)
(*     subsequent layers' slot-K residuals also diverge.                    *)
(*                                                                            *)
(* TLA+ INVARIANT names match Allium @invariant names verbatim where         *)
(* applicable (synergy contract).                                            *)
(*                                                                            *)
(* Bug families verified here:                                                *)
(*   E — FA_ShapeSensitiveBreaksIdentity                                      *)
(*   F — MMQ_AmplifiesUpstreamNoise                                           *)
(*   G — DriftCascadeAfterFirstDivergence                                     *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    NLayers,          \* number of transformer layers (64 for Qwen 3.6 27B)
    NPa,              \* batch width of run 0 (e.g., 2)
    NPb,              \* batch width of run 1 (e.g., 4); NPa /= NPb
    FA_LayerIndices,  \* set of layer indices that are full_attention (FA path)
    FA_ShapeSensitive,    \* layer indices at which FA's kernel breaks invariance for THIS prompt
    MMQ_AmplifiesNoise,   \* layer indices at which MMQ amplifies noise; happens when an upstream layer has sub-fp32-eps drift but FA at this layer is byte-identical, and MMQ's amplification factor pushes the per-layer residual above F16 precision
    InitialDriftLevel  \* drift level at pc=0 (= entry of layer 0). Captures the empirical observation that upstream-of-this-snapshot kernels may have already introduced sub-eps drift. For OutlierP0 (p0 at layer 19), set to "sub_fp32_eps" to model the empirical state at layer 18 entry; for ModalP3, set to "byte_identical".

ASSUME NLayers \in Nat /\ NLayers > 0
ASSUME NPa \in Nat /\ NPa > 0
ASSUME NPb \in Nat /\ NPb > 0 /\ NPa /= NPb
ASSUME FA_LayerIndices \subseteq 0..(NLayers - 1)
ASSUME FA_ShapeSensitive \subseteq FA_LayerIndices
ASSUME MMQ_AmplifiesNoise \subseteq 0..(NLayers - 1)
ASSUME InitialDriftLevel \in { "byte_identical", "sub_fp32_eps" }

\* Per-run state: at each layer the slot-K residual is either byte-identical
\* between the two runs at this layer, or has diverged.
\* "diverged" is monotonic: once diverged, never un-diverges (cascade).

VARIABLES
    pc,               \* program counter; in 0..NLayers (NLayers = done)
    diverged_at,      \* layer index at which divergence first appeared (or -1 if none yet)
    drift_magnitude   \* abstract magnitude: byte_identical | sub_eps | visible

vars == << pc, diverged_at, drift_magnitude >>

DriftLevels == { "byte_identical", "sub_fp32_eps", "sub_f16_precision", "fp16_visible" }

\* ---------------------------------------------------------------------------
\* Init
\* ---------------------------------------------------------------------------

Init ==
    /\ pc = 0
    /\ diverged_at = -1
    /\ drift_magnitude = InitialDriftLevel

\* ---------------------------------------------------------------------------
\* Layer Compute actions
\* ---------------------------------------------------------------------------

\* The kernel-template-at-layer-L choice depends on layer type and batch
\* width. Per Allium @DispatcherTemplateUniformity_*: for NP in {2,4,8}
\* and the production sm_75 sm path, the dispatched template is the
\* same across NP values. The shape-sensitivity is INTERNAL to the
\* kernel, not in template choice.
\* So at this abstraction level we always pick the same template, and
\* the question is whether the same template produces the same output
\* across batch widths.

\* When the current layer L is FA AND L is in FA_ShapeSensitive, the
\* FA kernel produces visibly-different slot-K output. This is the
\* @WmmaFAShapeSensitiveAtSomeInputs Allium invariant.
FA_Drifts(L) ==
    /\ L \in FA_LayerIndices
    /\ L \in FA_ShapeSensitive

\* When the current layer L has MMQ amplification AND there is already
\* sub-eps drift entering, MMQ pushes drift to visible level. This is the
\* @MmqShapeSensitiveViaAmplification Allium invariant.
MMQ_Amplifies(L) ==
    /\ L \in MMQ_AmplifiesNoise
    /\ drift_magnitude \in { "sub_fp32_eps" }

\* Computing layer L:
LayerCompute ==
    /\ pc < NLayers
    /\ \/  \* Case 1: FA at L is shape-sensitive AND drift wasn't yet visible
           \* → drift becomes visible (fp16) at this layer's flash_attn output.
           \* This is the empirical p1/p3 layer 3 case.
           /\ FA_Drifts(pc)
           /\ drift_magnitude' = "fp16_visible"
           /\ diverged_at' = IF diverged_at = -1 THEN pc ELSE diverged_at
        \/  \* Case 2: MMQ amplifies sub-eps drift at this layer
           \* → drift jumps from sub_eps to fp16_visible.
           \* This is the empirical p0 layer 19 case.
           /\ MMQ_Amplifies(pc)
           /\ drift_magnitude' = "fp16_visible"
           /\ diverged_at' = IF diverged_at = -1 THEN pc ELSE diverged_at
        \/  \* Case 3: drift already visible — cascade. Per
           \* @DriftCascadeAfterFirstDivergence: once visible at slot K,
           \* all subsequent layers' slot-K residual also diverges.
           /\ drift_magnitude = "fp16_visible"
           /\ drift_magnitude' = "fp16_visible"
           /\ diverged_at' = diverged_at
        \/  \* Case 4: no kernel violates invariance at L; per-slot
           \* compute remains byte-identical.
           /\ ~FA_Drifts(pc)
           /\ ~MMQ_Amplifies(pc)
           /\ ~(drift_magnitude = "fp16_visible")
           /\ drift_magnitude' = drift_magnitude
           /\ diverged_at' = diverged_at
    /\ pc' = pc + 1

Next == LayerCompute

Spec == Init /\ [][Next]_vars

\* ---------------------------------------------------------------------------
\* Invariants
\* ---------------------------------------------------------------------------

\* Type invariant
TypeOK ==
    /\ pc \in 0..NLayers
    /\ diverged_at \in -1..(NLayers - 1)
    /\ drift_magnitude \in DriftLevels

\* @DriftCascadeAfterFirstDivergence (Allium): once diverged_at /= -1 AND
\* drift_magnitude is fp16_visible, the magnitude stays at fp16_visible (or
\* higher) forever.
DriftCascadeAfterFirstDivergence ==
    diverged_at /= -1 /\ drift_magnitude = "fp16_visible"
    => UNCHANGED drift_magnitude \/ drift_magnitude' = "fp16_visible"

\* @PerPromptDriftSiteVariability (Allium): different (FA_ShapeSensitive,
\* MMQ_AmplifiesNoise) configurations yield different first-divergent layers.
\* Modal case: FA_ShapeSensitive includes the first FA layer (3) — drift
\* visible at layer 3. Outlier case: FA_ShapeSensitive = {} AND
\* MMQ_AmplifiesNoise = {19} — drift visible at layer 19 only.
\* This is modeled by the constants; the TLC config files instantiate the
\* two cases separately (PipelineDeterminism_ModalP3.cfg and
\* PipelineDeterminism_OutlierP0.cfg).

\* "byte-identity" is NOT held at the END of execution in any configuration
\* that has either FA_ShapeSensitive non-empty or MMQ_AmplifiesNoise non-empty
\* with a feeding sub-eps drift. This is the
\* @NotHeld_LayerLevelByteIdentity_NP2_NP4 Allium non-invariant.
\* We express this as: if FA_ShapeSensitive is non-empty, there exists a
\* reachable state where drift_magnitude = fp16_visible at pc = NLayers.
\* That's a liveness-ish property; for a safety invariant, we check that
\* the converse (byte-identity at end) does NOT hold by negation.

ByteIdentityAtEnd_FailsIfFAShapeSensitiveOrMMQAmplifies ==
    (pc = NLayers /\ drift_magnitude = "byte_identical")
    => (FA_ShapeSensitive = {} /\ MMQ_AmplifiesNoise = {})

\* ---------------------------------------------------------------------------
\* Allium↔TLA+ synergy: invariant name mapping for check-bindings.py
\* ---------------------------------------------------------------------------

\* This module bounds these Allium invariants (must equal TRUE under Spec):
WmmaFAShapeSensitiveAtSomeInputs == TRUE
\* ^ Bound externally by data/deltanet/s2-3-op-level-localization.json
\* (p1, p3 layer 3 flash_attn output max_abs_diff > 0)

WmmaFAByteIdenticalAtOtherInputs == TRUE
\* ^ Bound externally by data/deltanet/s2-3-op-level-localization.json
\* (p0 layer 19 flash_attn output max_abs_diff = 0)

MmqShapeSensitiveViaAmplification == TRUE
\* ^ Bound externally by data/deltanet/s2-3-op-level-localization.json
\* (p0 layer 19 amplification 5.96e-8 → 0.006)

DispatcherTemplateUniformity_FA == TRUE
\* ^ Bound externally by static analysis of fattn.cu:140 + fattn-wmma-f16.cu:81-104

DispatcherTemplateUniformity_MMQ == TRUE
\* ^ Bound externally by static analysis of mmq.cuh:4175-4220

DispatcherTemplateUniformity_DeltaNet == TRUE
\* ^ Bound externally by static analysis of delta-net.cu:219

ProjectionGEMMSlotInvariance == TRUE
\* ^ Bound externally by data/deltanet/s2-3-op-level-localization.json
\* (Qcur, Kcur, Vcur etc. IDENTICAL at slot 0 across NP values)

DeltaNetSlotInvariance == TRUE
\* ^ Bound externally by data/deltanet/s2-3-op-level-localization.json
\* (DeltaNet outputs IDENTICAL at p3 layer 0 across NP values)

RepDeterminismAtSameNP == TRUE
\* ^ Bound externally by data/deltanet/s1-1-rep-identity.json

StorageBranchInvariant == TRUE
\* ^ Bound externally by DFLASH_DIAG trace

PerPromptDriftSiteVariability == TRUE
\* ^ Bound externally by data/deltanet/s2-1-per-prompt-first-div.json

TokenLevelGreedyArgmaxNP1_NP2_NP4_Identity == TRUE
\* ^ Renamed 2026-06-14 (was TokenLevelGreedyArgmaxNP1_NP2_Identity),
\*   np1==np2 extended to np1==np2==np4 after the nwarps=4 MMVQ fix + path B.
\*   Bound externally by data/phase_dflash_t8/gate7-token-diff-summary.json
\*   + PHASE_LM_DECODE_CAPTURE D1 (verify-production-determinism PASS 3/3 at
\*   NP_LIST="1 2 4").

NP8TokenDeterminismNotHeld_IntermittentTPRace == TRUE
\* ^ Corrected 2026-06-14 (was TokenLevelGreedyArgmaxNP4_NP8_DeterministicDrift).
\*   np=8 is an intermittent (~8%) single-slot host-timing race, NOT
\*   deterministic drift; a multi-GPU tensor-parallel determinism problem
\*   (PHASE_NP8_FLAKE). Bound externally by PHASE_LM_DECODE_CAPTURE D1
\*   (np=8 12 PASS / 1 FAIL over 13 reps, clocks+power pinned).

=============================================================================
