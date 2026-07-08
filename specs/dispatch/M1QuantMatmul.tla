--------------------------- MODULE M1QuantMatmul ---------------------------
(*****************************************************************************)
(* TLA+ spec for the observational equivalence of decode under the          *)
(* small-batch (ne11 <= K) quantized-matmul kernel substitution ("the M1    *)
(* kernel"). Co-designed with the Allium contract                           *)
(*   specs/dispatch/m1_quant_matmul.allium                                  *)
(* as one spec artifact (binding: specs/dispatch/allium-tla-binding.json).  *)
(*                                                                           *)
(* The decode-dominant cost on sm_75 is the Q4_0 / Q4_0_AR16 weight matmul. *)
(* Today (ggml-cuda.cu:2801-2806) quantized matmuls are forced through the  *)
(* MMQ split-K GEMM path "regardless of batch size" for shape-pinned        *)
(* determinism; at ne11 <= 8 the mmq_x=8 N-tile wastes (8-ne11)/8 of the    *)
(* int8 tensor-core work. The M1 kernel computes only the ne11 real columns *)
(* (no N-tile waste) for a decode speedup. The risk it must not incur is    *)
(* changing the decode output: the Allium FaithfulByteIdentity obligation   *)
(* requires M1's column-k output == the mmq_x=8 kernel's column-k output,   *)
(* byte-for-byte, for EVERY real column k < ne11.                           *)
(*                                                                           *)
(* WHY THE FAMILY (ne11 in [1,K]), not just M=1: multi-slot serving         *)
(* (--parallel N>1 -> decode batch = active slots) and speculative decoding *)
(* (MTP draft-K -> batch K+1) produce decode batches M=2..K. MMQ is         *)
(* batch-invariant (batch-invariance.allium MMQ_FaithfulPropagation), so a  *)
(* column's output is independent of how many columns share its batch.      *)
(* Hence if M1's column-k is byte-identical to MMQ's column-k for every k,  *)
(* substituting M1 for any ne11 <= K leaves the token stream unchanged      *)
(* ACROSS np: a token decoded at M=1 or as column k of an M=4 batch gets    *)
(* identical bits. That preserves TokenLevelGreedyArgmaxNP1_NP2_NP4_Identity*)
(* and its multi-slot extension np1 == np2 == ... == npK. K is a perf       *)
(* threshold (the crossover where MMQ's amortized overhead catches the      *)
(* read-once GEMV); correctness is independent of K.                        *)
(*                                                                           *)
(* MODEL: a BASELINE run (current MMQ at every batch width) and a TEST run  *)
(* (M1 substituted) decode columns at nondeterministic batch widths         *)
(* M in 1..MaxBatch. "argmax token" is abstract, parameterised by the       *)
(* kernel Mechanism and the batch width M. The load-bearing invariant is    *)
(* ObservationalEquivalence: the two runs' token streams agree at every     *)
(* reachable state — the scheduler-level refinement of FaithfulByteIdentity.*)
(*                                                                           *)
(* NEGATIVE CONTROL is FAMILY-SPECIFIC. M1QuantMatmul_Divergent.cfg sets    *)
(* Mechanism = "M1_DIVERGENT_MULTICOL": a kernel that is CORRECT at the     *)
(* batch-1 column (it would pass a naive M=1 byte-compare) but WRONG for a  *)
(* column in a multi-column batch (M>=2) — modelling a kernel that violates *)
(* the per-column FaithfulByteIdentity / M1MustMatchMmqScaleAccumOrder only *)
(* at ne11>1. TLC MUST find a counterexample (ObservationalEquivalence      *)
(* violated) — proving the FAMILY contract bites where a batch-1-only test  *)
(* would not, i.e. it catches the np1 != npK break. The positive config     *)
(* (M1QuantMatmulMC.cfg, Mechanism = "M1_FAITHFUL") verifies it holds.      *)
(*                                                                           *)
(* CODE REFS (paths from /home/dconnolly/yarn-agentic):                      *)
(*   ik_llama.cpp/ggml/src/ggml-cuda.cu:2801-2819   dispatch (MMQ force)     *)
(*   ik_llama.cpp/ggml/src/ggml-cuda/mmq.cu + mmq.cuh   the mmq_x=8 kernel   *)
(*   ik_llama.cpp/ggml/src/ggml-cuda/quantize.cu   quantize_q8_1 (block_q8_1)*)
(*                                                 vs quantize_mmq_q8_1      *)
(*                                                 (block_q8_1_mmq)          *)
(*****************************************************************************)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    BaselineToken,      \* token a faithful kernel (and MMQ) emits for a column
    PerturbedToken,     \* token a non-faithful column emits
    MaxStep,            \* bound on the decode-step counter for finite MC
    MaxBatch,           \* K — the max carve-out batch width (ne11 in 1..MaxBatch)
    Mechanism           \* in Mechanisms — the kernel the TEST run uses for ne11 <= K.
                        \* The BASELINE run always uses "MMQ_BASELINE".

VARIABLES
    test_tokens,        \* Seq of tokens emitted so far by the TEST run (M1 for ne11<=K)
    base_tokens,        \* Seq of tokens emitted so far by the BASELINE run (MMQ)
    step_count

vars == <<test_tokens, base_tokens, step_count>>

----------------------------------------------------------------------------
(* Constants and helpers.                                                    *)
----------------------------------------------------------------------------

Mechanisms == {"MMQ_BASELINE", "M1_FAITHFUL", "M1_DIVERGENT_MULTICOL"}

\* Predicate: the kernel is byte-identical to the MMQ mmq_x=8 per-column output.
\* MMQ_BASELINE trivially is; a correct M1 (M1_FAITHFUL) is by the Allium
\* FaithfulByteIdentity obligation (block_q8_1_mmq per M1MustUseMmqSrc1Quant,
\* MMQ's k-block float-scale order per M1MustMatchMmqScaleAccumOrder), at EVERY
\* real column. M1_DIVERGENT_MULTICOL is NOT — it is faithful only at the
\* batch-1 column.
Faithful(m) ==
    \/ m = "MMQ_BASELINE"
    \/ m = "M1_FAITHFUL"

\* The argmax token a run emits for a column whose decode batch has width M
\* (M in 1..MaxBatch). A faithful kernel emits BaselineToken regardless of M
\* (byte-identity + MMQ batch-invariance: the column's output does not depend on
\* how many columns share its batch). M1_DIVERGENT_MULTICOL emits BaselineToken
\* at M=1 (passes a naive batch-1 byte-compare) but PerturbedToken at M>=2 —
\* the family-specific failure mode that breaks np1 != npK and that a
\* batch-1-only contract/test cannot catch.
EmitsToken(m, M) ==
    IF Faithful(m)         THEN BaselineToken
    ELSE IF M = 1          THEN BaselineToken
    ELSE                        PerturbedToken

----------------------------------------------------------------------------
(* Init.                                                                     *)
----------------------------------------------------------------------------
Init ==
    /\ test_tokens = <<>>
    /\ base_tokens = <<>>
    /\ step_count = 0

----------------------------------------------------------------------------
(* Action: Decode                                                            *)
(*                                                                           *)
(* Both runs decode one column at a nondeterministic batch width M in        *)
(* 1..MaxBatch (modelling np=1 .. np=K and spec-decode draft widths). The    *)
(* baseline uses the MMQ kernel; the test run uses the constant Mechanism.   *)
(* Each appends its argmax token for that column.                            *)
(*****************************************************************************)
Decode ==
    /\ step_count < MaxStep
    /\ \E M \in 1..MaxBatch:
        /\ test_tokens' = Append(test_tokens, EmitsToken(Mechanism, M))
        /\ base_tokens' = Append(base_tokens, EmitsToken("MMQ_BASELINE", M))
    /\ step_count' = step_count + 1

----------------------------------------------------------------------------
(* Next-state.                                                               *)
----------------------------------------------------------------------------
Next == Decode

Spec == Init /\ [][Next]_vars /\ WF_vars(Decode)

----------------------------------------------------------------------------
(* Invariants.                                                               *)
----------------------------------------------------------------------------

TypeOK ==
    /\ test_tokens \in Seq({BaselineToken, PerturbedToken})
    /\ base_tokens \in Seq({BaselineToken})
    /\ step_count \in 0..MaxStep
    /\ Mechanism \in Mechanisms
    /\ MaxBatch \in Nat \ {0}

\* The load-bearing safety property. At every reachable state, the two runs'
\* token streams are equal — substituting the M1 kernel for any ne11 <= K does
\* not change the decode output at any batch width. The scheduler-level
\* refinement of FaithfulByteIdentity; preserves
\* batch-invariance.allium TokenLevelGreedyArgmaxNP1_NP2_NP4_Identity (and its
\* multi-slot extension).
ObservationalEquivalence ==
    test_tokens = base_tokens

\* The mechanism-level binding of the Allium FaithfulByteIdentity obligation:
\* a faithful M1 kernel (byte-identical at every real column) keeps
\* ObservationalEquivalence. The negative control (M1_DIVERGENT_MULTICOL)
\* falsifies it ONLY at M>=2 — which is exactly the family bug a batch-1
\* contract misses.
FaithfulByteIdentity ==
    Faithful(Mechanism) => (test_tokens = base_tokens)

----------------------------------------------------------------------------
(* Liveness — the test run eventually decodes at least once, so the spec     *)
(* cannot pass vacuously by never exercising the substituted kernel.         *)
----------------------------------------------------------------------------
EventuallyDecodes ==
    <>(Len(test_tokens) > 0)

==============================================================================
