------------------------- MODULE CreateSplitBalance -------------------------
(* CreateSplitBalance: TLA+ specification of the create_split() function at
   src/llama-load-tensors.cpp:351-414, the row-chunking primitive at the
   heart of the ik fork's GRAPH-mode multi-GPU allocation. Phase 46 Path B
   extracts this function to a shared header (src/ggml-mgpu-split.cpp) so
   that both LM and CLIP consume it; this spec (§12.2 spec #3) asserts the
   properties B.0 closure requires before that extraction happens.

   The function divides nchunk integer chunks among NDevice devices,
   biased by --tensor-split ratio and per-device mem_used pressure.
   Three-phase algorithm:

     1. Initial-allocation loop (load-tensors.cpp:364-373).
        For each device i, compute a desired share, round to int, clamp
        negative to 0, accumulate into sum. May over- or under-allocate.

     2. Down-correction loop (load-tensors.cpp:374-393).
        While sum > nchunk, find the device with largest positive error
        (result[i] - desired[i]) AND result[i] > 0, decrement.

     3. Up-correction loop (load-tensors.cpp:394-411).
        While sum < nchunk, find the device with largest positive error
        (desired[i] - result[i]), increment.

     4. Granularity scaling (load-tensors.cpp:412).
        Multiply each result[i] by granularity. Trivially preserves the
        sum invariant; not modeled here.

   Properties verified:
     - Safety: result is always non-negative (NonNegative).
     - Safety: sum tracks Sum(result) (SumTracks).
     - Liveness: both correction loops terminate (DownTerminates,
       UpTerminates, LoopTerminates).
     - Safety: at exit, sum = nchunk (SumInvariantAtExit) — the binding
       contract that maps to MgpuSplitConfig.allium's @CapacityHonored
       composed with the broader allocation pipeline.

   Model abstraction: floating-point arithmetic in the C++ original
   (the rounding + mem_used bias) is abstracted into a non-deterministic
   choice of any valid initial allocation. The correction loops then
   operate deterministically (or rather, with a non-deterministic
   tie-breaking choice that still satisfies the same invariants). This
   is sound because the properties asserted here do not depend on the
   exact floating-point values — only on the integer post-conditions.

   Composition: takes MgpuSplitConfig.allium's @CapacityHonored,
   @MemUsedNonNegative, @SplitsMonotonic as axioms. Hands off to
   CrossCodepathConsistency.allium (spec #5), which asserts that this
   algorithm produces the same result under LM consumption as under
   CLIP consumption.
*)

EXTENDS Naturals, FiniteSets, Sequences, TLC

CONSTANTS
    NDevice,         \* number of participating devices
    NChunk,          \* total chunks to distribute (= nr / granularity)
    MaxInitialSum    \* upper bound on initial-loop sum (TLC bounding)

ASSUME
    /\ NDevice         \in Nat \ {0}
    /\ NChunk          \in Nat \ {0}
    /\ MaxInitialSum   \in Nat \ {0}
    /\ MaxInitialSum >= NChunk \div 2   \* keep the model interesting

\* Result vector — chunks allocated to each device.
VARIABLES
    result,      \* function [1..NDevice -> Nat]
    sum,         \* current sum of result[]
    i,           \* index for init loop
    pc           \* "init_loop" | "down_loop" | "up_loop" | "done"

vars == << result, sum, i, pc >>

\* Sum of a function over a domain. (Workaround: TLA+ has no built-in
\* sum over function values.)
RECURSIVE SumF(_, _)
SumF(f, S) ==
    IF S = {} THEN 0
    ELSE LET x == CHOOSE x \in S : TRUE
         IN  f[x] + SumF(f, S \ {x})

InitialResult == [d \in 1..NDevice |-> 0]

Init ==
    /\ result = InitialResult
    /\ sum    = 0
    /\ i      = 1
    /\ pc     = "init_loop"

\* ============================================================
\* Phase 1: initial-allocation loop
\* (mirrors load-tensors.cpp:364-373)
\* ============================================================
\* For each device, pick any non-negative integer ≤ MaxInitialSum.
\* This abstracts away the floating-point arithmetic: the spec's
\* invariants must hold no matter what the initial allocation is.
InitLoopStep ==
    /\ pc = "init_loop"
    /\ i <= NDevice
    /\ \E v \in 0..MaxInitialSum :
         /\ result' = [result EXCEPT ![i] = v]
         /\ sum'    = sum + v
         /\ i'      = i + 1
         /\ pc'     = "init_loop"

InitLoopExit ==
    /\ pc = "init_loop"
    /\ i > NDevice
    /\ result' = result
    /\ sum'    = sum
    /\ i'      = i
    /\ pc'     = IF sum > NChunk THEN "down_loop"
                 ELSE IF sum < NChunk THEN "up_loop"
                 ELSE "done"

\* ============================================================
\* Phase 2: down-correction loop
\* (mirrors load-tensors.cpp:374-393)
\* ============================================================
\* While sum > nchunk: find any device d with result[d] > 0,
\* decrement it. The real code picks the device with largest
\* positive error; we model with non-deterministic choice over
\* {d : result[d] > 0}, which is sound because the invariants
\* we check (non-negativity, termination, sum-tracking) hold
\* under any choice in that set.
DownStep ==
    /\ pc = "down_loop"
    /\ sum > NChunk
    /\ \E d \in 1..NDevice :
         /\ result[d] > 0
         /\ result' = [result EXCEPT ![d] = result[d] - 1]
         /\ sum'    = sum - 1
         /\ i'      = i
         /\ pc'     = IF sum - 1 > NChunk THEN "down_loop" ELSE "done"

\* ============================================================
\* Phase 3: up-correction loop
\* (mirrors load-tensors.cpp:394-411)
\* ============================================================
\* While sum < nchunk: find any device (no result>0 guard),
\* increment it. Non-deterministic over all devices.
UpStep ==
    /\ pc = "up_loop"
    /\ sum < NChunk
    /\ \E d \in 1..NDevice :
         /\ result' = [result EXCEPT ![d] = result[d] + 1]
         /\ sum'    = sum + 1
         /\ i'      = i
         /\ pc'     = IF sum + 1 < NChunk THEN "up_loop" ELSE "done"

Next ==
    \/ InitLoopStep
    \/ InitLoopExit
    \/ DownStep
    \/ UpStep
    \/ /\ pc = "done"
       /\ UNCHANGED vars

Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ WF_vars(Next)

\* ============================================================
\* INVARIANTS
\* ============================================================

TypeOK ==
    /\ result \in [1..NDevice -> Nat]
    /\ sum    \in Nat
    /\ i      \in 1..(NDevice + 1)
    /\ pc     \in {"init_loop", "down_loop", "up_loop", "done"}

\* result[d] never goes negative.
NonNegative ==
    \A d \in 1..NDevice : result[d] >= 0

\* sum always tracks Sum(result). (Safety on the bookkeeping
\* variable — the C++ code maintains this by construction;
\* the spec models it explicitly so TLC catches off-by-one bugs
\* in any future refactor.)
SumTracks ==
    sum = SumF(result, 1..NDevice)

\* The binding post-condition that B.1's extraction must preserve:
\* at termination, sum = nchunk.
SumInvariantAtExit ==
    pc = "done" => sum = NChunk

\* Approximate balance: at exit, no device's allocation differs
\* from the "ideal" (NChunk / NDevice) by more than NChunk —
\* a trivially-true bound here since result[d] ≤ sum = NChunk
\* and the lower bound is 0. The real balance is asserted
\* observationally via @CapacityHonored and the empirical
\* per-device VRAM check in §3 Goal #2. The point of this
\* trivial bound is to confirm the spec's state space is
\* sound (every reachable result has all entries in [0, NChunk]).
BoundedAtExit ==
    pc = "done" =>
        \A d \in 1..NDevice : result[d] \in 0..NChunk

\* ============================================================
\* LIVENESS
\* ============================================================

\* Both correction loops terminate. Decisive argument:
\*   DownStep:  decreases sum by 1; sum is bounded below by NChunk
\*              (by guard) and starts at finite MaxInitialSum.
\*   UpStep:    increases sum by 1; sum is bounded above by NChunk.
\* TLC verifies these under WF_vars(Next).
LoopTerminates == <>(pc = "done")

\* Refined liveness: termination from a specific starting condition.
\* (These follow from LoopTerminates; spelled out for documentation.)
DownTerminates == (pc = "down_loop") => <>(pc = "done")
UpTerminates   == (pc = "up_loop")   => <>(pc = "done")

=============================================================================
\* End MODULE CreateSplitBalance
