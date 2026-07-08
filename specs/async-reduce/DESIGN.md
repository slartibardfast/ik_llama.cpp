# Phase AsyncReduce — design (Option B)

## Context

Phase CY.F.16 Option A landed (arch-force `reduce_type = F32`) and achieves byte-determinism at NP ∈ {1, 4, 8}. The cost is **~2× cross-device reduce bandwidth at prefill** — each per-device F32 tensor (n_embd × n_tokens × 4 bytes) crosses the PCIe peer link in full F32, where the F16 cast previously halved it.

Option B is the SoTA permanent fix: **async F32 cross-device reduce with stream-overlap**, hiding the bandwidth cost behind layer-level compute.

## Targets

- Single binary deliverable: a new `ggml_reduce_async` op (or augmented `ggml_reduce` mode) that:
  - Accepts F32 per-device inputs
  - Schedules the peer-access copy on a **dedicated communication CUDA stream**
  - Signals a CUDA event on completion
  - Downstream consumers `cudaStreamWaitEvent` before reading
- Determinism: bit-identical output to the synchronous F32 reduce already shipped in Option A.
- Perf: prefill PP at NP=8 matches or beats the pre-Option-A path (which was F16-cast, batch-shape-non-deterministic).
- No correctness regressions: all existing tests (`test-fattn-*`, `test-rmsnorm-*`, `test-mmq-*`, V4 production harness) PASS after switch.

## Non-targets

- Replacing the existing synchronous reduce path (kept as fallback when stream-overlap isn't possible — single-device, AMD, etc.).
- Reduce-scatter sharding of downstream consumers (Option C; future).
- Cross-host inter-GPU reduce (NCCL territory; out of scope).

## Architecture

```
Device 0:                                        Device 1:
+--------------------+                          +--------------------+
| compute stream     |                          | compute stream     |
|   ... FFN ops ...  |                          |   ... FFN ops ...  |
|   per-dev result   |                          |   per-dev result   |
|   tensor[0] = F32  |                          |   tensor[1] = F32  |
|   record evt_done0 |                          |   record evt_done1 |
+--------------------+                          +--------------------+
         |                                                |
         | wait                                           | wait
         v                                                v
+--------------------+                          +--------------------+
| comm stream        |  cudaMemcpyPeerAsync     | comm stream        |
|   peer-copy        |  ----------------------> |   peer-copy        |
|   tensor[1] -> dev0|  <---------------------- |   tensor[0] -> dev1|
|   ggml_add F32     |                          |   (mirror)         |
|   record evt_red0  |                          |   record evt_red1  |
+--------------------+                          +--------------------+
         |                                                |
         | wait                                           | wait
         v                                                v
+--------------------+                          +--------------------+
| compute stream     |                          | compute stream     |
|   wait evt_red0    |                          |   wait evt_red1    |
|   downstream FFN   |                          |   downstream FFN   |
|   (RMSNorm etc.)   |                          |   (RMSNorm etc.)   |
+--------------------+                          +--------------------+
```

**Key insight**: while comm stream is busy peer-copying layer N's residual, compute stream can start layer N+1's matmul (which doesn't depend on this reduce's result yet because layer N+1 reads layer N's already-completed FFN output, not the upcoming reduced sum). Or more precisely: the compute stream waits on `evt_red` only at the point where downstream needs the reduced result — typically the RMSNorm at the START of the next layer. So compute up to that point runs in parallel with comm.

**Hidden cost**: TU102 PCIe 3.0 x16 peer bandwidth ≈ 12 GB/s. Per-layer reduce transfer at NP=8 prefill: 5120 × 96 × 4 bytes = 1.96 MiB ≈ 160 µs. Per-layer compute ≈ 10–20 ms. **Transfer fits in <2% of compute window — easily hidden.**

## Allium contracts

`specs/async-reduce/async-reduce.allium`:

- `@DeterministicAcrossSchedule` — output bit-identical regardless of which CUDA stream runs first
- `@F32Throughout` — no precision-losing cast anywhere in the reduce path
- `@EventOrderingHoldsBeforeRead` — downstream RMSNorm starts only after `cudaStreamWaitEvent` succeeds
- `@NoDataRaceOnReducedTensor` — peer-copy writes precede event signal
- `@StreamIsolation` — comm stream operations don't reorder compute stream operations and vice versa
- `@OverlapCompute` — when downstream waits on reduce event AFTER non-dependent compute, overlap is observed (perf invariant, validated via nsys)
- `@FallbackToSync` — when stream-overlap is impossible (single device, AMD), behavior degrades to the existing synchronous F32 reduce

## TLA+ properties

`specs/async-reduce/AsyncReduce.tla`:

- **DeadlockFreedom**: scheduler can always make progress; comm stream waiting on compute does not block compute stream waiting on comm in a cycle
- **SafetyConsistency**: any read after `WaitEvent` returns the reduced F32 value (no partial / stale)
- **LinearizabilityAcrossStreams**: per-layer reduces complete in layer-order from the consumer's perspective (no out-of-order reads of layer N+1's reduce before layer N's)
- **TerminationUnderFairness**: with fair scheduler, all events eventually signal

## Implementation files (planned, in order)

1. **API surface** — extend `ggml_reduce` or add `ggml_reduce_async`:
   - `ggml/include/ggml.h` — public op declaration + `ggml_reduce_async_desc` struct (comm stream, event)
   - `ggml/src/ggml.c` — graph node creation
2. **CUDA backend** — new dedicated comm stream + event:
   - `ggml/src/ggml-cuda/reduce.cu` — async kernel dispatch
   - `ggml/src/ggml-cuda.cu` — comm stream allocation, event lifecycle
3. **Scheduler integration**:
   - `ggml/src/ggml-backend.cpp` — `cudaStreamWaitEvent` insertion at consumer
4. **Build-graph wiring**:
   - `src/llama-build-context.cpp` — switch `ggml_reduce` → `ggml_reduce_async` for FFN cross-device sum
   - `src/llama-delta-net.cpp` — same for DeltaNet output reduce
5. **Tests**:
   - `tests/test-ggml-reduce-async-correctness.cpp` — output bit-identical to sync F32 reduce
   - `tests/test-ggml-reduce-async-overlap.cpp` — nsys-based timeline verification
   - `tests/test-ggml-reduce-async-deadlock.cpp` — stress with many concurrent layers
   - Production V4 harness — must PASS at NP={1, 2, 4, 8}

## Migration plan

- **T1** — author Allium spec, run `check-bindings.py`, lock invariants
- **T2** — author TLA+ spec, verify deadlock-freedom and linearizability under TLC
- **T3** — implement API surface + CUDA backend
- **T4** — switch FFN reduce to async, validate correctness (bit-identity to Option A)
- **T5** — switch DeltaNet reduce to async, validate
- **T6** — perf measurement: prefill PP at NP=8 vs Option A; expect ~match (overlap hides cost)
- **T7** — re-validate V4 harness multi-run stability
- **T8** — flip default for split_mode=GRAPH multi-GPU (keep sync as fallback for single-device)
- **T9** — measure decode perf (should be unaffected since cast didn't fire at decode either way)
- **T10** — Option A's arch-force can be REMOVED once Option B is default — Option B gives same determinism without the bandwidth cost

## Open questions

1. Does ggml's existing CUDA backend already use multiple streams per device? If so, can we hook into the existing infrastructure or do we need a new stream class?
2. Is `cudaMemcpyPeerAsync` actually overlap-able with compute on TU102, or does the PCIe transaction serialize against in-flight kernels?
3. Should `evt_done` and `evt_red` be merged into a single event per layer-reduce, or kept separate for finer-grained scheduling?
4. How does this interact with `split_mode=GRAPH` graph reuse (`cparams.graph_reuse`)? Reused graphs cache the kernel launch list — do we need new launch list per call?

## Closure binding (composite gate)

Ship Option B as the new default when all of:

1. T1–T5 GREEN (correctness via bit-identity to Option A)
2. T6 prefill PP ≥ Option A (and target: match pre-Option-A F16-cast path within ±5%)
3. T7 V4 NP={1, 2, 4, 8} byte-deterministic across 10 consecutive runs
4. T8 fallback path verified on single-device (no regression)
5. T10 Option A arch-force can be safely removed
