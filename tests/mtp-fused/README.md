# mtp-fused — RED tests for the multi-step fused MTP draft graph

These tests are propagated from `mtp_fused_draft.allium` (in the parent
yarn-agentic repo) and drive test-first implementation of the fused
N-step MTP draft graph.

Each test fails today (RED) by referencing symbols that do not yet
exist in this fork:
- `LLAMA_MTP_OP_DRAFT_GEN_FUSED` — new variant of `enum llama_mtp_op_type`
- `llama_mtp_fused_draft_invoke()` — new public API in `include/llama.h`
- `llama_mtp_fused_step_count_t` — small struct returning observability
- `ggml_backend_cuda_graph_compute_call_count()` — instrumentation API
  used to assert SingleGraphCompute

The tests are designed so that each one drives a single spec invariant
green when an implementation slice lands. The intent is that the
implementation works one invariant at a time, with the corresponding
test going RED → GREEN as the slice lands.

## Test → spec invariant map

| Test | Spec invariant(s) bound |
|---|---|
| `test-mtp-fused-symbols.cpp` | `contract-signature.FusedDraftStep.invoke`, compile-time presence of `LLAMA_MTP_OP_DRAFT_GEN_FUSED` |
| `test-mtp-fused-step-count-bound.cpp` | `StepCountConformsToBound`, `StepCountWithinBound` |
| `test-mtp-fused-single-compute.cpp` | `SingleGraphCompute`, `NoSyncBetweenSteps` |
| `test-mtp-fused-argmax-correctness.cpp` | `ArgmaxCorrectnessIntrinsic`, `ArgmaxWinnerWhenArgmaxOnly`, `NTokensProducedInOrder` |
| `test-mtp-fused-determinism.cpp` | `DeterminismUnderArgmax` |
| `test-mtp-fused-kv-coverage.cpp` | `KvWritesAreExactlyNSteps`, `KvWrittenAtPositionsPThroughPPlusN` |
| `test-mtp-fused-kv-chop-rewrite.cpp` | `KvChopRewriteIsIdempotent` |
| `test-mtp-fused-prob-populated.cpp` | `ProbabilityFieldPopulated` |

## Implementation hooks the tests reference

These symbols must land in the implementation for the tests to compile,
let alone pass:

```cpp
// include/llama.h — additions

enum llama_mtp_op_type {
    LLAMA_MTP_OP_NONE             = 0,
    LLAMA_MTP_OP_WARMUP           = 1,
    LLAMA_MTP_OP_UPDATE_ACCEPTED  = 2,
    LLAMA_MTP_OP_DRAFT_GEN        = 3,
    LLAMA_MTP_OP_DRAFT_GEN_FUSED  = 4,    // NEW (mtp_fused_draft.allium)
};

// Result of one fused draft round.
struct llama_mtp_fused_result {
    int32_t      n_steps;
    int32_t      tokens[LLAMA_MTP_FUSED_MAX_STEPS];
    float        probs[LLAMA_MTP_FUSED_MAX_STEPS];
};

LLAMA_API int32_t llama_mtp_fused_draft_invoke(
        struct llama_context     * ctx,
        llama_token                seed_token,
        const float              * seed_hidden,    // n_embd floats
        int32_t                    n_steps,
        struct llama_mtp_fused_result * out);

// Observability — increments per ggml_graph_compute call on this backend.
LLAMA_API int32_t llama_mtp_fused_last_compute_count(struct llama_context * ctx);
```

Tests that need a working CUDA backend are gated behind `if (GGML_CUDA)`
in CMakeLists.

## Build and run

After implementing the symbols above:

```
cd ik_llama.cpp
cmake --build build --target test-mtp-fused-symbols test-mtp-fused-step-count-bound \
                            test-mtp-fused-single-compute test-mtp-fused-argmax-correctness \
                            test-mtp-fused-determinism test-mtp-fused-kv-coverage \
                            test-mtp-fused-kv-chop-rewrite test-mtp-fused-prob-populated
ctest -R mtp-fused
```
