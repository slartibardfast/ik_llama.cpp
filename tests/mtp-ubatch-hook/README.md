# mtp-ubatch-hook — RED tests for the per-ubatch MTP-KV hook

These tests are propagated from `mtp_ubatch_hook.allium` (in the parent
yarn-agentic repo) and drive test-first implementation of the inline
per-ubatch MTP-KV update mechanism that replaces our existing
`MTP_OP_WARMUP` / `MTP_OP_UPDATE_ACCEPTED` post-hoc decode paths.

## Test → spec invariant map

| Test | Spec invariant(s) bound |
|---|---|
| `test-hook-tag-tensor.cpp` | `THPreNormTensorTagged` |
| `test-hook-fires-once.cpp` | `FiresOncePerUbatch`, `SingleMtpDecodePerInvocation` |
| `test-hook-no-secondary-decode.cpp` | `NoSecondaryDecode` |
| `test-hook-cross-ubatch-pairing.cpp` | `CrossUbatchPairingOnContinuation`, `CrossUbatchResetOnDiscontinuation` |
| `test-hook-idempotent-chop.cpp` | `IdempotentUnderRepeatedChop` |
| `test-hook-lockstep.sh` | `MtpKvLockstepWithMainKv` (server-level) |
| `test-hook-reject-tail.sh` | `MtpKvHasAcceptedPrefix`, `MtpKvLacksRejectedSuffix`, `ChopRangeIsRejectedSuffix` (server-level) |
| `test-hook-sole-populator.sh` | `HookIsSoleMtpKvPopulator` (server-level, instrumented) |

## Implementation hooks the tests reference

These symbols must land for the cgraph-level tests to compile:

```cpp
// include/llama.h — additions

// Observability: read the count of hook firings since context create.
LLAMA_API int32_t llama_mtp_hook_fire_count(struct llama_context * ctx);

// Observability: read the count of MTP-block decodes (i.e. graph
// computes on the MTP-only graph) since context create.
LLAMA_API int32_t llama_mtp_decode_count(struct llama_context * ctx);

// Tagged-tensor lookup for the main forward graph's residual stream
// row tagged "h_pre_norm" before final norm.
LLAMA_API struct ggml_tensor * llama_main_graph_h_pre_norm(
        struct llama_context * ctx);

// Inspection of MTP-layer KV slot presence (used by lockstep,
// reject-tail tests; also referenced by mtp-fused/ tests).
LLAMA_API bool llama_mtp_kv_has_pos(
        struct llama_context * ctx,
        llama_seq_id           seq_id,
        llama_pos              pos);
LLAMA_API int32_t llama_mtp_kv_pos_max(
        struct llama_context * ctx,
        llama_seq_id           seq_id);
```

## Build and run

```
cd ik_llama.cpp
cmake --build build --target test-hook-tag-tensor test-hook-fires-once \
                            test-hook-no-secondary-decode \
                            test-hook-cross-ubatch-pairing \
                            test-hook-idempotent-chop
ctest -R hook
bash tests/mtp-ubatch-hook/test-hook-lockstep.sh
bash tests/mtp-ubatch-hook/test-hook-reject-tail.sh
bash tests/mtp-ubatch-hook/test-hook-sole-populator.sh
```

## Note on infrastructure scope

`test-hook-lockstep.sh`, `test-hook-reject-tail.sh`, and
`test-hook-sole-populator.sh` are server-level integration tests in
the same shape as `tests/mtp-rollout-investigation/`. They require a
GGUF with MTP head preserved (e.g.
`/opt/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0-MTP.gguf`) and a built
`llama-server`. The cgraph-level tests (`*.cpp`) require neither and
are the test-first foundation.

`test-hook-sole-populator.sh` requires an instrumented build that
exposes per-MTP-KV-write source attribution (hook vs fused_draft).
This may need a separate test-harness extension beyond what
test-backend-ops can express today; that work is tracked separately.
