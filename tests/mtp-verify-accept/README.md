# mtp-verify-accept — RED tests for the verify-step accept-decision

These tests are propagated from `mtp_verify_accept.allium` (in the
parent yarn-agentic repo) and drive test-first implementation of the
verify-step accept-decision that sits between the per-ubatch MTP-KV
hook and the post-accept KV chop:

```
[hook fires per ubatch]
   -> verify decision        <-- THIS BUNDLE
   -> seq_rm rejected tail
   -> fused draft round
```

The accept-decision is host-side: its inputs are the draft_tokens
vector and the per-row verify logits (which the speculative caller
pulls from the main forward via `llama_get_logits_ith`). The contract
is a pure function — same inputs always produce the same decision.

Each test fails today (RED) by referencing symbols that do not yet
exist:
- `enum llama_accept_mode` — `argmax_match` / `probabilistic`
- `struct llama_accept_decision` — `{n_accepted, bonus_token, bonus_pos}`
- `llama_mtp_accept_verify()` — the contract entry point

The tests are designed so that each one drives a single spec invariant
green when an implementation slice lands.

## Test → spec invariant map

| Test | Spec invariant(s) bound |
|---|---|
| `test-accept-symbols.cpp`              | `contract-signature.AcceptVerify.invoke`, compile-time presence of `llama_accept_mode` / `llama_accept_decision` |
| `test-accept-logits-shape.cpp`         | `LogitsShapeMatchesUbatch` |
| `test-accept-longest-prefix.cpp`       | `LongestPrefixMatchUnderArgmax`, `NAcceptedWithinBound` |
| `test-accept-bonus-correctness.cpp`    | `BonusIsArgmaxAtFirstUnacceptedRow` |
| `test-accept-bonus-position.cpp`       | `BonusPosIsSeedPlusNAcceptedPlusOne` |
| `test-accept-determinism.cpp`          | `DeterminismUnderFixedLogits` |
| `test-accept-progress.cpp`             | `VerifyAlwaysMakesProgress` |
| `test-accept-probabilistic-rejected.cpp` | `ProbabilisticModeOutOfScope` |
| `test-accept-committed-shape.cpp`      | `TotalCommittedIsNAcceptedPlusOne`, `CommittedSequenceShape` |
| `test-accept-chop-coordination.sh`     | `ChopRangeFollowsAcceptDecision` (cross-spec → mtp_ubatch_hook.KvRangeChop) |
| `test-accept-bonus-becomes-seed.sh`    | `BonusBecomesNextSeed` (cross-round) |

## Implementation hooks the tests reference

These symbols must land for the bundle to compile, let alone pass:

```cpp
// include/llama.h — additions

enum llama_accept_mode {
    LLAMA_ACCEPT_MODE_ARGMAX_MATCH  = 0,
    LLAMA_ACCEPT_MODE_PROBABILISTIC = 1,
};

struct llama_accept_decision {
    int32_t      n_accepted;
    llama_token  bonus_token;
    int32_t      bonus_pos;
};

LLAMA_API int32_t llama_mtp_accept_verify(
        struct llama_context        * ctx,
        const llama_token           * draft_tokens,
        int32_t                       n_drafts,
        int32_t                       pos_seed,
        enum llama_accept_mode        mode,
        struct llama_accept_decision * out);
```

The implementation reads per-row logits via `llama_get_logits_ith`
(or, on hot paths, via the on-device `llama_get_draft_argmax`
cached argmax index) and runs the longest-prefix comparison
host-side. No GPU work is intrinsic to this contract.

## Build and run

After the symbols above land:

```
cd ik_llama.cpp
cmake --build build --target test-accept-symbols test-accept-logits-shape \
                            test-accept-longest-prefix test-accept-bonus-correctness \
                            test-accept-bonus-position test-accept-determinism \
                            test-accept-progress test-accept-probabilistic-rejected \
                            test-accept-committed-shape
ctest -R accept
```

The two `.sh` cross-spec tests live alongside the existing
`tests/mtp-rollout-investigation/_common.sh` harness pattern and
require an instrumented server build (debug endpoints listed in
each test's header).
