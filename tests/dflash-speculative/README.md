# tests/dflash-speculative

Behavioural test bundle for the DFlash speculative-decoding workstream.

**Spec:** `specs/dflash/dflash.allium`
**Design:** `specs/dflash/DESIGN.md`
**Branch:** `production/2026-q2-next` (yarn-agentic + ik_llama.cpp)

These tests are **RED-first**: the symbol-surface test binds at compile
time (it fails to compile until the API stubs in `include/llama.h` and
`src/llama-dflash.cpp` exist — they do, currently returning
`LLAMA_DFLASH_NOT_IMPLEMENTED`), and the behavioural tests skip
(exit 77) until real implementation lands. As the implementation
arrives, the skipped tests one-by-one assert correctness and turn GREEN.

Each test maps to one or more invariants or contracts in the spec.

| Test | Spec construct | What it binds |
|---|---|---|
| `test-dflash-symbols.cpp` | DFlash API surface | Compile-RED: enums, status codes, function symbols are linkable |
| `test-dflash-load-vocab-identity.cpp` | `DrafterTargetVocabIdentity` | Drafter and target share exact vocab_size and token id space |
| `test-dflash-load-dense-qwen3.cpp` | `DrafterIsDenseQwen3` | Drafter `layer_types[i]` is `FULL_ATTENTION` or `SLIDING_ATTENTION`; never `linear_attention` |
| `test-dflash-load-source-layer-count.cpp` | `SourceLayerCountMatchesDrafterTraining` | `n_source_layers` read from GGUF metadata (no hardcoded 5) |
| `test-dflash-load-block-size.cpp` | `BlockSizeFixedPerDeployment` | `block_size` read from GGUF metadata; stable across queries |

Run via `ctest -L dflash-speculative` after a build. Tests that exit
77 are CTest-skipped, not failed — they remain in the suite as
documentation of the next behavioural binding to land.
