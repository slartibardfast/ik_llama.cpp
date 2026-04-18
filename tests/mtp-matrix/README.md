# MTP test matrix

Systematic backend × configuration × test matrix for MTP chained rollout on
ik_llama.cpp. Replaces the ad-hoc "it doesn't crash" gate that kept letting
subtle regressions through.

## Structure

```
tests/mtp-matrix/
├── lib/                    # shared helpers (sourced by test scripts)
│   ├── _common.sh          # matrix_setup / matrix_launch / matrix_completion
│   ├── _logits.sh          # logits_capture_to / logits_diff
│   └── _build_matrix.sh    # canonical list of builds
├── coherence/              # model-level: does it generate coherent text?
├── semantics/              # logit-level: iter0=single-pass, CPU↔Vulkan equiv
├── shape/                  # n_tokens boundary sweep × rollout sweep (42 tests)
├── ops/                    # ggml op-level tests (integrated into test-backend-ops.cpp)
├── scheduler/              # reserve_n / compute sequencing
├── server/                 # startup + lifecycle combos
├── regressions/            # byte-identical anchors for known-good behavior
└── run-all.sh              # runner — emits Markdown table of (build, test) results
```

## Builds covered

| id | label | binary | env | extra args |
|---|---|---|---|---|
| cpu-release | CPU Release (-O3) | build-cpu-release | | |
| cpu-debug | CPU Debug (-O0 -g) | build-cpu-debug | | |
| cpu-asan | CPU ASAN (-O1 -g) | build-cpu-asan | `new_delete_type_mismatch=0:halt_on_error=0` | |
| cpu-ubsan | CPU UBSAN (-O1 -g) | build-cpu-ubsan | `halt_on_error=0` | |
| vk-mmvq-off | Vulkan 2-GPU, MMVQ off | build-vk | `GGML_VK_DISABLE_MMVQ=1` | `-ngl 999` |
| vk-mmvq-on | Vulkan 2-GPU, MMVQ on | build-vk | | `-ngl 999` |
| vk-navi-only | Vulkan 1-GPU NAVI21 | build-vk | `GGML_VK_DISABLE_MMVQ=1` | `-ngl 999 -ts 1,0 -mg 0` |
| vk-vega-only | Vulkan 1-GPU Vega10 | build-vk | `GGML_VK_DISABLE_MMVQ=1` | `-ngl 999 -ts 0,1 -mg 1` |
| vk-no-pp | Vulkan no pipeline-parallel | build-vk | `GGML_VK_DISABLE_MMVQ=1` | `-ngl 999 -sm none` |

Enable additional builds by placing the binary at
`build-<id>/bin/llama-server` (the runner auto-detects availability).

## Test categories

### coherence/
Model-level correctness: server generates a coherent response to "The capital
of France is" — output must contain "paris" (case-insensitive) within 10
tokens. Variants:
- `test-coherence-r{1..8}.sh` — one per rollout count
- `test-coherence-long-r1.sh` — n_predict=50, catches late-generation degradation
- `test-coherence-deterministic.sh` — same prompt, two calls, byte-identical

### semantics/
Logit-level correctness via completion_probabilities:
- `test-iter0-matches-single-pass.sh` — chained iter=0 logits must match rollout=1
  logits within 5e-3. Validates that adding rollout>1 doesn't corrupt iter=0.
- `test-backend-equivalence-r1.sh` — CPU and Vulkan must agree within 5e-2 at
  rollout=1. Requires `BIN_B` env with oracle binary path.
- `test-per-iteration-preservation.sh` — iter=0 must not change between
  rollout=2 and rollout=3. Catches scheduler aliasing between iterations.

### shape/
Scheduler graph-shape boundary sweep: 14 n_tokens values × 3 rollouts = 42
tests. Maps the exact n_tokens where behavior transitions (crash vs success).

### ops/
Integrated into `tests/test-backend-ops.cpp`:
- `test_concat_chain` — 2..5 sequential `ggml_concat(dim=1)` on f32
  [32768, N]. Validates the MTP stacking pattern.
- `test_argmax_getrows` — argmax → get_rows with set_input/set_output on
  intermediates. Validates the polaris pattern for scheduler preservation.

Run op tests with: `GGML_VK_DISABLE_MMVQ=1 ./build-vk/bin/test-backend-ops -b Vulkan0 -o CONCAT`.

### scheduler/
Reserve_n and compute interleave probes. Diagnose exactly which sequence
triggers the RADV heap corruption:
- `test-reserve-reserve-no-compute.sh` — reserve-reserve without intermediate
  compute. If this passes but `-compute-reserve` fails, bug requires compute.
- `test-reserve-compute-reserve.sh` — the classic failure pattern.
- `test-same-shape-repeat.sh` — 5 identical requests; catches state leaks.
- `test-shape-flip-flop.sh` — alternating shapes, forcing repeated reserve_n.
- `test-growing-shape-sweep.sh` — monotonically growing n_tokens to find exact
  shape boundary where crash triggers.

### server/
Server lifecycle coverage: startup with each (rollout, --no-warmup, --parallel,
-mtp toggle) combination.

### regressions/
Byte-identical anchors for known-good behavior:
- `test-exact-paris.sh` — output must start with exactly " Paris."
- `test-accept-floor.sh` — acceptance rate ≥ 0.25 (baseline 0.33)
- `test-tps-floor.sh` — t/s ≥ 8.0 (baseline ~12)
- `test-double-request-same-answer.sh` — determinism

## Running

```bash
# Full matrix (all builds × all categories)
bash tests/mtp-matrix/run-all.sh

# One category
bash tests/mtp-matrix/run-all.sh coherence

# One build
BUILD=vk-mmvq-off bash tests/mtp-matrix/run-all.sh

# One test
BIN=build-vk/bin/llama-server EXTRA_ENV="GGML_VK_DISABLE_MMVQ=1" EXTRA_ARGS="-ngl 999" \
    bash tests/mtp-matrix/coherence/test-coherence-r1.sh
```

Each test exits with:
- 0 = PASS
- 1 = FAIL (assertion)
- 2 = SKIP (infrastructure)
- 3 = CRASH (server died)

`run-all.sh` emits a Markdown table to stdout. Non-zero exit if any
FAIL/CRASH.

## Exit-code meaning at scale

- **All CPU builds FAIL on coherence tests** — known CPU backend numerical bug
  for Qwen3.5-MTP (produces deterministic garbage output "center有多么..."
  regardless of ASAN). Tracked as a separate issue; CPU is not a valid oracle
  for this model until fixed.
- **All Vulkan builds PASS on rollout=1** — baseline works across all Vulkan
  configs (2GPU, 1GPU NAVI21, 1GPU Vega, no-PP). MMVQ on/off doesn't affect
  rollout=1.
- **All Vulkan builds FAIL on rollout>=2** — RADV driver heap corruption is
  not a multi-GPU, pipeline-parallel, or MMVQ-specific issue; it's triggered
  by some aspect of the chained-rollout graph compute common to all Vulkan
  variants.

The matrix isolates these orthogonal bugs so fixes can proceed without
conflating them.
