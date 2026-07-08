# Running DFlash empirical gates on sm_75 (Turing)

Quickstart for the next session: how to run Gate-3.5 / Gate 0
against vLLM PR #40898 + z-lab/Qwen3.6-27B-DFlash on 2× RTX 6000.
Captures every nontrivial detail that took a session to discover.

## Prerequisites

Already in place on this host (verify before re-running):

- vLLM 0.20.2rc1.dev206+g23002d3f3 at `/opt/models/venv-vllm/`
  (PR #40898 head, python 3.13, torch 2.11+cu130)
- Target weights at `/mnt/archive/qwen3.6-27b-hf/` (52 GiB)
- Drafter at `/opt/models/qwen36-27b-dflash/` (3.3 GiB)
- Cache symlinks `~/.cache/{vllm,flashinfer,torch}`, `~/.triton`
  pointing into `/opt/models/cache/`

If the vLLM install is gone, re-build with:
```
systemd-run --user --unit=vllm-install --scope \
  --setenv=UV_CONCURRENT_BUILDS=1 \
  --setenv=MAX_JOBS=2 \
  --setenv=MAKEFLAGS=-j2 \
  ... uv pip install -e /opt/models/refs/vllm-pr-40898/vllm
```
(See feedback_dont_pressure_a_full_root memory entry for the
cgroup-isolated install rationale.)

## Environment for any gate invocation

`scripts/gate35-dflash-determinism.py` and
`scripts/gate0-dflash-speedup.py` set these as defaults, but if
running ad-hoc:

| env var | value | why |
|---|---|---|
| `HF_HOME` | `/mnt/archive/hf-cache` | Keep model downloads off / |
| `TMPDIR` | `/opt/models/tmp` | Triton scratch goes off / |
| `VLLM_CACHE_ROOT` | `/opt/models/cache/vllm` | Compiled vLLM graphs |
| `TRITON_CACHE_DIR` | `/opt/models/cache/triton` | Triton kernel cache |
| `TORCHINDUCTOR_CACHE_DIR` | `/opt/models/cache/torch-inductor` | TorchInductor compiled artefacts |
| `FLASHINFER_WORKSPACE_BASE` | `/opt/models/cache/flashinfer` | flashinfer JIT scratch |
| `VLLM_LOGGING_LEVEL` | `WARNING` | quieter logs |
| `CUDA_VISIBLE_DEVICES` | `0,1` | both 6000s |

## sm_75 monkey-patches (required)

Both harnesses import `scripts/vllm_sm75_patches.py` and call
`apply_all()` BEFORE importing vllm. Four patches:

1. `DFlashQwen3ForCausalLM.combine_hidden_states` — cast hidden
   states to fc weight dtype (fp32→fp16 mismatch fix).
2. `FlexAttentionImpl.forward` — `.view()` → `.reshape()` fallback
   for non-contiguous KV cache tensors.
3. `FlexAttention.get_kernel_options` — round `BLOCK_M`/`BLOCK_N`
   to next power-of-2 (Triton `tl.arange` requirement).
4. `DFlashQwen3ForCausalLM.precompute_and_store_context_kv` —
   dtype safety net for the second projection site (typically
   redundant given patch 1).

See `specs/dflash/upstream-pr-drafts.md` for the upstream-ready
versions.

## Firing a run

```
systemd-run --user --unit=gate35 --scope \
  --setenv=HF_HOME=/mnt/archive/hf-cache \
  --setenv=TMPDIR=/opt/models/tmp \
  --setenv=VLLM_CACHE_ROOT=/opt/models/cache/vllm \
  --setenv=TRITON_CACHE_DIR=/opt/models/cache/triton \
  --setenv=TORCHINDUCTOR_CACHE_DIR=/opt/models/cache/torch-inductor \
  --setenv=FLASHINFER_WORKSPACE_BASE=/opt/models/cache/flashinfer \
  --setenv=VLLM_LOGGING_LEVEL=WARNING \
  --setenv=CUDA_VISIBLE_DEVICES=0,1 \
  -- /opt/models/venv-vllm/bin/python \
     /home/llm/yarn-agentic/scripts/gate35-dflash-determinism.py \
  > /opt/models/gate35.log 2>&1 &
```

Run via `systemd-run --scope` so the workload lives in a cgroup
(easier to bound RSS and stop cleanly) and so logs survive shell
death.

## Expected wall-clock budget

- Model load (15 BF16→fp16 shards from /mnt/archive CIFS): ~10 min
- Drafter load + spec-decoder init: ~1 min
- JIT compile (Triton kernels for FlexAttention + flashinfer where
  used): ~5-10 min
- 3 generations × 96 tokens each, np=1 then np=2 then np=2: ~5 min
- Total wall time per Gate-3.5 invocation: **~25-30 min**

For Gate 0 (8 prompts, max_tokens=256, 2 LLM init cycles):
**~45-60 min**.

## Output

`/home/llm/yarn-agentic/data/gate35-dflash-determinism.json`
contains: verdict (GREEN/RED), n_diff_pairs, token streams,
decoded text, per-batch timings, divergence-position context for
each non-matching pair. See `scripts/gate35-dflash-determinism.py`
for the schema.

## When a run crashes

vLLM PR #40898 has multiple latent assumptions broken by the fp16
fallback path on sm_75. Each new failure mode is a candidate for a
runtime monkey-patch in `vllm_sm75_patches.py`. Pattern:

1. Read the crash log tail (filter out the `_POSIX_C_SOURCE` GCC
   warnings — they're harmless).
2. Identify the line in vLLM source that crashed.
3. Decide patch strategy:
   - dtype mismatch → cast input to weight dtype
   - `.view()` on non-contiguous → `.reshape()`
   - Triton `tl.arange` on non-POW2 → round constants UP to POW2
   - capability check too strict → relax conditional, document why
4. Add a `_patch_xxx` function to `vllm_sm75_patches.py`.
5. Enable in `apply_all()`.
6. Smoke-test via standalone python import.
7. Re-fire the gate.

## Escalation: force flashinfer backend (cheaper)

vLLM normally falls through its backend preference list:
1. `FLASH_ATTN` (rejected on sm_75 — no FA2 sm_75 wheel + capability check)
2. `FLASHINFER` (claims sm_75 support, head=128 supported, JIT-builds fine)
3. `TRITON_ATTN`
4. `FLEX_ATTENTION`
5. `TURBOQUANT`

When the current run falls to FlexAttention (where our patches live),
flashinfer was apparently skipped for some reason. To force flashinfer:

```python
from vllm.config import AttentionConfig
from vllm.platforms.interface import AttentionBackendEnum

llm = LLM(
    ...,
    attention_config=AttentionConfig(
        backend=AttentionBackendEnum.FLASHINFER,
    ),
)
```

Cheaper escalation than rebuilding flashinfer AoT. If flashinfer
accepts the workload, our FlexAttention patches become moot.

## Escalation: build flashinfer AoT for sm_75

If patches accumulate into more than 5–6 and the failure surface
keeps growing, switch strategies:

```
cd /opt/models/refs/vllm-pr-40898/flashinfer
TORCH_CUDA_ARCH_LIST=7.5 \
FLASHINFER_ENABLE_AOT=1 \
  /opt/models/venv-vllm/bin/pip install -e . --no-build-isolation
```

AoT mode bakes sm_75 kernels into the binary, eliminating the JIT
compile path entirely. ~1 hour build, but the FlexAttention fallback
(which is where most of our patches live) goes away.

## Reference

- Spec: `specs/dflash/dflash.allium`
- TLA models: `specs/dflash/DFlashCycle.tla`,
  `specs/dflash/DFlashMultiSlot.tla`
- vLLM source we monkey-patch: `/opt/models/venv-vllm/lib/python3.13/
  site-packages/vllm/` — see line numbers in each `_patch_*` docstring
- Patch upstream candidates: `specs/dflash/upstream-pr-drafts.md`
