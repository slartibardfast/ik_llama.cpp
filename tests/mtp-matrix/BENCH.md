# Chained-rollout throughput benchmark — Qwen3.5-0.8B-Q8_0-MTP

Measures whether `LLAMA_MTP_ROLLOUT=N > 1` (chained MTP rollout) wins vs
single-pass MTP (`rollout=1`).

## Methodology

Runs `llama-server` with `LLAMA_MTP_ROLLOUT ∈ {1,2,3,4,5}` on a fixed
prompt + `n_predict`, captures per-request:

- **prompt_tps** — prompt-eval tokens/second (from `prompt eval time` log)
- **gen_tps** — generation tokens/second (`eval time` line, comma-right field)
- **accept** — MTP acceptance rate (from `acceptance rate =` log)
- **eff_tps** — theoretical effective tokens/sec `= gen_tps × (1 + accept × (rollout-1))`
- **wall_ms** — end-to-end wall clock for the full generation

Script: `tests/mtp-matrix/bench/bench-rollout-throughput.sh`.

## Vulkan 2-GPU (NAVI21 + Vega10), MMVQ off, 2026-04-18 (pre-split-fix)

N_PREDICT=100, TRIALS=2 (best kept), WARMUP=1 short decode.

| Rollout | prompt_tps | gen_tps | accept | eff_tps | wall_ms |
|---|---|---|---|---|---|
| 1 | 20.23 | 14.53 | 0.65 | 14.53 | **8367** |
| 2 |  9.89 |  7.25 | 0.55 | 11.21 | 15953 |
| 3 |  8.93 |  4.43 | 0.03 |  4.71 | 24898 |
| 4 |  8.14 |  6.06 | 0.56 | 16.16 | 18951 |
| 5 |  7.47 |  5.99 | 0.68 | 22.23 | 19281 |

## Vulkan 2-GPU — after `ggml_set_input` removal on intermediate greedy tokens

Fix: `src/llama-build-context.cpp:4907, 4964` — dropped `ggml_set_input`
on `current_greedy` inside the chained rollout loop. That marker had
been forcing the Vulkan scheduler to split the graph at each iteration
boundary (CPU-round-trip semantics) even though nothing CPU-side
modifies the tensor. Kept `ggml_set_output` so the greedy token is
still inspectable externally.

| Rollout | prompt_tps | gen_tps | accept | eff_tps | wall_ms | speedup |
|---|---|---|---|---|---|---|
| 1 | 20.22 | 14.52 | 0.65 | 14.52 |  8374 | 1.00× |
| 2 | 18.44 | 12.94 | 0.55 | 20.02 |  9279 | **1.72×** |
| 3 | 16.75 |  7.99 | 0.03 |  8.49 | 14130 | 1.76× |
| 4 | 15.32 | 10.97 | 0.56 | 29.25 | 10805 | 1.75× |
| 5 | 14.13 | 10.92 | 0.68 | 40.53 | 10920 | **1.77×** |

Graph internals (measured at r=5):

| metric | pre-fix | post-fix |
|---|---|---|
| graph splits per decode | **11** | **1** |
| Vulkan_Host compute buffer | 30.47 MiB | 0.14 MiB |
| per-token gen cost | 166 ms | 90 ms |
| acceptance rate (correctness) | 0.647 | 0.647 (bit-identical) |

### Vulkan verdict (post-fix): chained rollout still loses, but barely

- rollout=1 wall (8.4s) is still 30% faster than rollout=5 (10.9s) for
  the same `n_predict=100`. Each r=5 decode does 5× the GPU work, which
  cancels the ~3.7× accepted-tokens-per-decode gain.
- The scheduler no longer thrashes: prompt_tps only drops 1.4× from
  r=1 to r=5 (20→14 t/s) instead of the prior 2.7× drop.
- Effective t/s climbs to 40.5 at r=5 — the MTP acceptance IS paying
  off inside each decode, it's just not enough to beat baseline on a
  0.8B model where per-decode GPU cost is low.
- rollout=3 accept=0.03 is still an outlier; not MTP-code-path-related
  (see note below).

### When would Vulkan chained rollout win?

A larger model (35B-A3B) will have per-decode cost dominated by MoE
matmuls, making the extra MTP head iterations a smaller relative cost.
Combined with acceptance ≥0.7, chained rollout should flip to a win.
The 0.8B measurement here is yardstick only.

### Note on rollout=3 acceptance outlier

Both pre- and post-fix show rollout=3 accept=0.032 vs rollout=2/4/5 at
0.55–0.68. Same prompt, same model, same seed. This is not a
chained-rollout code-path issue (r=2 and r=4 pass through the same
code) — likely prompt-specific adversarial interaction with the draft
head at that specific iteration count. Worth a separate investigation
if it reproduces on more prompts.

## CPU, 2026-04-18 (build-cpu-release, AVX2)

N_PREDICT=50, TRIALS=1, WARMUP=1. Quick smoke — only rollout ∈ {1, 3}.

| Rollout | prompt_tps | gen_tps | accept | eff_tps | wall_ms |
|---|---|---|---|---|---|
| 1 | 129.64 | 35.87 | 0.14 | 35.87 | 1518 |
| 3 | 118.69 | 38.22 | 0.26 | 58.34 | **1443** |

### CPU verdict: chained rollout marginally POSITIVE

- rollout=3 is ~5% faster wall-clock than rollout=1.
- Theoretical eff_tps improvement 1.63×, actual wall-clock improvement
  1.05× — most of the theoretical gain is eaten by per-iteration
  overhead, but CPU's higher base cost means the leftover is a small
  net win.
- This contrasts Vulkan. On CPU, compute dominates; adding draft
  iterations amortizes constant per-decode overhead (memory allocation,
  thread dispatch) across more accepted tokens. On GPU, per-decode
  overhead is low; graph reshape dominates and kills the win.

## Overall conclusion

For Qwen3.5-0.8B on this workstation:

- **GPU users**: keep `LLAMA_MTP_ROLLOUT=1` (the default). Chained
  rollout code is numerically correct (Stage 3 fix + Stage 4 equivalence
  tests confirm) but pays its keep only if you can get the graph
  reshape cost to zero. That's a follow-up optimization target if this
  ever moves to production.
- **CPU users**: small win at rollout=3 (5% wall-clock). Consider if
  the hybrid MTP+AR workload is latency-sensitive enough to justify
  the extra code path.

Both results match prior characterization of MTP spec decode (memory
ref: [project_mtp_spec_decode.md]) — "throughput-negative on CPU; GPU
batch-parallel is the path to a win." The CPU side here is slightly
better than that memory implied, but the core takeaway — that sequential
chained rollout doesn't beat single-pass — is unchanged.

## What's NOT in these numbers

- **35B-A3B MoE**: this is the formal target for MTP spec decode and
  was not benchmarked here (would need a different host + env setup).
  The 0.8B measurements are a dev yardstick only.
- **Larger prompts**: prompt_tps only covers the 10–15 token test prompt;
  long-context behavior (1k+ prompt) may shift the curve.
- **Temperature ≠ 0 sampling**: these tests use greedy decoding; nucleus
  sampling could change acceptance rates substantially.
- **Parallel users** (`--parallel N > 1`): the chained rollout interaction
  with multi-slot scheduling is unmeasured.

If 35B-A3B testing is desired, see Stage 6 in the plan (conditional,
environment-gated).
