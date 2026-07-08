# Behavioural (.allium) and timing (.tla) specs for ik_llama.cpp

These specs constrain behaviour of the `ik_llama.cpp` inference engine (KV-cache
layout, multi-GPU dispatch, MTP, dflash, async reduce, quantized matmul, and the
rest). They live beside the software per the host methodology (template ledger
entry `b6232a5`), on this dedicated orphan `host-specs` branch — governance only,
kept separate from the code branches so it never advances the deployed line.

## Layout

```
specs/<topic>/          one dir per concern (kv-cache, dflash, dispatch, ...)
  <name>.allium         behavioural spec (allium 3)
  <Name>.tla            timing/concurrency spec (TLA+)
  <Name>.cfg            TLC model-checker config
  <name>.obligations    per-obligation disposition manifest (allium plan)
```

## Lanes

- **allium** (requirements): `allium check` + `allium analyse` are green on all 47
  specs. `allium plan` derives obligations; each is dispositioned in a sibling
  `.obligations` manifest. The obligations are presently `waived` pending property-test
  authoring (see `call/0013` in the host repo) — the specs are a check+analyse-clean
  behavioural model, not yet a discharged property-test suite.
- **TLC** (timing): each `.tla` is model-checked via `tla2tools.jar` v1.8.0 on Temurin
  21. The CI runs the curated base configs (must pass).

CI: `.github/workflows/specs.yml`. The host references this branch by pin
(`host-specs@<sha>` in `.host-software`).
