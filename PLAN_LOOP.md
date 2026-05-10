# PHASE46 C.1 — Multi-slot MTP Determinism: Remaining Bugfix (Looping Tasks)

## Context

`ik_llama.cpp` (branch `production/2026-q2`) has a determinism bug that fires only with `-mtp + np>1 + concurrent dispatch + same-prompt-across-slots`. Symptom: the SSM/DeltaNet recurrent kernel's first per-block call diverges from subsequent calls, so byte-equality across slots fails.

**Diagnosis is complete** (22 Ralph iterations, 35 submodule commits). The fix path is to replace N consecutive per-block SSM kernel calls in `delta_net::build_qkv` with **one** multi-seq call — the kernel already supports `n_seqs > 1`. Steps 1–4 are landed in the submodule:

- `22090bc8` step 1/4 — plumb `state_seq_ids_multi` through `build_qkv`
- `33679051` step 2/4 — multi-seq state view assembly
- `1222e63c` step 3/4 — multi-seq output extraction + per-seq writeback
- `19ae255f`/`bf80c60f`/`fa772b92`/`bc023ac5` step 4/4 — `blocks_loop` multi-seq dispatch (env-gated `LLAMA_DNET_MULTI_SEQ=1`), with `ggml_cont` insertions for contiguous asserts

**Remaining blocker (step 5/5)**: when multi-seq dispatch fires, it passes `lctx.default_decoder.inp_s_seq_qnext` (shape `(1, n_tokens)`, slot-index format) into `ggml_ssm_conv`. The conv kernel asserts `sq->ne[0] == n_kv` where `n_kv = n_state_seqs > 1`, so it needs a **mask tensor** of shape `(n_state_seqs, n_tokens)` where `mask[k, t] = 1` iff token `t` belongs to seq `k`. Construct a sibling `inp_s_seq_qnext_multi` input tensor, fill it from the existing slot-index data at decode time, and pass it through the multi-seq dispatch site.

**Production bug fix already landed** at `0885b695` (`llama_kv_cache::checkpoint_save` GGML layout assert) — single-GPU + `-mtp` boots; not part of this plan.

## Looping tasks

Each task is sized to fit a 5-minute prompt-cache window. Default state: `[ ]` open. Close `[x]` only when the verification check binds on the step's actual claim (per `mtp-agentic/CLAUDE.md` §5 checkbox semantics).

### L.1 — Add `inp_s_seq_qnext_multi` field declaration  [x]

Add the I32 mask-tensor pointer next to existing `inp_s_seq_qnext` field, plus its initializer.

- Edit: `src/llama-decoder-internal.h:105` — add `struct ggml_tensor * inp_s_seq_qnext_multi = nullptr; // I32 [n_seq_max, n_batch]` directly below the existing `inp_s_seq_qnext` field.
- Edit: `src/llama-build-context.cpp:111` — add `lctx.default_decoder.inp_s_seq_qnext_multi = nullptr;` directly below the existing `inp_s_seq_qnext` initializer.
- Verify: `cmake --build build -j 32 --target llama llama-server` compiles cleanly. ✅ Done: targets linked. Pre-existing failure in `tests/mtp-verify-accept/test-accept-symbols.cpp` (unrelated symbol-renaming, ≠ this plan's scope) noted but not addressed (per `mtp-agentic/CLAUDE.md` §3 — note dead code, do not delete/silently fix).

### L.2 — Allocate mask tensor in graph build  [x]

In `src/graphs/build_qwen35.cpp`, allocate the new mask tensor right next to the existing `inp_s_seq_qnext` allocation at line 44. Mark it as a runtime-fillable input.

- Edit: after line 46 (`ggml_set_input(lctx.default_decoder.inp_s_seq_qnext);`), add:
  ```cpp
  const int64_t n_seq_max_local = (int64_t) lctx.cparams.n_seq_max;
  lctx.default_decoder.inp_s_seq_qnext_multi = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_seq_max_local, n_tokens);
  cb(lctx.default_decoder.inp_s_seq_qnext_multi, "inp_s_seq_qnext_multi", -1);
  ggml_set_input(lctx.default_decoder.inp_s_seq_qnext_multi);
  ```
- Repeat the same insertion at the second site (line 168) for the alternate codepath.
- Verify: build clean; server still boots at `np=1`. ✅ Done: `cmake --build build --target llama llama-server` clean; np=1 + `--draft 3` boot returns `{"status":"ok","slots_idle":1,"slots_processing":0}` from `/health`.

### L.3 — Fill mask tensor in input callback  [x]

In `src/llama.cpp` around line 4745 (the closing brace of the `inp_s_seq_qnext` fill block), append a parallel fill block that converts per-token slot indices into a `(n_kv, n_tokens)` mask. **Note:** must be defensive against `->buffer == nullptr` because the graph allocator prunes the tensor when no consumer is wired (initial segfault was a NULL buffer dereference inside `ggml_backend_buffer_is_host`); skip the fill silently if buffer is null. Once L.4 wires the consumer, the buffer becomes non-null and the fill runs.

- Edit: after the `for (int64_t j = 0; j < n_tokens; ++j) { ... data[j] = slot; }` block, add:
  ```cpp
  if (lctx.default_decoder.inp_s_seq_qnext_multi) {
      GGML_ASSERT(ggml_backend_buffer_is_host(lctx.default_decoder.inp_s_seq_qnext_multi->buffer));
      int32_t * mdata = (int32_t *) lctx.default_decoder.inp_s_seq_qnext_multi->data;
      const int64_t n_kv = lctx.default_decoder.inp_s_seq_qnext_multi->ne[0];
      const int64_t nt   = lctx.default_decoder.inp_s_seq_qnext_multi->ne[1];
      GGML_ASSERT(nt == n_tokens);
      for (int64_t k = 0; k < n_kv; ++k) {
          for (int64_t j = 0; j < n_tokens; ++j) {
              mdata[k * nt + j] = (data[j] == (int32_t) k) ? 1 : 0;
          }
      }
  }
  ```
- Verify: build clean; with `LLAMA_DNET_MULTI_SEQ=0` (default) and `np=3 +mtp`, server still boots and emits identical output to baseline (the new tensor is allocated but pruned by the allocator until L.4 wires it). ✅ Done: build clean; np=3 + `--draft 3` boots, `/health` returns `{"status":"ok","slots_idle":3}`, completion `"The capital of France is"` → `" Paris.\nThe capital of France is Paris.\nThe"`.

### L.4 — Switch dispatch site to use mask tensor  [x]

In `src/llama-delta-net.cpp:929–932`, replace the third argument to `build_layer_attn_linear_core` with the mask tensor when multi-seq dispatch is active.

- Edit at line 930: change `lctx.default_decoder.inp_s_seq_qnext,` to `lctx.default_decoder.inp_s_seq_qnext_multi,`.
- Verify: clean rebuild (`rm -rf build && cmake -B build && cmake --build build --target llama llama-server`). ✅ Done: clean rebuild green; 374/374 targets including `bin/llama-server`. Per `feedback_clean_cuda_rebuilds`.

### L.5 — Boot regression at `np=1`  [x]

Confirm `LLAMA_DNET_MULTI_SEQ=0` (default) still produces baseline behavior at `np=1 +mtp`.

- Run: `LLAMA_DNET_MULTI_SEQ=0 ./build/bin/llama-server -m <Qwen3.5-0.8B-BF16.gguf> -np 1 --draft 3 -c 8192 --port 18080` in a terminal; from another, `curl localhost:18080/health` and one `/v1/completions` request.
- Verify: server starts, `/health` 200 OK, completion emits coherent tokens. ✅ Done: ready in ~5s, `{"status":"ok","slots_idle":1}`, completion `"The capital of France is"` → `" Paris.\nThe capital of France is Paris.\nThe"`.

### L.6 — Boot at `np=3 +mtp` with multi-seq enabled  [x]

Activate the new path and confirm no asserts.

- Run: `LLAMA_DNET_MULTI_SEQ=1 ./build/bin/llama-server -m <Qwen3.5-0.8B-BF16.gguf> -np 3 --draft 3 -c 8192 --port 18081`.
- Issue 3 same-prompt requests concurrently (`xargs -P3` of `curl`).
- Verify: server does not crash; no `GGML_ASSERT` in journal/stderr; all 3 responses return. ✅ Done: server ready in ~5s; 3 concurrent same-prompt requests all returned (same `created` timestamp confirms concurrency); no GGML_ASSERT, no segfault. Early signal: all 3 produced text starting `" Paris.\nThe capital of France is Paris.\nThe capital of France"` — strong indicator that L.7 byte-equality will pass, but the formal sha256 binding is L.7's job.

### L.7 — Determinism sweep on Qwen 3.5 0.8B (np ∈ {2, 3, 4, 8})  [x]

Run the determinism check across the full np sweep on the rapid-iteration model. Each np value is a separate inner loop iteration; close the L.7 box only when **all four** rows pass.

- For each `NP ∈ {2, 3, 4, 8}`:
  - Run: `LLAMA_DNET_MULTI_SEQ=1 ./build/bin/llama-server -m <Qwen3.5-0.8B-BF16.gguf> -np $NP --draft 3 -c 8192 --port 18081 &`
  - Issue `$NP` same-prompt requests concurrently (`xargs -P$NP` of `curl /v1/completions`); capture per-slot output streams.
  - sha256 each slot's full output; record into a results table (`np, draft, sha256_count_distinct`).
- Verify: for every `NP`, `sha256_count_distinct == 1` (all slots produced byte-equal output). ✅ ALL PASS:
- Sub-rows:
  - [x] L.7.np2 — np=2 byte-equal — sha256 `a3395601…e499` ×2 → 1 distinct
  - [x] L.7.np3 — np=3 byte-equal — sha256 `a3395601…e499` ×3 → 1 distinct
  - [x] L.7.np4 — np=4 byte-equal — sha256 `60d82442…84a4` ×4 → 1 distinct
  - [x] L.7.np8 — np=8 byte-equal — sha256 `a3395601…e499` ×8 → 1 distinct

(Note: np=4's hash differs from np=2/3/8 — same prompt, but batch shape varies with slot count, which is allowed; the binding claim is per-run cross-slot byte-equality, which all four runs satisfy.)

### L.8 — Determinism sweep on production Qwen 3.6 27B (np ∈ {2, 3, 4, 8})

Repeat the sweep on the production model. ctx is reduced for higher np to stay within 48 GiB aggregate VRAM.

- For each `(NP, CTX)` ∈ `{(2, 262144), (3, 262144), (4, 131072), (8, 65536)}`:
  - Run: `LLAMA_DNET_MULTI_SEQ=1 ./build/bin/llama-server -m <Qwen3.6-27B-IQ4.gguf> -np $NP --draft 3 -c $CTX --device CUDA0,CUDA1 --split-mode graph --tensor-split 1,1 --port 18082 &`
  - Issue `$NP` same-prompt concurrent requests; sha256 compare.
- Verify: every `(NP, CTX)` row produces byte-equal slot outputs at `--draft 3` (production setting in `profiles/qwen36.sh`). If `np=8 @ ctx 64K` does not fit in VRAM, drop ctx to `32768` and note in MEMORY.md; do not skip the row.
- Sub-rows (must all close):
  - [ ] L.8.np2 — np=2 @ 262K byte-equal
  - [ ] L.8.np3 — np=3 @ 262K byte-equal
  - [ ] L.8.np4 — np=4 @ 131K byte-equal
  - [ ] L.8.np8 — np=8 @ 64K byte-equal

### L.9 — Default-on guard or keep env-gated decision

Decide based on L.7+L.8 outcomes whether to flip `LLAMA_DNET_MULTI_SEQ` default to on, leave env-gated for soak, or auto-detect (multi-seq dispatch when `n_state_seqs > 1` AND all blocks same length AND `mtp` flag set).

- If both green: change default in `src/llama-delta-net.cpp:907–909` from "off unless env=1" to "on unless env=0", AND extend the multi-seq dispatch precondition at line 910 from `multi_seq_dispatch && blocks.size() > 1` to `(multi_seq_dispatch || lctx.cparams.mtp) && blocks.size() > 1 && all_same_length`. Land as a separate commit so the soak window is reviewable.
- If only L.7 green / L.8 fails: keep env-gated, MEMORY.md entry naming the model-size-specific failure, leave step open as `[ ]` with a subtask for the 27B-specific bisection.
- Verify: production engine at `/home/llm/ik_llama.cpp/` updated, `systemctl --user restart llama-server`, healthcheck.sh green, 3-slot determinism replays clean.

### L.10 — Documentation closure

Append-only MEMORY.md entry summarizing the fix (5 commits across 4 files; specific commit SHAs; final byte-equality result on both models). Update `mtp-agentic/PLAN.md` C.1 row to `[x]` with verification evidence quote. Cancel any leftover diagnostic env-gates that are now obsolete (note in MEMORY, do not delete per repo CLAUDE.md §3).

- Verify: `git -C /home/llm/mtp-agentic log --oneline -5` shows MEMORY + PLAN commits; mdBook static site regenerates with C.1 marked done.

## Critical files

**Will edit** (5 files, ~50–80 LOC total):
- `/home/llm/mtp-agentic/ik_llama.cpp/src/llama-decoder-internal.h:105` (L.1)
- `/home/llm/mtp-agentic/ik_llama.cpp/src/llama-build-context.cpp:111` (L.1)
- `/home/llm/mtp-agentic/ik_llama.cpp/src/graphs/build_qwen35.cpp:44, 166` (L.2)
- `/home/llm/mtp-agentic/ik_llama.cpp/src/llama.cpp:4745` area (L.3)
- `/home/llm/mtp-agentic/ik_llama.cpp/src/llama-delta-net.cpp:929–932, 906–910` (L.4, L.9)

**Read for reference**:
- `/home/llm/mtp-agentic/ik_llama.cpp/ggml/src/ggml.c:10388` (the `sq->ne[0] == n_kv` assert that step 5 satisfies)
- `/home/llm/mtp-agentic/ik_llama.cpp/ggml/src/ggml-cuda/ssm-conv.cu` (kernel that consumes the mask)
- Iter 22 entry in `/home/llm/mtp-agentic/MEMORY.md` (handover spec this plan operationalizes)

**No new files**.

## Verification (end-to-end)

```bash
cd /home/llm/mtp-agentic/ik_llama.cpp

# Clean rebuild (touched headers; ninja can miss CUDA fan-out deps)
rm -rf build
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF
cmake --build build -j 32

# Sanity (no behavior change with default-off)
./build/bin/test-qnext-multislot-dispatch
./build/bin/test-backend-ops

# Qwen 3.5 0.8B determinism sweep (L.7) — for NP in 2 3 4 8
for NP in 2 3 4 8; do
  LLAMA_DNET_MULTI_SEQ=1 ./build/bin/llama-server -m <0.8B.gguf> \
    -np $NP --draft 3 -c 8192 --port 18081 &
  # $NP concurrent same-prompt requests, sha256 compare → expect 1 distinct hash
done

# Qwen 3.6 27B determinism sweep (L.8) — np × ctx pairs sized to 48 GiB VRAM
for PAIR in "2 262144" "3 262144" "4 131072" "8 65536"; do
  set -- $PAIR; NP=$1; CTX=$2
  LLAMA_DNET_MULTI_SEQ=1 ./build/bin/llama-server -m <27B.gguf> \
    -np $NP --draft 3 -c $CTX --device CUDA0,CUDA1 --split-mode graph \
    --tensor-split 1,1 --port 18082 &
  # $NP concurrent same-prompt requests, sha256 compare → expect 1 distinct hash
done
```

## Out of scope

- Tracks A (PHASE45 throughput finish), B (determinism API + harness), D (ceiling exploration), E (future-hw async overlap). Those are in the broader determinism program; this plan is scoped to closing the C.1 multi-slot SSM bugfix only.
- Removing diagnostic env-gates (14 in tree from iters 1–22). Keep for now; remove in a separate cleanup PR after soak.
- Upstream PR. In-fork only.

## Looping discipline

- One task per loop iteration. Commit & push after each task lands (per repo CLAUDE.md §5 — PLAN edits also per-edit).
- Any task that takes >1 cache window: stop, name the gap as a subtask under that L.x in this plan file, leave the box `[ ]`, hand back to user.
- Do **not** declare a step `[x]` unless its verification check binds on its actual claim (e.g., L.7 closes only when 3 sha256s are byte-equal — server-boots-without-crash is L.6, not L.7).
- After L.8 closes (all four np ∈ {2, 3, 4, 8} sub-rows green on both 0.8B and 27B), the bug is fixed against the user's stated target. L.9–L.10 are closure tasks.
