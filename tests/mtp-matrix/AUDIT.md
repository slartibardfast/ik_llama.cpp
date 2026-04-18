# Code audit — Class A (async-without-sync) and Class B (permute-without-cont)

Systematic audit following the two bugs found this port:
- The **MTP chained-rollout crash** = async-without-sync at `src/llama.cpp:4303-4309`
- The **CPU DeltaNet garbage output** = IQK kernel hardcoding strides where
  upstream graph passes permuted views

Goal: find any more sites of the same two bug classes and prevent future
regressions of the same shape.

## Class A — `ggml_backend_tensor_get_async` without subsequent sync

Mechanism: `ggml_backend_tensor_get_async` queues a GPU→host DMA but does
NOT block. If the host reads the destination buffer before a later
`ggml_backend_synchronize` / `llama_synchronize`, it races the driver's
DMA engine. Stale data at best; driver-side buffer-bookkeeping corruption
at worst (the RADV signature we hit).

All call sites inside `src/llama.cpp`:

| Line | Destination | Why it's safe |
|---|---|---|
| 4196 | `logits_out + i_out*n_vocab` (into `lctx.logits`) | `lctx.logits` is only read via public API `llama_get_logits_ith` etc., which call `llama_synchronize` internally. The user cannot legitimately access the data before that sync. |
| 4201 | `logits_out` | Same: lctx.logits, synced at API boundary. |
| 4204 | `logits_out` | Same. |
| 4233 | `embd_out` (into `lctx.embd`) | Same pattern: accessed via `llama_get_embeddings*`, synced at API boundary. |
| 4250 | `embd_seq_out` | Same. |
| 4286 | `lctx.mtp_logits_buf.data()` (SINGLE-PASS readback) | Followed by `lctx.mtp_logits_extracted.resize` + another async get at 4296, and the caller treats `lctx.mtp_logits_buf` as an API-boundary buffer accessed via `llama_get_mtp_logits*` — those call `llama_synchronize`. |
| 4296 | `lctx.mtp_logits_extracted.data()` | Same: API-boundary buffer, user reads via `llama_get_mtp_draft_token`. |
| 4303 | `iter_major.data()` (CHAINED-ROLLOUT readback) | **WAS THE BUG.** Local stack buffer, read immediately by `memcpy` loop at 4311. Fix: `ggml_backend_synchronize(backend_mtp)` inserted at 4305-4309 before the memcpy. |
| 4513 | `lctx.embd` (encoder path) | Encoder-path output, synced at API boundary by `llama_synchronize`. |
| 4534 | `embd_seq_out` (encoder pooled) | Same encoder-path pattern. |
| 4550 | `embd_out` (encoder non-pooled) | Same. |

**Invariant**: any async get whose destination is a `lctx.*` buffer is
safe because the public API chain syncs. Any async get whose destination
is a **local stack/heap buffer immediately read in the same function** is
the bug class — only 4303 fell into this category. Fixed and now the
ONLY exception in the tree.

**Recommendation for future code**: if you call
`ggml_backend_tensor_get_async` with a destination that is NOT in
`lctx.*` and that you plan to read in the same scope, wrap with:
```cpp
ggml_backend_tensor_get_async(backend, tensor, local_buf, offset, size);
ggml_backend_synchronize(backend);  // REQUIRED before reading local_buf
```

## Class B — `ggml_permute` / `ggml_transpose` without subsequent
materialization when the consumer hardcodes strides

Mechanism: permute/transpose produce non-contiguous views whose logical
layout differs from their physical memory. Consumers that assume
contiguous memory and compute offsets via a hardcoded formula read the
wrong bytes. Consumers that respect tensor `nb[]` strides are safe.

Sites in `src/llama-build-context.cpp` and `src/llama-delta-net.cpp`:

| File | Line | Op sequence | Consumer | Verdict |
|---|---|---|---|---|
| `llama-delta-net.cpp` | 108-110 | permute v/g/beta | `ggml_delta_net_ext` | **Fixed in Stage 3 at `ggml/src/ggml.c:22835+` (slow path). The IQK fast path was innocent — its hardcoded g/beta formulas happen to match post-permute strides, and v uses vnb1/vnb2/vnb3. The slow path (runs when `emit_intermediates=true`, i.e. `n_tokens>1` prompt eval) hardcoded v/g/beta offsets that assumed a different physical layout. Fix: reuse `src2->nb[1..3]` for v, change g/beta formula to match IQK's `batch*n_tok*n_heads + t*n_heads + head`.** |
| `llama-delta-net.cpp` | 389-393 | permute+l2_norm q/k (n_seq_tokens>1) | `build_fused_delta_net` | Safe — `l2_norm` materializes contiguously. |
| `llama-delta-net.cpp` | 395-398 | l2_norm+permute q/k (n_seq_tokens==1) | `build_fused_delta_net` | Permuted view with n_tokens=1. For n_tokens=1, the hardcoded stride formula accidentally yields correct offsets (only one position to index). Technically fragile but currently works. |
| `llama-build-context.cpp` | 538 | permute pos_bias | `ggml_cont(pos_bias)` immediately | Safe — followed by cont. |
| `llama-build-context.cpp` | 633 | `v_cur = ggml_transpose(v_cur)` | `ggml_cpy(v_cur, v_cache_view)` at :637 | Safe — `ggml_cpy` respects source `nb[]` strides for arbitrary layouts. |
| `llama-build-context.cpp` | 1603 | permute `q` | Feeds attention via generic ops | Safe — downstream ops use stride-aware access. |
| `llama-build-context.cpp` | 1689 | `ggml_cont(ggml_transpose(...))` | `ggml_mul_mat` | Safe — wrapped in cont before matmul. |
| `llama-build-context.cpp` | 1737 | permute kqv | `ggml_cont_2d(kqv_merged)` immediately | Safe — followed by cont. |
| `llama-build-context.cpp` | 1782 | permute kqv | `ggml_cont_2d(kqv_merged)` immediately | Safe — followed by cont. |

**Result**: 0 new Class B bugs. The known Class B (CPU DeltaNet garbage)
was initially diagnosed as an IQK fast-path issue; the actual root cause
was in the CPU **slow path** at `ggml/src/ggml.c:22835+`. Fixed in Stage 3
by making slow path stride-aware for v (via `src2->nb[1..3]`) and
realigning its g/beta formulas to the IQK fast-path formulas which
already match the post-permute physical layout. No graph-level
`ggml_cont` needed — all fixes are backend-scoped.

**Recommendation for future code**: when calling `ggml_permute` or
`ggml_transpose`, verify the immediate consumer. If the consumer is a
backend-specific kernel that uses push-constant strides, it's fine. If
the consumer hardcodes a stride formula (rare but possible in IQK fast
paths and some older shaders), wrap with `ggml_cont` explicitly. Treat
any such wrap as suspect on Vulkan — materialization of unusual permute
patterns can trigger edge cases.

## What we didn't audit (out of scope)

- `ggml_view_*` producing non-contiguous sub-views — too numerous,
  consumer-specific. Instead use the DeltaNet failure pattern as a
  "catch via downstream tests" signal.
- Test and example code paths (`tests/**`, `examples/**`) — not
  production-critical and tests run rarely.
- Other `ggml_backend_tensor_*_async` variants like `set_async` — none
  exist in the grep results for this repo.

## Closing status

Class A: 1 bug found + fixed at 4305; 10 other sites verified safe via
API-boundary sync pattern. 0 open Class A issues.

Class B: 1 known issue fixed in Stage 3 at `ggml.c` slow path. 0 new or
open Class B issues.
