# vLLM upstream PR drafts — sm_75 / fp16 / DFlash compatibility

Three small, well-localized fixes against vLLM PR #40898 (the DFlash
speculative decoder, vllm commit `23002d3f3`) that allow it to run on
sm_75 (Turing) hardware via the fp16 fallback path. None changes
decoding semantics; each fixes a bug surfaced only when the BF16
assumption is broken.

Maintained locally as runtime monkey-patches in
`scripts/vllm_sm75_patches.py`. When upstream lands the fix, delete
the corresponding patch from that module.

---

## PR 1 — `combine_hidden_states`: cast hidden_states to fc weight dtype

**File**: `vllm/model_executor/models/qwen3_dflash.py`
**Line**: 656–668 (the `combine_hidden_states` method)

**Problem**

On hardware without BF16 (compute capability < 8.0), vLLM is invoked
with `dtype="float16"`. The drafter's `fc.weight` is cast to fp16 at
load. But `target_hidden_states` arrives from EagleProposer's hidden-
state collection path as fp32 — the concat across selected target
layers upcasts. The matmul fails:

```
RuntimeError: expected mat1 and mat2 to have the same dtype,
              but got: float != c10::Half
```

BF16 hides this on Hopper/Ampere because some torch ops auto-promote
to fp32 more silently with BF16 inputs. fp16 is stricter.

**Fix**

```python
def combine_hidden_states(self, hidden_states):
    if not self.model.use_aux_hidden_state:
        return hidden_states
    needs_squeeze = hidden_states.dim() == 1
    if needs_squeeze:
        hidden_states = hidden_states.unsqueeze(0)
+   # Cast input to fc weight dtype. EagleProposer's collection path
+   # can upcast hidden states to fp32; without this, fp16 weights
+   # cause matmul dtype mismatch on hardware without BF16.
+   fc_dtype = self.model.fc.weight.dtype
+   if hidden_states.dtype != fc_dtype:
+       hidden_states = hidden_states.to(dtype=fc_dtype)
    result = self.model.fc(hidden_states)
    if needs_squeeze:
        result = result.squeeze(0)
    return result
```

**Test plan**: invoke vllm DFlash with `dtype="float16"` on any sm <80
device. Without the fix: dtype-mismatch crash at first inference.
With the fix: clean.

---

## PR 2 — FlexAttention: use `.reshape()` instead of `.view()` on KV cache

**File**: `vllm/v1/attention/backends/flex_attention.py`
**Line**: 1115–1116 (DECODER branch in `FlexAttentionImpl.forward`)

**Problem**

```
key_cache = key_cache.view(-1, self.num_kv_heads, self.head_size)
value_cache = value_cache.view(-1, self.num_kv_heads, self.head_size)
```

The `kv_cache.unbind(0)` result is not always contiguous in paged-KV
layouts (specifically when seen via the DFlash spec-decoder's slot-
mapping path). `.view()` requires contiguous source:

```
RuntimeError: view size is not compatible with input tensor's size
              and stride (at least one dimension spans across two
              contiguous subspaces). Use .reshape(...) instead.
```

**Fix**

```python
-key_cache = key_cache.view(-1, self.num_kv_heads, self.head_size)
-value_cache = value_cache.view(-1, self.num_kv_heads, self.head_size)
+key_cache = key_cache.reshape(-1, self.num_kv_heads, self.head_size)
+value_cache = value_cache.reshape(-1, self.num_kv_heads, self.head_size)
```

`.reshape()` produces identical output; only allows the copy fallback
when contiguity is broken. Free win on already-contiguous input
(reshape returns a view in that case).

**Test plan**: any FlexAttention-backed DECODER attention call with
a non-contiguous KV cache slice. DFlash + sm_75 hits this; possibly
others.

---

## PR 3 — FlexAttention: round `BLOCK_M`/`BLOCK_N` to power-of-2

**File**: `vllm/v1/attention/backends/flex_attention.py`
**Function**: `get_kernel_options` (around line 1152)

**Problem**

Triton's `tl.arange(0, N)` requires N to be a compile-time power of 2.
When `attn_metadata.direct_build=True`, `get_kernel_options` returns
`BLOCK_N = block_n` verbatim from the caller's `attn_metadata.
block_mask.BLOCK_SIZE`. With DFlash this can be non-POW2 (e.g. 17 from
`1 + num_speculative_tokens = 16` + boundary, or KV-block-size
variants). The kernel then fails to compile:

```
offs_n = kv_start + tl.arange(0, BLOCK_N)
arange's range must be a power of 2
```

**Fix**

```python
+def _next_pow2(n: int) -> int:
+    if n <= 1: return 1
+    return 1 << (n - 1).bit_length()

 def get_kernel_options(query, block_m, block_n, use_direct_build):
     ...
     if use_direct_build:
-        kernel_options["BLOCK_M"] = block_m
-        kernel_options["BLOCK_N"] = block_n
+        # Triton tl.arange requires POW2 N; round up to next POW2.
+        # Larger block is masked internally by the kernel.
+        kernel_options["BLOCK_M"] = _next_pow2(block_m)
+        kernel_options["BLOCK_N"] = _next_pow2(block_n)
         return kernel_options
```

**Test plan**: any DFlash invocation with non-POW2 effective block
size. With fix: kernel compiles. Without: Triton compile error.

---

## Submission notes

All three fixes are:
- ≤ 10 lines each
- Pure additions/replacements; no architectural changes
- Decode-semantics-preserving (no value drift)
- Helpful to anyone running DFlash on sm < 80 hardware (RTX 4090,
  RTX 6000, A10, T4, etc.)

If upstream review prefers a single PR, merge them under "DFlash:
sm_75 / fp16 compatibility (3 fixes)".

Local maintenance: `scripts/vllm_sm75_patches.py` applies all three
as runtime monkey-patches; remove the relevant `_patch_*()` function
from that module once each PR lands and a new vllm release ships.
