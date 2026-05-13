# DFlash speculative decoding CUDA kernels

Built when `-DGGML_CUDA_DFLASH=ON` (default: OFF). Hard-gated at configure
time to `CMAKE_CUDA_ARCHITECTURES=75` only.

**Contract**: `specs/dflash/kernel-design.md` § 6 (kernel specifications),
§ 7 (Allium ↔ kernel binding table), § 8 (determinism guarantees).
Behavioural invariants: `specs/dflash/dflash.allium`.

Kernels (added incrementally; see `specs/dflash/kernel-design.md` § 10):

- `dflash-combine-features.cu` — anchor-level FC + hidden_norm (§ 6.6)
- `dflash-inject-kv.cu` — per-layer K_proj + V_proj + K_norm + RoPE + cache write (§ 6.2)
- `dflash-drafter-forward.cu` — persistent drafter mega-kernel (§ 6.1)
- `dflash-verify-attn.cu` — verify-shape attention (§ 6.3)
- `dflash-state-checkpoint.cu` / `dflash-state-restore.cu` — DeltaNet state ping-pong (§ 6.4)
- `dflash-argmax-match.cu` — per-slot accept-prefix decision (§ 6.5)

Determinism preconditions (§ 8): no `atomicAdd<float>`; compile-time tile dims;
one CTA per output tile; warp-shuffle inner reductions; SMEM-tree block reductions;
fixed block-idx → tile mapping; fixed WMMA fragment shape; fixed split-SIZE.
