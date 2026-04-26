# mesa-repro — standalone Vulkan harness for ACO/RADV spec-constant f32 nondeterminism on RDNA2

## The bug being targeted

On AMD Radeon RX 6800 XT (RADV NAVI21, RDNA2, wave32), the `ggml-vulkan` `mul_mat_vecq` shader produces different column-0 f32 output when compiled with `NUM_COLS=1` vs `NUM_COLS=2` specialization constants. The math is identical at the GLSL level; only the ACO scheduler/register allocator reshapes column-0 codegen when additional columns raise register pressure, which changes the f32 rounding order across the long fma/integer-dot chain.

The divergence is confirmed by `tests/test-backend-ops.cpp`'s `BI_MUL_MAT` suite at `q8_0`/`q4_K`, `K=2048` on NAVI21. It is absent on VEGA (wave64, same driver family) and on CPU backends. The production workaround is `GGML_VK_DISABLE_MMVQ=1`, which routes through the F32 dequantize path; see `MEMORY.md` entry `project_mmvq_batch_invariance.md` for full context and impact on MTP-IR.

This directory holds a minimal standalone Vulkan harness intended to be the input for an upstream MESA issue at https://gitlab.freedesktop.org/mesa/mesa/-/issues.

## Hardware tested

- GPU: AMD Radeon RX 6800 XT (RADV NAVI21, RDNA2, wave32)
- Driver: Mesa RADV 26.0.99
- OS: Linux

## What's here

- `repro.cpp` — raw libvulkan harness (~450 lines, boilerplate-heavy). Loads `shader.spv`, creates three compute pipelines from the same SPIR-V with `NUM_COLS` specialization constants {1, 2, 4}, dispatches each against the same deterministic int8 weight / Q8_1-like activation buffers, and compares column-0 output bit-for-bit. Exits 0 if all byte-identical, 1 on any divergence.
- `shader.comp` — minimal Q8_0/Q8_1-shaped compute shader with:
  - `layout(constant_id = 0) const uint NUM_COLS = 1;`
  - 128-thread workgroup, wave32 enforced via `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT` + required size 32
  - `dotPacked4x8AccSatEXT` packed int8 MACs
  - `float16_t` per-block scales
  - hoisted b-side cache shared across columns (mimics ggml `cache_b_qs`)
  - `temp[NUM_COLS][NUM_ROWS]` 2D accumulator with `NUM_ROWS=2`
  - runtime push-constant `rt_num_rows` to keep the inner loop non-constant
- `CMakeLists.txt` — finds `Vulkan` + `glslc`, compiles `shader.comp` → `shader.spv` at build time, links `repro` to libvulkan only.

## Build

```
cmake -B build
cmake --build build
```

## Run

```
./build/repro build/shader.spv
```

## Current status — does it reproduce?

On the target machine (RX 6800 XT, Mesa 26.0.99) **the minimal shader shipped here does NOT yet reproduce the divergence**:

```
device[0]: AMD Radeon RX 6800 XT (RADV NAVI21) (driver 0x6800063, vendor 0x1002)
subgroup size range: [32..64], required stages mask=0xf0
--- pipeline A: NUM_COLS=1
  y[0] = 12520.4922  (bits=0x4643a1f8)
  y[1] = -144538.719  (bits=0xc80d26ae)
--- pipeline B: NUM_COLS=2
  y[0] = 12520.4922  (bits=0x4643a1f8)
  y[1] = -144538.719  (bits=0xc80d26ae)
  ...
--- pipeline C: NUM_COLS=4
  y[0] = 12520.4922  (bits=0x4643a1f8)
  y[1] = -144538.719  (bits=0xc80d26ae)
  ...
col0 row 0: 1=0x4643a1f8  2=0x4643a1f8 (ulp +0)  4=0x4643a1f8 (ulp +0)
col0 row 1: 1=0xc80d26ae  2=0xc80d26ae (ulp +0)  4=0xc80d26ae (ulp +0)
--- result: column 0 BYTE-IDENTICAL across NUM_COLS={1,2,4}
```

`RADV_DEBUG=shaders` NIR disassembly confirms the column-0 fma/sdot chain is structurally identical between the three variants — ACO does not reshape column 0 when it has nothing else to schedule around it. The full `test-backend-ops BI_MUL_MAT` suite at `q8_0` K=2048 DOES reproduce on this hardware, so the bug is present. The shape our minimal shader is missing:

- ggml's real `mul_mat_vecq` uses struct-based SSBO layout (`block_q8_1_x4`) with 4-deep nested member access (`.ds[idx].xyzw`, `.qs[a*8+b]`) — the per-field address math interacts with column indexing in a way ACO cannot constant-fold away.
- The `repack()` dequant step for `q4_K`/`q4_0` (`vui & 0x0F0F0F0F`, `(vui >> 4) & 0x0F0F0F0F`) adds per-column integer ALU work that changes ACO's scheduling. Our Q8_0-only variant skips this.
- Real MMVQ has a multi-level manual outer unroll (4 → 2 → 1) driven by runtime `num_iters`, whereas our outer loop is fully `[[unroll]]`-expanded.
- The actual test case likely also exercises `NUM_ROWS > 2` multi-row outputs where the row index is runtime-bound, which our harness clamps to `NUM_ROWS=2`.

## Next iteration

The infrastructure (Vulkan init, spec-constant wiring, wave32 pin, deterministic data, bit-exact compare) is correct and ready to be re-used. The next cut at this should either:

1. Pull in the actual `mul_mat_vecq.comp` + `mul_mat_vecq_funcs.glsl` + `mul_mat_vec_base.glsl` + `types.glsl` + `mul_mat_vec_iface.glsl` from `ggml/src/ggml-vulkan/vulkan-shaders/` and swap the inline shader for that, running against a hand-built `block_q8_1_x4` buffer. This is heavier (~4 files, struct-shaped SSBOs, more bindings) but is known to reproduce.
2. Or reach into the runtime: compile the `ggml-vulkan` MMVQ pipelines via `ggml-vulkan` itself and dispatch them with hand-crafted inputs. That's what the existing `tests/test-vulkan-batch-invariance.cpp` in the polaris fork does.

Either way, once a minimal SPIR-V + descriptor layout reproduces, the artefacts go into the MESA issue alongside:

- `RADV_DEBUG=shaders` NIR/ACO dumps for both pipelines
- `RADV_PERFTEST=aco` confirmation
- The `shader.spv` blobs
- Mesa + kernel version triple
