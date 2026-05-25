// ggml-mgpu-split.h
//
// Multi-GPU split-config infrastructure shared between LM (llama_model)
// and CLIP (clip_ctx). Phase 46 Path B (yarn-agentic
// PHASE46-MULTIGPU-CLIP-TENSOR-SPLIT.md §12) extracts the LM's
// GRAPH-mode row-chunking primitive into this header so both consumers
// run the same code with the same invariants.
//
// Formal specs (yarn-agentic specs/mgpu-split/):
//   - MgpuSplitConfig.allium       — struct invariants
//   - BuftSetupLoop.tla            — buft-assignment loop
//   - CreateSplitBalance.tla       — this file's algorithm
//   - ClipCrossDeviceFlow.tla      — CLIP topology (extends AsyncReduce)
//   - CrossCodepathConsistency.allium — LM ⇔ CLIP equivalence

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

// ============================================================
// Multi-GPU split-config struct (C++ only; PHASE46 B.2)
// ============================================================
// The shared mgpu_split_config struct. Fields mirror exactly the
// MgpuSplitConfig.allium spec (yarn-agentic specs/mgpu-split/).
// Both llama_model (LM-side) and clip_ctx (CLIP-side) hold a
// pointer to one of these post-Path B.
//
// Invariants are documented inline; ggml_mgpu_split_config_check()
// verifies the load-bearing subset at runtime (release-mode-safe).

#ifdef __cplusplus
#include <vector>
#include <utility>

enum ggml_mgpu_split_mode {
    GGML_MGPU_SPLIT_MODE_NONE  = 0,
    GGML_MGPU_SPLIT_MODE_LAYER = 1,
    GGML_MGPU_SPLIT_MODE_ATTN  = 2,
    GGML_MGPU_SPLIT_MODE_GRAPH = 3,
};

struct ggml_mgpu_split_config {
    // Cardinality. @ListLengthsMatchNDevice in MgpuSplitConfig.allium
    // — every per-device list below has length n_device.
    int                              n_device;

    // Per-device CUDA device IDs (CUDA0 = 0, CUDA1 = 1, ...). For
    // Phase 46 production this is {0, 1} on the TU102 NV2-NVLink pair.
    std::vector<int>                 devices;

    // Per-device capacity bound in bytes. Frozen at init.
    // @CapacityImmutable in spec #1.
    std::vector<size_t>              capacity_per_device;

    // Per-device cumulative usage in bytes. The one mutable field;
    // grows monotonically as create_split places weight rows.
    // @MemUsedNonNegative, @CapacityHonored in spec #1.
    std::vector<size_t>              mem_used_per_device;

    // Normalised split CDF in [0.0, 1.0]. splits[n_device-1] = 1.0.
    // @SplitsMonotonic, @SplitsNormalized, @SplitsImmutable in spec #1.
    std::vector<float>               splits;

    // The shared CUDA split buffer type. Non-null iff
    //   (n_device > 1) AND (split_mode in {ATTN, GRAPH}).
    // @SplitBuftPresentIffGraphLikeAndMultiDevice in spec #1.
    ggml_backend_buffer_type_t       split_buft;

    // Split mode. Frozen at init. @SplitModeImmutable in spec #1.
    ggml_mgpu_split_mode             split_mode;

    // Layer enumeration. @LayerListLengthsMatchNLayer asserts that
    // buft_layer.size() == default_layer_device.size() == n_layer.
    int                              n_layer;
    int                              i_gpu_start;

    // Per-layer buft pair (split, offload). For LAYER/NONE modes
    // .first is null. @NoOrphanGpuLayers in spec #1.
    std::vector<std::pair<ggml_backend_buffer_type_t,
                          ggml_backend_buffer_type_t>>  buft_layer;

    // Per-layer default device. -1 for CPU layers (i < i_gpu_start),
    // in [0, n_device) otherwise. @LayerDeviceInRange in spec #1.
    std::vector<int>                 default_layer_device;
};

// Construct a fresh, fully-initialized ggml_mgpu_split_config. All
// length-n_device vectors are sized to n_device with zeros; the
// length-n_layer vectors are sized with sentinel (-1 / null pairs).
// Caller fills in splits, capacity, split_mode, then runs
// ggml_mgpu_split_config_check() before use.
ggml_mgpu_split_config ggml_mgpu_split_config_make(
        int n_device,
        int n_layer);

// Runtime invariant check. Asserts every static invariant from
// MgpuSplitConfig.allium that's checkable from the struct alone
// (cardinality, monotonicity, normalisation, immutability via
// snapshot comparison is the caller's responsibility for fields
// they've sworn not to mutate).
//
// Returns 0 on success. Non-zero return is a count of failed
// invariants; check_id identifies which one failed first (for
// diagnostics). Designed to be a single `assert()` in callers.
//
// This is the runtime counterpart to the formal-spec
// MgpuSplitConfig.allium — same invariants, two checkpoints
// (Alloy Analyzer at design time, this function at runtime).
int ggml_mgpu_split_config_check(
        const ggml_mgpu_split_config & cfg,
        const char ** out_failed_invariant_name);

#endif // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif

// ggml_mgpu_create_split
//
// Row-chunked split distributor. Divides `nr` (rows) among `n_device`
// devices, biased by:
//   - the CDF in `splits[0..n_device-1]` (cumulative shares in [0, 1],
//     monotonic, ending at 1.0); and
//   - the per-device `mem_used[0..n_device-1]` bytes (in-place balance:
//     a device with higher current usage is allocated proportionally
//     less of `nr`).
//
// On exit, `result[0..n_device-1]` contains the number of rows
// allocated to each device, scaled by `granularity`. Properties
// (verified by yarn-agentic specs/mgpu-split/CreateSplitBalance.tla):
//
//   - sum(result[i] / granularity for i in 0..n_device-1) == nr / granularity
//   - result[i] >= 0 for all i
//   - termination in finite steps regardless of inputs
//
// If `granularity` is negative, returns a uniform [nr, nr, ..., nr]
// of length n_device (the "no chunking" fast path).
//
// Asserts:
//   - nr % granularity == 0 when granularity >= 0
//   - n_device >= 1
//
// `verbose` triggers a LLAMA_LOG_INFO trace of the intermediate
// values, useful when diagnosing allocation imbalance.
GGML_API void ggml_mgpu_create_split(
        int            nr,
        int            granularity,
        size_t         n_device,
        const float  * splits,
        const size_t * mem_used,
        int            verbose,
        int          * result);

// ggml_mgpu_alloc_split_tensors
//
// Allocate `n_device` per-device tensors sharing a logical shape, each
// holding a row-chunk per `split_counts[i]`. The original tensor's
// dimensions and dtype are preserved on dims other than `split_dim`;
// dim `split_dim` is set to `split_counts[i]` for the i-th device's
// tensor.
//
// On exit, `out_tensors[0..n_device-1]` holds the per-device tensors
// (or nullptr where split_counts[i] == 0). Naming follows
// `<tensor->name>.<i>`.
//
// `split_dim` semantics (mirrors llama-load-tensors.cpp:3643):
//   - split_dim < 0  : per-device replication of the full tensor
//                      (each device gets a tensor of identical shape;
//                      "splits" are equal weight per device).
//   - split_dim == 0 : split along the first (row) dimension; each
//                      device's tensor has ne[0] = split_counts[i].
//   - split_dim == 1 : split along the second dimension; each device's
//                      tensor has ne[1] = split_counts[i].
//
// `mem_used[i]` is incremented by ggml_nbytes(out_tensors[i]) for each
// allocated tensor — the caller's running per-device usage tracker is
// updated in place. This matches the LM-side semantics in
// llama-load-tensors.cpp:3686-3692.
//
// Asserts:
//   - split_dim <= 1
//   - n_device >= 2 (single-device callers shouldn't reach this path)
GGML_API void ggml_mgpu_alloc_split_tensors(
        int                  split_dim,
        struct ggml_context * ctx,
        const struct ggml_tensor * tensor,
        size_t               n_device,
        const int          * split_counts,
        struct ggml_tensor ** out_tensors,
        size_t             * mem_used);

#ifdef __cplusplus
}
#endif
