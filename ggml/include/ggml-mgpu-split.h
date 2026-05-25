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
