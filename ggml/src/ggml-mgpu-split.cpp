// ggml-mgpu-split.cpp
//
// Phase 46 Path B (yarn-agentic PHASE46 §12.4 B.1) — implementation
// of the multi-GPU split-config primitives shared between LM and CLIP.
// Behaviour ported byte-equivalent from the LM-local original at
// llama-load-tensors.cpp:351-414 (create_split) and :3643-3693
// (prepare_split_tensors).
//
// Formal contract: specs/mgpu-split/CreateSplitBalance.tla. Termination
// and balance properties machine-verified under TLC at 2026-05-25.

#include "ggml-mgpu-split.h"

#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

// Match the LM's LLAMA_LOG_INFO sink for diagnostic parity.
// When verbose > 0, prints to stderr just like the LM-local original.
static void mgpu_log(int verbose, const char * fmt, ...) {
    if (!verbose) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

extern "C" {

void ggml_mgpu_create_split(
        int            nr,
        int            granularity,
        size_t         n_device,
        const float  * splits,
        const size_t * mem_used,
        int            verbose,
        int          * result) {
    GGML_ASSERT(n_device >= 1);
    GGML_ASSERT(result != nullptr);
    GGML_ASSERT(splits != nullptr);
    GGML_ASSERT(mem_used != nullptr);

    // No-chunking fast path: granularity < 0 returns uniform [nr, ...].
    // Mirrors llama-load-tensors.cpp:355.
    if (granularity < 0) {
        for (size_t i = 0; i < n_device; ++i) {
            result[i] = nr;
        }
        return;
    }

    GGML_ASSERT(nr % granularity == 0);

    // tot_memory_used starts at 1 (not 0) to keep the float division at
    // line 367 defined when every device has zero usage. Preserves
    // bytewise equivalence with the LM-local original.
    size_t tot_memory_used = 1;
    for (size_t i = 0; i < n_device; ++i) {
        tot_memory_used += mem_used[i];
    }

    int nchunk = nr / granularity;
    mgpu_log(verbose, "--- ggml_mgpu_create_split: %d chunks\n", nchunk);

    // Phase 1: initial allocation. For each device, compute a desired
    // share and round to int. The bias term
    //   (p - mem_used[i] / tot_memory_used)
    // shifts allocation toward less-loaded devices.
    int sum = 0;
    float last_split = 0.0f;
    for (size_t i = 0; i < n_device; ++i) {
        float p  = splits[i] - last_split;
        float p0 = p;
        p += (p - 1.0f * mem_used[i] / tot_memory_used);
        result[i] = (int) roundf(p * nchunk);
        if (result[i] < 0) result[i] = 0;
        mgpu_log(verbose, "i = %zu, p0 = %g, p = %g, result = %d\n",
                 i, (double) p0, (double) p, result[i]);
        sum += result[i];
        last_split = splits[i];
    }

    // Phase 2: down-correction. While sum > nchunk, find the device
    // with the largest positive over-allocation error AND result > 0,
    // decrement. Verified by CreateSplitBalance.tla DownTerminates.
    while (sum > nchunk) {
        last_split = 0.0f;
        float best_err = -INFINITY;
        int   ibest    = -1;
        for (size_t i = 0; i < n_device; ++i) {
            if (result[i] > 0) {
                float p = splits[i] - last_split;
                p += (p - 1.0f * mem_used[i] / tot_memory_used);
                float n_want = p * nchunk;
                float err = result[i] - n_want;
                if (err > best_err) {
                    best_err = err;
                    ibest    = (int) i;
                }
            }
            last_split = splits[i];
        }
        GGML_ASSERT(ibest >= 0 && result[ibest] > 0);
        --result[ibest];
        --sum;
    }

    // Phase 3: up-correction. While sum < nchunk, find the device with
    // the largest positive under-allocation error, increment. Verified
    // by CreateSplitBalance.tla UpTerminates.
    while (sum < nchunk) {
        last_split = 0.0f;
        float best_err = -INFINITY;
        int   ibest    = -1;
        for (size_t i = 0; i < n_device; ++i) {
            float p = splits[i] - last_split;
            p += (p - 1.0f * mem_used[i] / tot_memory_used);
            float n_want = p * nchunk;
            float err = n_want - result[i];
            if (err > best_err) {
                best_err = err;
                ibest    = (int) i;
            }
            last_split = splits[i];
        }
        GGML_ASSERT(ibest >= 0);
        ++result[ibest];
        ++sum;
    }

    // Phase 4: scale by granularity. Mirrors llama-load-tensors.cpp:412.
    for (size_t i = 0; i < n_device; ++i) {
        result[i] *= granularity;
    }
}

void ggml_mgpu_alloc_split_tensors(
        int                  split_dim,
        struct ggml_context * ctx,
        const struct ggml_tensor * tensor,
        size_t               n_device,
        const int          * split_counts,
        struct ggml_tensor ** out_tensors,
        size_t             * mem_used) {
    GGML_ASSERT(split_dim <= 1);
    GGML_ASSERT(n_device >= 2);
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(split_counts != nullptr);
    GGML_ASSERT(out_tensors != nullptr);
    GGML_ASSERT(mem_used != nullptr);

    const std::string name{tensor->name};

    for (size_t i = 0; i < n_device; ++i) {
        if (split_counts[i] <= 0) {
            out_tensors[i] = nullptr;
            continue;
        }

        int64_t ne0 = tensor->ne[0];
        int64_t ne1 = tensor->ne[1];
        int64_t ne2 = tensor->ne[2];

        if (split_dim == 0) {
            ne0 = split_counts[i];
        } else if (split_dim == 1) {
            ne1 = split_counts[i];
        }
        // split_dim < 0: ne0/ne1/ne2 unchanged (per-device replication).

        out_tensors[i] = ggml_new_tensor_3d(ctx, tensor->type, ne0, ne1, ne2);
        auto name_i = name + '.' + std::to_string(i);
        ggml_set_name(out_tensors[i], name_i.c_str());

        mem_used[i] += ggml_nbytes(out_tensors[i]);
    }
}

} // extern "C"

// ============================================================
// ggml_mgpu_split_config helpers (PHASE46 B.2; C++ linkage)
// ============================================================

ggml_mgpu_split_config ggml_mgpu_split_config_make(int n_device, int n_layer) {
    GGML_ASSERT(n_device >= 1);
    GGML_ASSERT(n_layer  >= 0);

    ggml_mgpu_split_config cfg;
    cfg.n_device = n_device;
    cfg.devices.assign(n_device, 0);
    cfg.capacity_per_device.assign(n_device, 0);
    cfg.mem_used_per_device.assign(n_device, 0);
    cfg.splits.assign(n_device, 0.0f);
    cfg.split_buft = nullptr;
    cfg.split_mode = GGML_MGPU_SPLIT_MODE_NONE;
    cfg.n_layer = n_layer;
    cfg.i_gpu_start = 0;
    cfg.buft_layer.assign(n_layer, {nullptr, nullptr});
    cfg.default_layer_device.assign(n_layer, -1);
    return cfg;
}

int ggml_mgpu_split_config_check(const ggml_mgpu_split_config & cfg,
                                  const char ** out_failed_invariant_name) {
    int failures = 0;
    auto fail = [&](const char * name) {
        if (failures == 0 && out_failed_invariant_name) {
            *out_failed_invariant_name = name;
        }
        ++failures;
    };

    // @DevicesNonEmpty
    if (cfg.n_device < 1) fail("DevicesNonEmpty");

    // @ListLengthsMatchNDevice
    if ((int) cfg.devices.size()              != cfg.n_device ||
        (int) cfg.capacity_per_device.size()  != cfg.n_device ||
        (int) cfg.mem_used_per_device.size()  != cfg.n_device ||
        (int) cfg.splits.size()               != cfg.n_device) {
        fail("ListLengthsMatchNDevice");
    }

    // @LayerListLengthsMatchNLayer
    if ((int) cfg.buft_layer.size()           != cfg.n_layer ||
        (int) cfg.default_layer_device.size() != cfg.n_layer) {
        fail("LayerListLengthsMatchNLayer");
    }

    // @SplitsMonotonic
    for (int i = 0; i + 1 < cfg.n_device; ++i) {
        if (cfg.splits[i] > cfg.splits[i + 1]) {
            fail("SplitsMonotonic");
            break;
        }
    }

    // @SplitsNormalized — splits ends at 1.0, all values in [0, 1].
    // Use a small epsilon for the upper-bound check because float
    // parsing of "1,1" may produce 0.99999... at the last entry.
    if (cfg.n_device >= 1) {
        const float eps = 1e-5f;
        if (cfg.splits[cfg.n_device - 1] < 1.0f - eps ||
            cfg.splits[cfg.n_device - 1] > 1.0f + eps) {
            fail("SplitsNormalized");
        }
        for (int i = 0; i < cfg.n_device; ++i) {
            if (cfg.splits[i] < -eps || cfg.splits[i] > 1.0f + eps) {
                fail("SplitsNormalized");
                break;
            }
        }
    }

    // @SplitBuftPresentIffGraphLikeAndMultiDevice
    bool graph_like_multi = (cfg.n_device > 1) &&
                            (cfg.split_mode == GGML_MGPU_SPLIT_MODE_ATTN ||
                             cfg.split_mode == GGML_MGPU_SPLIT_MODE_GRAPH);
    if (graph_like_multi != (cfg.split_buft != nullptr)) {
        fail("SplitBuftPresentIffGraphLikeAndMultiDevice");
    }

    // @LayerDeviceInRange
    for (int i = 0; i < cfg.n_layer; ++i) {
        int d = cfg.default_layer_device[i];
        if (d != -1 && (d < 0 || d >= cfg.n_device)) {
            fail("LayerDeviceInRange");
            break;
        }
        if (i < cfg.i_gpu_start && d != -1) {
            fail("LayerDeviceInRange");
            break;
        }
        if (i >= cfg.i_gpu_start && d == -1) {
            fail("LayerDeviceInRange");
            break;
        }
    }

    // @CapacityHonored — mem_used <= capacity. (Non-strict so this
    // check passes at construction time when both are zero.)
    for (int i = 0; i < cfg.n_device; ++i) {
        if (cfg.mem_used_per_device[i] > cfg.capacity_per_device[i] &&
            cfg.capacity_per_device[i] > 0) {
            fail("CapacityHonored");
            break;
        }
    }

    // @NoOrphanGpuLayers
    for (int i = cfg.i_gpu_start; i < cfg.n_layer; ++i) {
        if (cfg.buft_layer[i].second == nullptr) {
            fail("NoOrphanGpuLayers");
            break;
        }
        if (graph_like_multi && cfg.buft_layer[i].first != cfg.split_buft) {
            fail("NoOrphanGpuLayers");
            break;
        }
    }

    return failures;
}
