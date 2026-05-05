#pragma once

// Per-backend instrumentation accumulator for the CUDA graph cache.
//
// Records (a) per-(topology_key, shape_key) hit counts and last-use
// timestamps; (b) capture/instantiate/update/launch timings; (c)
// cudaMemGetInfo deltas around insert and destroy events; (d)
// cudaErrorGraphExecUpdateFailure occurrences with the shape keys
// involved; (e) `disable_due_to_too_many_updates` trip events. All
// records are written as JSONL.
//
// Activation: GGML_CUDA_GRAPH_PROBE=1 in the environment. When unset,
// active() returns 0 and every record_* call is a no-op (single
// conditional branch).
//
// Output: <dump_dir>/<backend_id>-<probe>.jsonl. dump_dir =
// GGML_CUDA_GRAPH_PROBE_DIR if set, else
// /mnt/archive/cuda-graph-probe/<runid>/.
//
// Flushing: signal-driven (SIGUSR1) + periodic (every
// GGML_CUDA_GRAPH_PROBE_FLUSH_SEC seconds, default 30) + teardown
// (~ggml_backend_cuda_context).

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

struct ggml_backend_cuda_context;

struct cuda_graph_probe_state {
    struct timing_rec {
        uint64_t  ts_ns;
        const char * event;          // string literal, no allocation
        double    duration_us;
        int       n_nodes;
        uint64_t  topology_key;
    };
    struct vram_rec {
        uint64_t  ts_ns;
        const char * event;          // "insert" or "destroy"
        uint64_t  topology_key;
        int64_t   free_before_bytes;
        int64_t   free_after_bytes;
    };
    struct upd_fail_rec {
        uint64_t  ts_ns;
        uint64_t  topology_key;
        uint64_t  shape_key_old;
        uint64_t  shape_key_new;
    };
    struct disable_rec {
        uint64_t  ts_ns;
        uint64_t  topology_key;
        int       consecutive_updates;
    };

    std::mutex mu;
    std::vector<timing_rec>   timings;
    std::vector<vram_rec>     vram_deltas;
    std::vector<upd_fail_rec> update_failures;
    std::vector<disable_rec>  disable_too_many;

    // Cached counters surfaced via the C accessor API.
    std::atomic<uint64_t> update_failure_count{0};
    std::atomic<uint64_t> disable_vram_pressure_count{0};
};

namespace cuda_graph_probe {

// Cached env read; safe before any backend init.
int active();

// Idempotent: lazy-creates the dump directory, picks a runid, installs the
// SIGUSR1 handler, spawns the background flush thread. First active record
// call invokes this.
void ensure_initialized();

// Walks all registered backend contexts and flushes their accumulators.
// Returns 0 on success, -1 if probe inactive.
int flush_all();

// Per-context flush — appends JSONL records to the per-backend file paths.
int flush_context(ggml_backend_cuda_context & ctx);

// Pre-teardown drain — called from ~ggml_backend_cuda_context. Stops the
// background thread on the LAST context's destruction.
void on_context_destroyed(ggml_backend_cuda_context & ctx);

// Recorders. event strings must be string literals (no allocation/copy).
void record_timing(ggml_backend_cuda_context & ctx,
                   uint64_t topology_key, const char * event,
                   double duration_us, int n_nodes);
void record_vram(ggml_backend_cuda_context & ctx,
                 uint64_t topology_key, const char * event,
                 int64_t free_before, int64_t free_after);
void record_update_failure(ggml_backend_cuda_context & ctx,
                           uint64_t topology_key,
                           uint64_t shape_key_old, uint64_t shape_key_new);
void record_disable_too_many(ggml_backend_cuda_context & ctx,
                             uint64_t topology_key, int consecutive_updates);

} // namespace cuda_graph_probe
