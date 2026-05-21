#pragma once

#include "ggml.h"

#include <cstdint>

struct ggml_backend_cuda_context;

struct ggml_graph_node_properties {
    void * node_address;
    ggml_op node_op;
    ggml_type node_type;                       // dst tensor dtype
    ggml_type src_type[GGML_MAX_SRC];          // per-src dtype (GGML_TYPE_COUNT = no src)
    int64_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];
    void * src_address[GGML_MAX_SRC];
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
};

struct ggml_cuda_graph {
#ifdef USE_CUDA_GRAPH
    ~ggml_cuda_graph();
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    size_t num_nodes = 0;
    std::vector<cudaGraphNode_t> nodes;
    std::vector<cudaKernelNodeParams> params;
    bool disable_due_to_gpu_arch = false;
    bool disable_due_to_too_many_updates = false;
    bool disable_due_to_failed_graph_capture = false;
    int number_consecutive_updates = 0;
    std::vector<ggml_graph_node_properties> ggml_graph_properties;
    bool use_cpy_indirection = false;
    std::vector<char *> cpy_dest_ptrs;
    char ** dest_ptrs_d;
    int dest_ptrs_size = 0;
    // Index to allow each cpy kernel to be aware of it's position within the graph
    // relative to other cpy nodes.
    int graph_cpynode_index = -1;

    // PHASE_NSTREAM_KV_PERF Tier 2: read-view indirection table.
    // Parallel to cpy_dest_ptrs / dest_ptrs_d, but for the K/V READ
    // views consumed by FA / per-slot-kv kernels. Per-tick the table
    // is populated from each registered read view's current data
    // pointer (via the op_params slot set by
    // ggml_set_read_view_indirect_slot). The consumer kernels read
    // K/V data through this table to sidestep the cuda-graph
    // stale-arg problem for VIEW src addresses.
    //
    // Filled at check_node_graph_compatibility_and_refresh_copy_ops
    // (or its successor) before cudaStreamBeginCapture. Size = max
    // slot index + 1 across all registered views in the current
    // graph.
    //
    // See specs/kv-cache/per_stream_read_view_patching.allium for
    // the ReadViewPatchedByUpdate contract and the design summary in
    // PHASE_NSTREAM_KV_PERF.md.
    bool use_read_view_indirection = false;
    std::vector<void *> read_view_src_ptrs;   // host staging
    void ** read_view_src_ptrs_d = nullptr;   // GPU table
    int read_view_src_ptrs_capacity = 0;      // alloc'd slots

    // Graph cache instrumentation fields. Populated only when the probe
    // is active (GGML_CUDA_GRAPH_PROBE=1); otherwise left at defaults.
    uint64_t topology_key = 0;     // hash(n_nodes, op[i] for i in nodes); skips ne
    uint64_t shape_key    = 0;     // hash including ne[d] per node (current behaviour)
    uint64_t hits_total   = 0;     // # successful lookups landing on this entry
    uint64_t last_use_us  = 0;     // steady_clock micros at last hit
    ggml_backend_cuda_context * owner_ctx = nullptr;  // back-pointer for dtor probe
#endif
};

