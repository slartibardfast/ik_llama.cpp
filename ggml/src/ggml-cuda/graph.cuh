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

    // Graph cache instrumentation fields. Populated only when the probe
    // is active (GGML_CUDA_GRAPH_PROBE=1); otherwise left at defaults.
    uint64_t topology_key = 0;     // hash(n_nodes, op[i] for i in nodes); skips ne
    uint64_t shape_key    = 0;     // hash including ne[d] per node (current behaviour)
    uint64_t hits_total   = 0;     // # successful lookups landing on this entry
    uint64_t last_use_us  = 0;     // steady_clock micros at last hit
    ggml_backend_cuda_context * owner_ctx = nullptr;  // back-pointer for dtor probe
#endif
};

