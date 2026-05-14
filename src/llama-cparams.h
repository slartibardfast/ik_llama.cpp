#pragma once

#include "llama-impl.h"

#include <cstdint>

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_threads;       // number of threads to use for generation
    uint32_t n_threads_batch; // number of threads to use for batch processing

    std::vector<std::string> devices;
    std::vector<std::string> devices_draft;

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    float defrag_thold;

    bool embeddings;
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    int  mla_attn;
    int  attn_max_batch;
    bool fused_moe_up_gate;
    bool grouped_expert_routing;
    bool fused_up_gate;
    bool fused_mmad;
    bool rope_cache;
    bool graph_reuse;
    bool k_cache_hadamard;
    bool v_cache_hadamard;
    bool split_mode_graph_scheduling;
    //bool split_mode_f16;
    bool scheduler_async;
    int  min_experts;
    float thresh_experts;
    bool mtp;
    // Phase 36 Step 3 (per-ubatch MTP KV hook): when true, the qwen35
    // verify graph appends a kv-only MTP layer compute that writes MTP
    // KV entries for ALL batch positions in one shot. Speculative.cpp
    // skips the separate MTP_OP_UPDATE_ACCEPTED dispatch and uses
    // llama_kv_cache_seq_rm to trim rejected positions.
    bool mtp_inline_kv_hook;
    // Phase 36 Step 1: number of fused draft steps (0 = not fused).
    // Set by llama_mtp_fused_draft_invoke before each fused decode;
    // read by build_qwen35_mtp_fused / set_inputs.
    int  mtp_fused_n_steps;
    // Phase 38 C: extended-chain count. When > 0, fused runs
    // n_steps + n_extend internal chain steps but emits only
    // n_steps drafts. Populates persist[n_steps..n_steps+n_extend-1]
    // with the extended residuals (used as seed for the all-accept
    // case in Phase 38 E speculative dispatch). 0 = no extension
    // (default; matches pre-Phase-38 behavior).
    int  mtp_fused_n_extend;
    int  worst_graph_tokens;

    // DFlash extract-features hook. Capture residual-stream snapshots at
    // the K source-layer indices for the DFlash drafter. When
    // dflash_extract_count > 0, the qwen35 build graph dups the per-layer
    // residual at each requested index, marks it ggml_set_output, and
    // stashes the handle on default_decoder.t_dflash_extract[idx].
    int      dflash_extract_count = 0;
    // Cap is 80 to cover full-stack capture on the largest production target
    // (Qwen 3.6 27B has 65 layers). Mirrored as the array size on
    // default_decoder.dflash_extract_buf/_n in llama-decoder-internal.h and
    // as the upper bound in llama_set_dflash_extract_layers in llama.cpp.
    int32_t  dflash_extract_layers[80] = {};

    enum ggml_type reduce_type;
    enum llama_pooling_type pooling_type;
    enum llama_mtp_op_type mtp_op_type;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    void * cuda_params;
};
