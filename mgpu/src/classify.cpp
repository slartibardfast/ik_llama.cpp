// libmgpu — weight-name → split-kind classifier
//
// See mgpu/include/mgpu/classify.h for the contract.
#include "mgpu/classify.h"

#include <cstring>

extern "C" enum mgpu_split_kind mgpu_classify_weight(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return MGPU_SPLIT_NONE;
    }

    // attn_qkv (fused Q|K|V along dim 1) — REPLICATE, NOT col-parallel.
    // A naive col-parallel split of a fused [Q|K|V] weight along dim 1
    // chunks the concat'd Q/K/V output features uniformly, which CROSSES
    // the Q→K and K→V boundaries instead of preserving them per device.
    // E.g. with 2 devices and Q|K|V each n_embd wide, device 0 gets
    // [full Q | first half K], device 1 gets [second half K | full V].
    // No per-device head-slicing works on that layout. REPLICATE keeps
    // the full fused weight on every device; the consumer
    // (mgpu_build_attn_megatron_fused_qkv) does a full per-device QKV
    // matmul, then view-3d slices Q/K/V at standard offsets, then
    // view-slices a head-range for THIS device, then attention math
    // runs head-partitioned. The duplicated QKV matmul compute is small
    // relative to the wo + FFN parallelism gain.
    if (strstr(name, "attn_qkv") != nullptr)  return MGPU_SPLIT_REPLICATE;

    // Column-parallel: input-side SEPARATE-Q/K/V matmuls. The reduction
    // axis (dim 0) stays full on each device; output features (dim 1)
    // are split. Each device produces an output-feature slice that
    // flows directly into the same device's row-parallel matmul
    // without a reduce in between.
    //
    // Order matters: more-specific names checked first.
    if (strstr(name, "attn_q.")  != nullptr)  return MGPU_SPLIT_COL_PAR;
    if (strstr(name, "attn_k.")  != nullptr)  return MGPU_SPLIT_COL_PAR;
    if (strstr(name, "attn_v.")  != nullptr)  return MGPU_SPLIT_COL_PAR;
    if (strstr(name, "ffn_up_gate") != nullptr) return MGPU_SPLIT_COL_PAR;
    if (strstr(name, "ffn_up.")  != nullptr)  return MGPU_SPLIT_COL_PAR;
    if (strstr(name, "ffn_gate.")!= nullptr)  return MGPU_SPLIT_COL_PAR;

    // Row-parallel: output-side matmuls. Reduction axis (dim 0) split;
    // output features (dim 1) full. Each device produces a partial
    // full-shape output; ggml_reduce sums across devices.
    if (strstr(name, "attn_out.")!= nullptr)  return MGPU_SPLIT_ROW_PAR;
    if (strstr(name, "attn_o.")  != nullptr)  return MGPU_SPLIT_ROW_PAR;
    if (strstr(name, "ffn_down.")!= nullptr)  return MGPU_SPLIT_ROW_PAR;

    // Replicate: tensors that flow into non-matmul ops (UPSCALE,
    // RESHAPE+PERMUTE chains, NORM, etc.) or that must be locally
    // addressable on every device. Position embeddings in vision
    // transformers (qwen3vl etc.) go through UPSCALE; norms apply
    // per-element. All small enough that per-device duplication is
    // cheap.
    if (strstr(name, "position_embd") != nullptr) return MGPU_SPLIT_REPLICATE;
    if (strstr(name, "_norm")         != nullptr) return MGPU_SPLIT_REPLICATE;
    if (strstr(name, ".bias")         != nullptr) return MGPU_SPLIT_REPLICATE;
    if (strstr(name, "patch_bias")    != nullptr) return MGPU_SPLIT_REPLICATE;
    if (strstr(name, "patch_embd")    != nullptr) return MGPU_SPLIT_REPLICATE;

    // Default: leave unsplit. Safer than guessing — a missed
    // classification of a matmul weight that should have been COL_PAR
    // or ROW_PAR just leaves it single-device (small residency cost)
    // rather than producing wrong-shape matmuls (IMA).
    return MGPU_SPLIT_NONE;
}
