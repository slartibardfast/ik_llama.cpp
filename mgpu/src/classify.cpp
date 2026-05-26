// libmgpu — weight-name → split-kind classifier
//
// See mgpu/include/mgpu/classify.h for the contract.
#include "mgpu/classify.h"

#include <cstring>

extern "C" enum mgpu_split_kind mgpu_classify_weight(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return MGPU_SPLIT_NONE;
    }

    // Column-parallel: input-side matmuls. The reduction axis (dim 0)
    // stays full on each device; output features (dim 1) are split.
    // Each device's matmul output is a per-device feature slice that
    // flows directly into the same device's row-parallel matmul without
    // a reduce in between.
    //
    // Order matters: more-specific names checked first (avoid e.g. a
    // misclassification of "attn_qkv" because "attn_q" is its prefix).
    if (strstr(name, "attn_qkv") != nullptr)  return MGPU_SPLIT_COL_PAR;
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
