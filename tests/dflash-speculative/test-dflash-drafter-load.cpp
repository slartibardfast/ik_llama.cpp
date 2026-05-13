// test-dflash-drafter-load.cpp
//
// Smoke test for the dflash-drafter-loader.h. Opens the production
// drafter GGUF, walks all 58 tensors, allocates GPU buffers, frees.
//
// Verifies that the loader can find all expected per-layer + global
// tensors and that the metadata matches the drafter's config.json.
//
// Path defaults to /opt/models/qwen36-27b-dflash/qwen36-27b-dflash-f16.gguf.
// Override with environment variable DFLASH_DRAFTER_GGUF.
//
// Exit codes:
//   0  PASS — loader succeeded, all tensors found, metadata matches
//   1  FAIL — load error or metadata mismatch
//  77  SKIP — drafter GGUF not present at the expected path

#include "dflash-drafter-loader.h"

#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>

namespace {

bool file_exists(const char * path) {
    struct stat st{};
    return stat(path, &st) == 0;
}

int check_metadata(const dflash_reference::DrafterWeights & w) {
    int fails = 0;
    auto check = [&](const char * label, int expected, int actual) {
        if (expected != actual) {
            std::fprintf(stderr, "  [FAIL] %s: expected %d, got %d\n", label, expected, actual);
            ++fails;
        } else {
            std::printf("  [ok] %s = %d\n", label, actual);
        }
    };
    check("n_layers",          5,      w.n_layers);
    check("hidden_size",       5120,   w.hidden_size);
    check("intermediate_size", 17408,  w.intermediate_size);
    check("n_q_heads",         32,     w.n_q_heads);
    check("n_kv_heads",        8,      w.n_kv_heads);
    check("head_dim",          128,    w.head_dim);
    // vocab_size is not in the drafter GGUF — shared with target.
    check("sliding_window",    2048,   w.sliding_window);
    check("block_size",        16,     w.block_size);
    check("mask_token_id",     248070, w.mask_token_id);

    if (w.rope_theta != 10000000.0f) {
        std::fprintf(stderr, "  [WARN] rope_theta = %.1f (expected 10000000.0)\n", w.rope_theta);
    } else {
        std::printf("  [ok] rope_theta = %.1f\n", w.rope_theta);
    }

    if (w.target_layer_ids.size() != 5) {
        std::fprintf(stderr, "  [FAIL] target_layer_ids size = %zu (expected 5)\n",
                     w.target_layer_ids.size()); ++fails;
    } else {
        std::printf("  [ok] target_layer_ids = [%d, %d, %d, %d, %d]\n",
                    w.target_layer_ids[0], w.target_layer_ids[1], w.target_layer_ids[2],
                    w.target_layer_ids[3], w.target_layer_ids[4]);
    }

    if (w.layer_types.size() != 5) {
        std::fprintf(stderr, "  [WARN] layer_types size = %zu (expected 5)\n",
                     w.layer_types.size());
    } else {
        std::printf("  [ok] layer_types = [%d, %d, %d, %d, %d]\n",
                    w.layer_types[0], w.layer_types[1], w.layer_types[2],
                    w.layer_types[3], w.layer_types[4]);
    }

    return fails;
}

int check_tensors_uploaded(const dflash_reference::DrafterWeights & w) {
    int fails = 0;
    for (int l = 0; l < w.n_layers; ++l) {
        if (!w.attn_norm[l] || !w.attn_q[l] || !w.attn_q_norm[l] || !w.attn_k[l] ||
            !w.attn_k_norm[l] || !w.attn_v[l] || !w.attn_output[l] || !w.ffn_norm[l] ||
            !w.ffn_gate[l] || !w.ffn_up[l] || !w.ffn_down[l]) {
            std::fprintf(stderr, "  [FAIL] layer %d missing one or more tensors\n", l);
            ++fails;
        }
    }
    if (!w.dflash_fc)          { std::fprintf(stderr, "  [FAIL] dflash_fc missing\n");          ++fails; }
    if (!w.dflash_hidden_norm) { std::fprintf(stderr, "  [FAIL] dflash_hidden_norm missing\n"); ++fails; }
    if (!w.output_norm)        { std::fprintf(stderr, "  [WARN] output_norm missing (may live in target)\n"); }

    if (fails == 0) {
        std::printf("  [ok] all per-layer + global drafter tensors uploaded\n");
        std::printf("  [ok] %zu GPU buffers allocated\n", w.gpu_buffers.size());
    }
    return fails;
}

} // anonymous namespace

int main() {
    const char * path = std::getenv("DFLASH_DRAFTER_GGUF");
    if (!path) path = "/opt/models/qwen36-27b-dflash/qwen36-27b-dflash-f16.gguf";

    std::printf("=== test-dflash-drafter-load ===\n");
    std::printf("path: %s\n", path);

    if (!file_exists(path)) {
        std::fprintf(stderr, "[SKIP] drafter GGUF not at %s\n", path);
        return 77;
    }

    dflash_reference::DrafterWeights w;
    if (!dflash_reference::load_drafter(path, w)) {
        std::fprintf(stderr, "[FAIL] load_drafter failed\n");
        return 1;
    }

    int fails = 0;
    std::printf("\n--- metadata ---\n");
    fails += check_metadata(w);
    std::printf("\n--- tensor upload ---\n");
    fails += check_tensors_uploaded(w);

    dflash_reference::free_drafter(w);

    if (fails > 0) {
        std::fprintf(stderr, "\n[OVERALL] %d failures\n", fails);
        return 1;
    }
    std::printf("\n[PASS] drafter loader OK\n");
    return 0;
}
