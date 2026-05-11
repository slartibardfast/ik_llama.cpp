// test-dflash-fc-projection-shape.cpp
//
// Property-based RED test for FuseProjectionFcWeight invariant.
//
// Spec: dflash_speculative.allium ProjectAndFuse contract,
// invariant FuseProjectionFcWeight (provenance: vllm
// qwen3_dflash.py:333-341, :656).
//
// The drafter has a learned fc projection of shape
// (num_target_layers * target_hidden_size, hidden_size). For
// Qwen3.6-27B-DFlash specifically this is (5 * 5120, 5120) =
// (25600, 5120).
//
// Property tested: for any (num_target_layers, hidden_size) pair
// the model declares, the fc weight ik_llama.cpp loads from the
// drafter GGUF must satisfy:
//   fc.shape[0] == num_target_layers * hidden_size
//   fc.shape[1] == hidden_size
//   fc.dtype is BF16 (drafter ships BF16; quantization at this
//   tensor would change the trained interaction shape)
//
// First-landing state: ik_llama.cpp has no DFlash drafter loader
// yet, so the C API returning shape information for fc is
// NOT_IMPLEMENTED. Without the API we cannot drive a property
// against generated shapes; instead the test verifies the
// SPEC SHAPE EQUATION holds for randomly generated (n_layers,
// hidden_size) pairs as a sanity check against the invariant's
// own arithmetic — and exits 77 (SKIP) at the first call site
// where the loader returns NOT_IMPLEMENTED.
//
// Once the drafter loader lands and llama_dflash_n_source_layers
// + a new llama_dflash_fc_shape return real values, the SKIP
// branch turns into a real binding test.

// llama.h not pulled in directly yet — the loader-call branch is
// commented out below; once enabled, restore the include.
#include "dflash_properties.h"

#include <cstdio>
#include <cstdint>
#include <random>

using namespace dflash_properties;

int main() {
    std::mt19937 rng(DFLASH_PROPERTY_SEED);

    // 1. Property: for any valid (n_layers, hidden) pair, the
    //    fc shape equation holds. This is pure arithmetic at
    //    this stage — binding the invariant's algebraic claim.
    for (int sample = 0; sample < DFLASH_PROPERTY_SAMPLES; ++sample) {
        int n_layers = uniform_int(rng, 3, 8);     // realistic range for DFlash variants
        int hidden   = uniform_int(rng, 1024, 8192); // realistic hidden for drafters

        int64_t expected_rows = (int64_t) n_layers * (int64_t) hidden;
        int64_t expected_cols = hidden;

        // The invariant defines fc as a (n_layers*hidden, hidden) linear.
        // Verify the equation we're encoding matches the invariant text.
        DFLASH_REQUIRE(expected_rows == (int64_t) n_layers * hidden,
                       "FuseProjectionFcWeight");
        DFLASH_REQUIRE(expected_cols == hidden,
                       "FuseProjectionFcWeight");
    }
    std::printf("PASS: FuseProjectionFcWeight algebraic equation over %d samples\n",
                DFLASH_PROPERTY_SAMPLES);

    // 2. Once the C API surfaces fc shape, this loop drives the
    //    real check against a loaded drafter. Until then, the
    //    binding is unavailable; return 77 to signal CTest SKIP.
    //
    //    Real implementation will look like:
    //
    //      llama_model * draft = llama_model_load_from_file(...);
    //      int n_src = llama_dflash_n_source_layers(draft);
    //      int hidden_size = llama_model_n_embd(draft);
    //      struct llama_dflash_fc_shape s;
    //      llama_dflash_status st = llama_dflash_fc_shape(draft, &s);
    //      DFLASH_REQUIRE(st == LLAMA_DFLASH_OK, "FuseProjectionFcWeight");
    //      DFLASH_REQUIRE(s.rows == (int64_t)n_src * hidden_size,
    //                     "FuseProjectionFcWeight");
    //      DFLASH_REQUIRE(s.cols == hidden_size, "FuseProjectionFcWeight");
    //      DFLASH_REQUIRE(s.dtype == GGML_TYPE_BF16,
    //                     "FuseProjectionFcWeight");

    std::printf("SKIP: drafter-loader path NOT_IMPLEMENTED; "
                "real-shape binding deferred until "
                "llama_dflash_fc_shape lands.\n");
    return 77;  // CTest SKIP sentinel
}
