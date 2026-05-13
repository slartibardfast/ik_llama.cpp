// test-llama-bench-spec-init.cpp
//
// Smoke test for the spec init paths used by llama-bench
// (examples/llama-bench/llama-bench.cpp). Asserts that
// common_speculative_init() succeeds for each of {none, mtp, dflash}
// given the production target (and drafter for dflash).
//
// Closure binding #7 for PHASE_DFLASH.md T8 Phase 1 — exercises the
// init paths bench uses so regressions in spec-init wiring show up as
// a fast test rather than a multi-minute closure run.
//
// Skips (exit 77) if TARGET model isn't available; dflash leg skips
// (continues to next spec) if DRAFTER isn't available.
//
// Env:
//   LLAMA_TEST_TARGET  — target GGUF path (required; else exit 77)
//   LLAMA_TEST_DRAFTER — DFlash drafter GGUF path (optional; skips dflash)

#include "common.h"
#include "speculative.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static int run_init(const std::string & target_path, common_speculative_type spec,
                    int n_draft, const std::string & drafter_path) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    if (spec == COMMON_SPECULATIVE_TYPE_MTP) mparams.mtp = true;
    llama_model * model = llama_model_load_from_file(target_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "  load failed for %s\n", target_path.c_str());
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    if (spec == COMMON_SPECULATIVE_TYPE_MTP) {
        cparams.mtp = true; cparams.mtp_op_type = MTP_OP_NONE;
        cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { llama_free_model(model); return 2; }

    int rc = 0;
    if (spec == COMMON_SPECULATIVE_TYPE_NONE) {
        // No spec init; trivially OK.
        printf("  none: OK\n");
    } else {
        common_params_speculative sp;
        sp.type = spec; sp.n_max = n_draft; sp.n_min = 0;
        sp.p_min = 0.0f; sp.p_split = 0.0f;
        if (spec == COMMON_SPECULATIVE_TYPE_MTP) {
            if (llama_model_n_nextn_layer(model) <= 0) {
                printf("  mtp: SKIP (model has no NextN layers)\n");
                rc = 0;
            } else {
                sp.cparams_dft = cparams;
                sp.cparams_dft.mtp = true;
                sp.cparams_dft.mtp_op_type = MTP_OP_WARMUP;
                sp.cparams_dft.embeddings = true;
            }
        } else if (spec == COMMON_SPECULATIVE_TYPE_DFLASH) {
            sp.mparams_dft.path = drafter_path;
        }
        if (rc == 0 && !(spec == COMMON_SPECULATIVE_TYPE_MTP && llama_model_n_nextn_layer(model) <= 0)) {
            common_speculative * s = common_speculative_init(sp, ctx, 0);
            if (!s) { fprintf(stderr, "  init failed\n"); rc = 3; }
            else { printf("  %s: OK\n", common_speculative_type_to_str(spec).c_str()); common_speculative_free(s); }
        }
    }
    llama_free(ctx); llama_free_model(model);
    return rc;
}

int main() {
    const char * target  = std::getenv("LLAMA_TEST_TARGET");
    const char * drafter = std::getenv("LLAMA_TEST_DRAFTER");
    if (!target) { fprintf(stderr, "LLAMA_TEST_TARGET unset — SKIP\n"); return 77; }
    llama_backend_init();
    int fails = 0;
    fails += run_init(target, COMMON_SPECULATIVE_TYPE_NONE,   0, "") != 0;
    fails += run_init(target, COMMON_SPECULATIVE_TYPE_MTP,    3, "") != 0;
    if (drafter) fails += run_init(target, COMMON_SPECULATIVE_TYPE_DFLASH, 4, drafter) != 0;
    else fprintf(stderr, "  dflash: SKIP (LLAMA_TEST_DRAFTER unset)\n");
    llama_backend_free();
    return fails ? 1 : 0;
}
