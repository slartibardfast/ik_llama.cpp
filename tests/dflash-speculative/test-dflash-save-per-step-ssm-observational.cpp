// test-dflash-save-per-step-ssm-observational.cpp
//
// L1 binding test for the P0.A.3 root cause narrowed by the bisect
// ladder (see PHASE_NSTREAM_KV_PERF.md "P0.A.3 test ladder for
// save_per_step_ssm / save_all_steps binding").
//
// The observation under test:
//
//   Toggling `save_per_step_ssm = true` (via llama_spec_ckpt_init
//   PER_STEP + llama_spec_ckpt_save before the verify-style decode)
//   must NOT change the per-row argmax of the verify-style multi-row
//   decode that follows.
//
//   Both runs use the same prompt, same KV-cache contents at the start
//   of the verify cycles, same sampling (greedy argmax), and the same
//   batch shape. The only difference is whether save_per_step_ssm is
//   set BEFORE the verify decode.
//
// On HEAD: the bisect ladder (run E in PHASE_NSTREAM_KV_PERF.md) shows
// that gating save_per_step_ssm OFF flips the CLI from degenerate
// "...quick quick quick..." to a different (still non-degenerate)
// token sequence, with a different accept rate. This libllama-level
// test reproduces the same flag flip without instantiating the DFlash
// drafter pipeline, isolating the kernel-side cause from cb_eval,
// extract layers, combine_features and inject_kv_fused.
//
// Test shape mirrors test-dflash-extract-observational.cpp (the test
// that exonerated cb_eval): prefill the prompt, then drive M
// verify-style 5-row decodes auto-regressively. The DFlash CLI's
// kernel-perturbing path is the multi-row verify decode under
// save_per_step_ssm — this test exercises exactly that.
//
// Env:
//   LLAMA_TEST_TARGET — target GGUF (skip with 77 if unset).
//
// Optional env:
//   LLAMA_TEST_PROMPT       — override prompt string.
//   LLAMA_TEST_NGEN         — number of tokens to generate (default 32).
//   LLAMA_TEST_VERIFY_WIDTH — rows per verify decode (default 5).
//   LLAMA_TEST_MAX_DRAFT    — max_tokens arg to spec_ckpt_init
//                             (default verify_width).

#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int decode_prompt_and_generate(
        llama_context * ctx,
        const std::vector<llama_token> & prompt_tokens,
        int n_generate,
        bool arm_spec_ckpt,
        int  max_draft,
        std::vector<llama_token> & out_argmax_prefill,
        std::vector<llama_token> & out_generated) {

    const int n_prompt = (int) prompt_tokens.size();
    const int n_vocab  = llama_n_vocab(llama_get_model(ctx));

    llama_kv_cache_seq_rm(ctx, 0, -1, -1);

    // Always run prefill with spec-none — the bisect localised the
    // perturbation to the verify-style multi-row decode under
    // save_per_step_ssm, not the single-pass prefill.
    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; ++i) {
        common_batch_add(batch, prompt_tokens[i], i, {(llama_seq_id) 0}, true);
    }
    int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        std::fprintf(stderr, "[save-ssm] prefill llama_decode failed rc=%d\n", rc);
        return 1;
    }
    const float * logits_prefill = llama_get_logits(ctx);
    if (!logits_prefill) {
        std::fprintf(stderr, "[save-ssm] llama_get_logits after prefill returned NULL\n");
        return 1;
    }
    out_argmax_prefill.clear();
    out_argmax_prefill.reserve(n_prompt);
    for (int r = 0; r < n_prompt; ++r) {
        const float * row = logits_prefill + (size_t) r * (size_t) n_vocab;
        int best = 0; float best_v = row[0];
        for (int v = 1; v < n_vocab; ++v) {
            if (row[v] > best_v) { best_v = row[v]; best = v; }
        }
        out_argmax_prefill.push_back((llama_token) best);
    }

    out_generated.clear();
    out_generated.reserve((size_t) n_generate);

    const int verify_width = []() {
        const char * env = std::getenv("LLAMA_TEST_VERIFY_WIDTH");
        return (env && *env) ? std::atoi(env) : 5;
    }();
    if (verify_width < 1) {
        std::fprintf(stderr, "[save-ssm] bad LLAMA_TEST_VERIFY_WIDTH < 1\n");
        return 1;
    }

    llama_token next = out_argmax_prefill.back();
    int pos = n_prompt;
    const int n_cycles = (n_generate + verify_width - 1) / verify_width;

    for (int c = 0; c < n_cycles; ++c) {
        // ARM (or do nothing) BEFORE each verify decode, matching what
        // common_speculative_state_dflash::draft does once per cycle in
        // common/speculative.cpp around lines 980-1096. The init+save
        // pair sets save_per_step_ssm=true for this verify decode in
        // PER_STEP mode (or is a no-op when arm_spec_ckpt is false).
        if (arm_spec_ckpt) {
            const int mode = llama_spec_ckpt_init(ctx, LLAMA_SPEC_CKPT_PER_STEP, max_draft);
            if (mode != LLAMA_SPEC_CKPT_PER_STEP) {
                std::fprintf(stderr,
                             "[save-ssm] spec_ckpt_init returned mode=%d "
                             "(expected PER_STEP=%d) — model does not support "
                             "per-step ckpt; SKIP\n",
                             mode, (int) LLAMA_SPEC_CKPT_PER_STEP);
                return 77;
            }
            if (!llama_spec_ckpt_save(ctx, 0)) {
                std::fprintf(stderr, "[save-ssm] spec_ckpt_save failed\n");
                return 1;
            }
        }

        // Build the verify-style multi-row batch: anchor + (verify_width-1)
        // placeholder rows, all at successive positions. Source tokens
        // are deterministic given the prior cycle's last argmax. The
        // bug signature lives in the TARGET forward over multi-row F16
        // l_out tensors, not in the draft tokens.
        llama_batch b = llama_batch_init(verify_width, 0, 1);
        for (int k = 0; k < verify_width; ++k) {
            common_batch_add(b, next, pos + k, {(llama_seq_id) 0}, true);
        }
        rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0) {
            std::fprintf(stderr, "[save-ssm] verify cycle %d decode rc=%d\n", c, rc);
            if (arm_spec_ckpt) llama_spec_ckpt_discard(ctx);
            return 1;
        }
        const float * gl = llama_get_logits(ctx);
        if (!gl) {
            if (arm_spec_ckpt) llama_spec_ckpt_discard(ctx);
            return 1;
        }
        for (int k = 0; k < verify_width; ++k) {
            const float * row = gl + (size_t) k * (size_t) n_vocab;
            int best = 0; float best_v = row[0];
            for (int v = 1; v < n_vocab; ++v) {
                if (row[v] > best_v) { best_v = row[v]; best = v; }
            }
            out_generated.push_back((llama_token) best);
        }
        next = out_generated.back();
        pos += verify_width;

        // Mirror the CLI: after the verify decode, discard the
        // checkpoint so the next cycle starts cleanly. The bisect
        // showed restore() is NOT the sole perturber; save() is. We
        // use discard here to keep the test scoped to the save_ flag.
        if (arm_spec_ckpt) {
            llama_spec_ckpt_discard(ctx);
        }
    }
    out_generated.resize((size_t) n_generate);
    return 0;
}

int main() {
    const char * target = std::getenv("LLAMA_TEST_TARGET");
    if (!target) {
        std::fprintf(stderr, "SKIP: set LLAMA_TEST_TARGET\n");
        return 77;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    mparams.split_mode   = LLAMA_SPLIT_MODE_GRAPH;
    static const char * dev_csv = "CUDA0,CUDA1";
    mparams.devices = dev_csv;
    llama_model * model = llama_model_load_from_file(target, mparams);
    if (!model) {
        std::fprintf(stderr, "load failed: %s\n", target);
        return 77;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = 4096;
    cparams.n_batch     = 2048;
    cparams.n_ubatch    = 2048;
    cparams.n_seq_max   = 1;
    cparams.type_k      = GGML_TYPE_Q4_0;
    cparams.type_v      = GGML_TYPE_Q4_0;
    cparams.flash_attn  = true;
    cparams.k_cache_hadamard = true;
    cparams.v_cache_hadamard = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_free_model(model);
        std::fprintf(stderr, "ctx init failed\n");
        return 77;
    }

    const std::string prompt = []() {
        const char * env = std::getenv("LLAMA_TEST_PROMPT");
        if (env && *env) return std::string(env);
        return std::string("The capital of France is Paris. The capital of Germany is");
    }();
    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true, true);
    if ((int) tokens.size() < 4) {
        std::fprintf(stderr, "[save-ssm] prompt too short (%zu tokens)\n", tokens.size());
        return 1;
    }

    const int n_generate = []() {
        const char * env = std::getenv("LLAMA_TEST_NGEN");
        if (env && *env) return std::atoi(env);
        return 32;
    }();

    const int verify_width = []() {
        const char * env = std::getenv("LLAMA_TEST_VERIFY_WIDTH");
        return (env && *env) ? std::atoi(env) : 5;
    }();
    const int max_draft = []() {
        const char * env = std::getenv("LLAMA_TEST_MAX_DRAFT");
        return (env && *env) ? std::atoi(env) : -1;
    }();
    const int max_draft_arg = max_draft > 0 ? max_draft : verify_width;

    // Run 1: spec-ckpt DISARMED — save_per_step_ssm stays false.
    std::vector<llama_token> argmax_baseline_prefill, gen_baseline;
    int rc_a = decode_prompt_and_generate(ctx, tokens, n_generate,
                                          /*arm_spec_ckpt=*/false, max_draft_arg,
                                          argmax_baseline_prefill, gen_baseline);
    if (rc_a == 77) { llama_free(ctx); llama_free_model(model); llama_backend_free(); return 77; }
    if (rc_a != 0) {
        llama_free(ctx); llama_free_model(model); llama_backend_free();
        return 1;
    }
    std::fprintf(stderr, "[save-ssm] disarmed: prefill=%zu rows, generated=%zu tokens\n",
                 argmax_baseline_prefill.size(), gen_baseline.size());

    // Run 2: spec-ckpt ARMED per verify cycle — save_per_step_ssm = true.
    std::vector<llama_token> argmax_armed_prefill, gen_armed;
    int rc_b = decode_prompt_and_generate(ctx, tokens, n_generate,
                                          /*arm_spec_ckpt=*/true, max_draft_arg,
                                          argmax_armed_prefill, gen_armed);
    if (rc_b == 77) { llama_free(ctx); llama_free_model(model); llama_backend_free(); return 77; }
    if (rc_b != 0) {
        llama_free(ctx); llama_free_model(model); llama_backend_free();
        return 1;
    }
    std::fprintf(stderr, "[save-ssm] armed: prefill=%zu rows, generated=%zu tokens\n",
                 argmax_armed_prefill.size(), gen_armed.size());

    // Gate 1: prefill argmax — must match (spec-ckpt is not armed during
    // the single-pass prefill in either run).
    int failures = 0;
    auto cmp = [&failures](const char * label,
                           const std::vector<llama_token> & a,
                           const std::vector<llama_token> & b) {
        if (a.size() != b.size()) {
            std::fprintf(stderr, "[FAIL] %s row count mismatch disarmed=%zu armed=%zu\n",
                         label, a.size(), b.size());
            failures++;
            return;
        }
        int first_diff = -1;
        int n_diff = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i] != b[i]) {
                if (first_diff < 0) first_diff = (int) i;
                ++n_diff;
            }
        }
        if (n_diff > 0) {
            std::fprintf(stderr,
                         "[FAIL] %s %d/%zu rows differ; first diff at row %d "
                         "(disarmed=%d armed=%d)\n",
                         label, n_diff, a.size(), first_diff,
                         (int) a[first_diff], (int) b[first_diff]);
            failures++;
        }
    };
    cmp("prefill argmax",   argmax_baseline_prefill, argmax_armed_prefill);
    cmp("generated tokens", gen_baseline,            gen_armed);

    if (failures == 0) {
        std::printf("[PASS] save_per_step_ssm is observationally equivalent to "
                    "spec-none on the verify-batch decode: prefill=%zu argmax rows "
                    "match, generated=%zu tokens match\n",
                    argmax_baseline_prefill.size(), gen_baseline.size());
    } else {
        std::printf("[FAIL] %d gate(s) failed — save_per_step_ssm perturbs the "
                    "verify-batch decode at libllama level\n", failures);
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return failures == 0 ? 0 : 1;
}
