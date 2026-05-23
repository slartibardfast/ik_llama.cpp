// test-dflash-extract-observational.cpp
//
// Binding test for the P0.A.3 root cause: residual extraction must
// NOT perturb the target's forward output. With the cb_eval hook
// installed (the current mechanism), the scheduler splits the
// per-backend graph at every watched `l_out-<il>` source node and
// re-issues ggml_backend_graph_compute_async on each sub-range,
// changing fusion / cudaGraph capture boundaries on the CUDA backend
// and propagating an argmax flip through downstream layers. Specs:
//
//   /home/llm/yarn-agentic/specs/dflash/cb_eval_residual_capture.allium
//       (contract ResidualExtract; the
//        ResidualExtractObservationallyEquivalent property)
//   /home/llm/yarn-agentic/specs/dflash/CbEvalObservational.tla
//       (TLA model: ObservationalEquivalence and SchedulerStaysFastPath
//        properties — out-of-scope for scripts/check-bindings.py which
//        only scans the core DFlash spec/binding pair)
//
// The test:
//
//   1. Decode a prompt of length P on a fresh context with extract
//      DISARMED. Record the argmax of every output row (i.e., for the
//      prompt's last token's next-token prediction; we ask for every
//      row to get the full per-position prediction).
//
//   2. Reset the KV cache, ARM extract for layers {1, 16, 31, 46, 61}
//      (the Qwen3.6-27B-DFlash production set), and decode the same
//      prompt with the same seed/cparams.
//
//   3. Assert the two argmax sequences are byte-identical.
//
//   4. Test-mechanism check (per feedback_verify_test_mechanism_before_trusting):
//      assert the extract buffer for layer 0 / seq 0 has > 0 floats,
//      proving the cb_eval path actually fired during step 2. A
//      passing test where the buffer is empty would mean extract
//      didn't actually run, the byte-equality is trivial, and the
//      test is non-binding.
//
// On HEAD (cb_eval mechanism): step 3 fails. After the fix
// (graph-tap-node mechanism per GraphTapEmission contract): step 3
// passes; step 4 still passes because tap nodes also fill the buffer.
//
// Env:
//   LLAMA_TEST_TARGET — target GGUF (skip with 77 if unset)
//
// Optional env:
//   LLAMA_TEST_EXTRACT_LAYERS — comma-separated layer ids (default
//       "1,16,31,46,61"). Useful for sweeping.
//   LLAMA_TEST_PROMPT — override prompt string (default short prompt).

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<int32_t> parse_layer_csv(const char * csv) {
    std::vector<int32_t> out;
    if (!csv || !*csv) {
        out = {1, 16, 31, 46, 61};
        return out;
    }
    const char * p = csv;
    while (*p) {
        char * end = nullptr;
        long v = std::strtol(p, &end, 10);
        if (end == p) break;
        out.push_back((int32_t) v);
        p = end;
        while (*p == ',' || *p == ' ') ++p;
    }
    if (out.empty()) out = {1, 16, 31, 46, 61};
    return out;
}

static std::vector<llama_token>
argmax_per_row(const float * logits, int n_rows, int n_vocab) {
    std::vector<llama_token> out;
    out.reserve(n_rows);
    for (int r = 0; r < n_rows; ++r) {
        const float * row = logits + (size_t) r * (size_t) n_vocab;
        int best = 0;
        float best_v = row[0];
        for (int v = 1; v < n_vocab; ++v) {
            if (row[v] > best_v) {
                best_v = row[v];
                best = v;
            }
        }
        out.push_back((llama_token) best);
    }
    return out;
}

static int decode_prompt_and_generate(
        llama_context * ctx,
        const std::vector<llama_token> & prompt_tokens,
        int n_generate,
        std::vector<llama_token> & out_argmax_prefill,
        std::vector<llama_token> & out_generated) {

    const int n_prompt = (int) prompt_tokens.size();
    const int n_vocab  = llama_n_vocab(llama_get_model(ctx));

    llama_kv_cache_seq_rm(ctx, 0, -1, -1);

    // Prefill — capture per-row argmax of all positions.
    llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; ++i) {
        common_batch_add(batch, prompt_tokens[i], i, {(llama_seq_id) 0}, true);
    }
    int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        std::fprintf(stderr, "[observational] prefill llama_decode failed rc=%d\n", rc);
        return 1;
    }
    const float * logits_prefill = llama_get_logits(ctx);
    if (!logits_prefill) {
        std::fprintf(stderr, "[observational] llama_get_logits after prefill returned NULL\n");
        return 1;
    }
    out_argmax_prefill = argmax_per_row(logits_prefill, n_prompt, n_vocab);

    // Verify-batch-style generation — repeated MULTI-ROW decodes that
    // mirror DFlash CLI's verify batch shape (5 rows per decode: an
    // anchor row + 4 draft rows). After the prefill, we drive M
    // verify cycles. Each cycle:
    //   1. Build a 5-token batch at positions [pos, pos+1, ..., pos+4]
    //      with logits=true on every row. Tokens are sourced
    //      auto-regressively from the previous cycle's argmax of the
    //      LAST row (so each cycle has a stable, deterministic input
    //      under spec-none).
    //   2. Decode. The graph builder emits an F16 `l_out-<il>` (since
    //      n_tokens > 1) which is what triggers the cb_eval F16 path
    //      most heavily.
    //   3. Argmax-sample every row, append all 5 to out_generated,
    //      advance pos by 5.
    //
    // This is the closest shape to DFlash CLI's actual decode pattern
    // without instantiating the drafter / combine_features /
    // inject_kv pipeline. If cb_eval-induced fusion-break perturbs
    // the target's multi-row F16 verify path, repeated verify cycles
    // accumulate the drift exactly like DFlash CLI does.
    out_generated.clear();
    out_generated.reserve((size_t) n_generate);

    const int verify_width = []() {
        const char * env = std::getenv("LLAMA_TEST_VERIFY_WIDTH");
        return (env && *env) ? std::atoi(env) : 5;
    }();
    if (verify_width < 1) {
        std::fprintf(stderr, "[observational] bad LLAMA_TEST_VERIFY_WIDTH < 1\n");
        return 1;
    }

    llama_token next = out_argmax_prefill.back();
    int pos = n_prompt;
    const int n_cycles = (n_generate + verify_width - 1) / verify_width;
    for (int c = 0; c < n_cycles; ++c) {
        // Source tokens for this verify-batch: the anchor (next), then
        // verify_width-1 placeholder tokens. The placeholders simulate
        // the drafter's output — we use the last sampled token padded
        // forward so the test is deterministic. The bug shape we care
        // about is in the TARGET forward, not the source tokens.
        llama_batch b = llama_batch_init(verify_width, 0, 1);
        for (int k = 0; k < verify_width; ++k) {
            common_batch_add(b, next, pos + k, {(llama_seq_id) 0}, true);
        }
        rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0) {
            std::fprintf(stderr, "[observational] verify cycle %d decode rc=%d\n", c, rc);
            return 1;
        }
        const float * gl = llama_get_logits(ctx);
        if (!gl) return 1;
        for (int k = 0; k < verify_width; ++k) {
            const float * row = gl + (size_t) k * (size_t) n_vocab;
            int best = 0;
            float best_v = row[0];
            for (int v = 1; v < n_vocab; ++v) {
                if (row[v] > best_v) { best_v = row[v]; best = v; }
            }
            out_generated.push_back((llama_token) best);
        }
        next = out_generated.back();
        pos += verify_width;
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
        std::fprintf(stderr, "[observational] prompt too short (%zu tokens)\n", tokens.size());
        return 1;
    }

    const int n_generate = []() {
        const char * env = std::getenv("LLAMA_TEST_NGEN");
        if (env && *env) return std::atoi(env);
        return 32;
    }();

    // Step 1: decode with extract DISARMED — prefill + autoregress.
    int32_t no_layers = 0;
    llama_set_dflash_extract_layers(ctx, &no_layers, 0);

    std::vector<llama_token> argmax_baseline_prefill, gen_baseline;
    if (decode_prompt_and_generate(ctx, tokens, n_generate,
                                   argmax_baseline_prefill, gen_baseline) != 0) {
        return 1;
    }
    std::fprintf(stderr, "[observational] baseline: prefill=%zu rows, generated=%zu tokens\n",
                 argmax_baseline_prefill.size(), gen_baseline.size());

    // Step 2: decode with extract ARMED — same prefill + autoregress.
    std::vector<int32_t> layers = parse_layer_csv(std::getenv("LLAMA_TEST_EXTRACT_LAYERS"));
    std::fprintf(stderr, "[observational] arming extract on layers:");
    for (int32_t il : layers) std::fprintf(stderr, " %d", (int) il);
    std::fprintf(stderr, "\n");
    llama_set_dflash_extract_layers(ctx, layers.data(), (int32_t) layers.size());

    std::vector<llama_token> argmax_extract_prefill, gen_extract;
    if (decode_prompt_and_generate(ctx, tokens, n_generate,
                                   argmax_extract_prefill, gen_extract) != 0) {
        return 1;
    }
    std::fprintf(stderr, "[observational] extract: prefill=%zu rows, generated=%zu tokens\n",
                 argmax_extract_prefill.size(), gen_extract.size());

    // Step 3: byte-equal argmax check — both the prefill argmax sequence
    // and the generated token sequence must match.
    int failures = 0;
    auto cmp = [&failures](const char * label,
                           const std::vector<llama_token> & a,
                           const std::vector<llama_token> & b) {
        if (a.size() != b.size()) {
            std::fprintf(stderr, "[FAIL] %s row count mismatch baseline=%zu extract=%zu\n",
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
                         "(baseline=%d extract=%d)\n",
                         label, n_diff, a.size(), first_diff,
                         (int) a[first_diff], (int) b[first_diff]);
            failures++;
        }
    };
    cmp("prefill argmax", argmax_baseline_prefill, argmax_extract_prefill);
    cmp("generated tokens", gen_baseline, gen_extract);

    // Step 4: mechanism check — extract buffer for layer 0 / seq 0 must
    // be non-empty. If empty, the test is not exercising the extract
    // path and the byte-equality of step 3 is trivially true (because
    // the would-be-perturbing path didn't run). Per
    // feedback_verify_test_mechanism_before_trusting.
    const int n_embd = llama_model_n_embd(model);
    const size_t expected_min = (size_t) tokens.size() * (size_t) n_embd;
    std::vector<float> buf(expected_min, 0.0f);
    size_t got = llama_get_dflash_extract_data_seq(ctx, 0, 0, buf.data(), buf.size());
    std::fprintf(stderr,
                 "[observational] extract slot 0 seq 0: %zu floats (expected >= %zu)\n",
                 got, expected_min);
    if (got < expected_min) {
        std::fprintf(stderr,
                     "[FAIL] gate 3: extract path did not populate buffer for layer slot 0; "
                     "the observational test is non-binding\n");
        failures++;
    }

    if (failures == 0) {
        std::printf("[PASS] DFlash residual extract is observationally equivalent to "
                    "spec-none: prefill=%zu argmax rows match, generated=%zu tokens match, "
                    "extract buffer = %zu floats\n",
                    argmax_baseline_prefill.size(), gen_baseline.size(), got);
    } else {
        std::printf("[FAIL] %d gate(s) failed — see stderr above\n", failures);
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return failures == 0 ? 0 : 1;
}
