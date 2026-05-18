// test-dflash-spec-batched-fanout.cpp
//
// DFlash multi-slot adapter-glue binding gate: confirms that the
// orchestrator-level fan-out
//   common_speculative_draft_batched({spec0, spec1})
// produces byte-identical candidates to two serial
//   common_speculative_draft(spec_i, ...)
// calls, AND that an asymmetric per-seq prefill produces per-slot
// candidates that differ between slots (proving seq_id routing through
// the all-DFlash gate).
//
// Two specs are constructed over the SAME ctx_tgt at n_seq_max=2; the
// second spec's state hits the share-the-bind path on the DFlash
// drafter (no second llama_set_dflash).
//
// Env:
//   LLAMA_TEST_TARGET  — target GGUF (skip 77 if unset)
//   LLAMA_TEST_DRAFTER — drafter GGUF (skip 77 if unset)

#include "common.h"
#include "speculative.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Prefill both seq_ids in a single 2-seq batched ubatch. Matches the
// production server flow more closely than two sequential single-seq
// decodes, and exercises the cb_eval per-seq demux path that Phase 3
// pins. Tokens may differ per seq (asymmetric mode).
static bool prefill_both(llama_context * ctx,
                         const std::vector<llama_token> & t0,
                         const std::vector<llama_token> & t1) {
    llama_kv_cache_seq_rm(ctx, 0, -1, -1);
    llama_kv_cache_seq_rm(ctx, 1, -1, -1);
    const int n_total = (int) t0.size() + (int) t1.size();
    llama_batch batch = llama_batch_init(n_total, 0, 1);
    for (int i = 0; i < (int) t0.size(); ++i) {
        const bool last = (i == (int) t0.size() - 1);
        common_batch_add(batch, t0[i], i, {(llama_seq_id) 0}, last);
    }
    for (int i = 0; i < (int) t1.size(); ++i) {
        const bool last = (i == (int) t1.size() - 1);
        common_batch_add(batch, t1[i], i, {(llama_seq_id) 1}, last);
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

int main() {
    const char * target_path  = std::getenv("LLAMA_TEST_TARGET");
    const char * drafter_path = std::getenv("LLAMA_TEST_DRAFTER");
    if (!target_path)  { fprintf(stderr, "SKIP: set LLAMA_TEST_TARGET\n");  return 77; }
    if (!drafter_path) { fprintf(stderr, "SKIP: set LLAMA_TEST_DRAFTER\n"); return 77; }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    mparams.split_mode   = LLAMA_SPLIT_MODE_GRAPH;
    static const char * dev_csv = "CUDA0,CUDA1";
    mparams.devices = dev_csv;
    llama_model * model = llama_model_load_from_file(target_path, mparams);
    if (!model) { fprintf(stderr, "load target failed: %s\n", target_path); return 77; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = 4096 * 2;
    cparams.n_batch     = 2048;
    cparams.n_ubatch    = 2048;
    cparams.n_seq_max   = 2;
    cparams.type_k      = GGML_TYPE_Q4_0;
    cparams.type_v      = GGML_TYPE_Q4_0;
    cparams.flash_attn  = true;
    cparams.mla_attn    = 3;
    cparams.k_cache_hadamard = true;
    cparams.v_cache_hadamard = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { llama_free_model(model); fprintf(stderr, "ctx init failed\n"); return 77; }

    // Build per-slot common_speculative_init params.
    common_params_speculative sp_params;
    sp_params.type = COMMON_SPECULATIVE_TYPE_DFLASH;
    sp_params.mparams_dft.path = drafter_path;
    sp_params.n_max = 4;

    common_speculative * spec0 = common_speculative_init(sp_params, ctx, /*seq_id=*/ 0);
    if (!spec0) { fprintf(stderr, "[FAIL] common_speculative_init(seq_id=0) failed\n"); return 1; }
    common_speculative * spec1 = common_speculative_init(sp_params, ctx, /*seq_id=*/ 1);
    if (!spec1) { fprintf(stderr, "[FAIL] common_speculative_init(seq_id=1) failed (share-the-bind broken?)\n"); return 1; }

    // Sanity: only one drafter object is bound to the ctx.
    llama_dflash_drafter * bound = llama_get_dflash_drafter(ctx);
    if (!bound) { fprintf(stderr, "[FAIL] no drafter bound after init\n"); return 1; }
    fprintf(stderr, "[fanout] drafter bound: %p\n", (void *) bound);

    const int32_t BS = llama_dflash_block_size(bound);
    if (BS <= 0) { fprintf(stderr, "[FAIL] bad BS=%d\n", BS); return 1; }
    fprintf(stderr, "[fanout] BS=%d\n", BS);

    // ── Symmetric: same prompt on both seqs → batched ≡ serial ──
    {
        const std::string prompt = "The capital of France is Paris.";
        std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true, true);
        if ((int) tokens.size() < 2) { fprintf(stderr, "[FAIL] prompt too short\n"); return 1; }
        if (!prefill_both(ctx, tokens, tokens)) { fprintf(stderr, "[FAIL] prefill sym\n"); return 1; }

        // Serial reference: common_speculative_draft on each spec.
        common_params_speculative tmp0 = sp_params;
        common_params_speculative tmp1 = sp_params;
        llama_tokens A0 = common_speculative_draft(spec0, tmp0, tokens, tokens.back());
        llama_tokens A1 = common_speculative_draft(spec1, tmp1, tokens, tokens.back());
        if ((int) A0.size() != sp_params.n_max || (int) A1.size() != sp_params.n_max) {
            fprintf(stderr, "[FAIL] serial sizes: A0=%zu A1=%zu (expected %d)\n",
                    A0.size(), A1.size(), sp_params.n_max);
            return 1;
        }

        // Batched: fan-out over both specs.
        std::vector<common_speculative_batched_in> inputs = {
            { spec0, tokens, tokens.back() },
            { spec1, tokens, tokens.back() },
        };
        std::vector<llama_tokens> B = common_speculative_draft_batched(inputs, sp_params);
        if (B.size() != 2 || (int) B[0].size() != sp_params.n_max || (int) B[1].size() != sp_params.n_max) {
            fprintf(stderr, "[FAIL] batched sizes: B[0]=%zu B[1]=%zu\n",
                    B.size() > 0 ? B[0].size() : 0, B.size() > 1 ? B[1].size() : 0);
            return 1;
        }

        fprintf(stderr, "[fanout/sym] A0: %d %d %d %d\n", A0[0], A0[1], A0[2], A0[3]);
        fprintf(stderr, "[fanout/sym] A1: %d %d %d %d\n", A1[0], A1[1], A1[2], A1[3]);
        fprintf(stderr, "[fanout/sym] B0: %d %d %d %d\n", B[0][0], B[0][1], B[0][2], B[0][3]);
        fprintf(stderr, "[fanout/sym] B1: %d %d %d %d\n", B[1][0], B[1][1], B[1][2], B[1][3]);

        int failures = 0;
        for (int b = 0; b < sp_params.n_max; ++b) {
            if (A0[b] != B[0][b]) { failures++; fprintf(stderr, "  slot0[%d]: A=%d B=%d\n", b, A0[b], B[0][b]); }
            if (A1[b] != B[1][b]) { failures++; fprintf(stderr, "  slot1[%d]: A=%d B=%d\n", b, A1[b], B[1][b]); }
        }
        if (failures != 0) {
            fprintf(stderr, "[FAIL] symmetric: %d mismatch(es) (batched != serial)\n", failures);
            return 1;
        }
        fprintf(stderr, "[PASS] symmetric: batched ≡ serial\n");
    }

    // ── Asymmetric: different prompts → per-slot candidates differ ──
    {
        const std::string p0 = "The capital of France is Paris.";
        const std::string p1 = "Once upon a time in a galaxy far far away";
        std::vector<llama_token> t0 = common_tokenize(ctx, p0, true, true);
        std::vector<llama_token> t1 = common_tokenize(ctx, p1, true, true);
        if ((int) t0.size() < 2 || (int) t1.size() < 2) { fprintf(stderr, "[FAIL] asym prompts too short\n"); return 1; }
        if (!prefill_both(ctx, t0, t1)) { fprintf(stderr, "[FAIL] prefill asym\n"); return 1; }

        std::vector<common_speculative_batched_in> inputs = {
            { spec0, t0, t0.back() },
            { spec1, t1, t1.back() },
        };
        std::vector<llama_tokens> B = common_speculative_draft_batched(inputs, sp_params);
        if (B.size() != 2 || (int) B[0].size() != sp_params.n_max || (int) B[1].size() != sp_params.n_max) {
            fprintf(stderr, "[FAIL] asym batched sizes: B[0]=%zu B[1]=%zu\n",
                    B.size() > 0 ? B[0].size() : 0, B.size() > 1 ? B[1].size() : 0);
            return 1;
        }
        fprintf(stderr, "[fanout/asym] B0: %d %d %d %d\n", B[0][0], B[0][1], B[0][2], B[0][3]);
        fprintf(stderr, "[fanout/asym] B1: %d %d %d %d\n", B[1][0], B[1][1], B[1][2], B[1][3]);

        // The per-slot candidates must differ when the per-seq prefills
        // differ — proves seq_id is routed through the all-DFlash gate.
        bool any_diff = false;
        for (int b = 0; b < sp_params.n_max; ++b) {
            if (B[0][b] != B[1][b]) { any_diff = true; break; }
        }
        if (!any_diff) {
            fprintf(stderr, "[FAIL] asymmetric: per-slot outputs are identical (seq_id not flowing?)\n");
            return 1;
        }
        fprintf(stderr, "[PASS] asymmetric: per-slot outputs differ\n");
    }

    common_speculative_free(spec0);
    common_speculative_free(spec1);
    llama_set_dflash(ctx, nullptr);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    std::printf("[PASS] DFlash spec_batched_fanout: symmetric ≡ serial + asymmetric per-slot routing\n");
    return 0;
}
