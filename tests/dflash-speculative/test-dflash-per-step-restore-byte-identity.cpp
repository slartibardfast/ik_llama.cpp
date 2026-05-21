// test-dflash-per-step-restore-byte-identity.cpp
//
// L2 binding test for P0.A.3 Suspect 4 (per_step_restore semantics).
// Tests the load-bearing claim of the per-step checkpoint protocol:
//
//   After llama_spec_ckpt_save(seq) → N-token verify decode →
//   llama_spec_ckpt_restore(seq, n_past, accepted_step), the
//   recurrent state in s_l[il] AND the KV cache layout AND the
//   sequence position MUST equal what they would be after a clean
//   (accepted_step+1)-token decode of the SAME token prefix from
//   the SAME pre-verify state.
//
// We bind this observationally: if state-after-restore equals
// state-after-fresh-decode, then any subsequent decode at position
// (n_past + accepted_step + 1) must produce byte-identical logits.
//
// Two contexts, same model, same kernels. Both decode an identical
// prompt then identical "bonus" token at position P + 3:
//
//   Path A — fresh ctx, spec-ckpt never armed:
//     1. Prefill prompt (P tokens).
//     2. Decode 3-token batch [t0, t1, t2] at positions [P, P+1, P+2].
//     3. Decode 1-token bonus at position P+3.
//     4. Capture logits at row 0 of step 3.
//
//   Path B — fresh ctx, spec-ckpt PER_STEP armed for verify:
//     1. Prefill prompt (P tokens).
//     2. llama_spec_ckpt_init(PER_STEP, max_tokens=5).
//     3. llama_spec_ckpt_save(seq=0).
//     4. Decode 5-token verify [t0, t1, t2, t3, t4] at positions
//        [P..P+4] (this is the CLI's verify-batch shape).
//     5. llama_spec_ckpt_restore(seq=0, n_past=P, accepted_step=2)
//        — stitches state back to "after [t0..t2]".
//     6. Decode 1-token bonus at position P+3.
//     7. Capture logits at row 0 of step 6.
//
// Gate: logits_A and logits_B must be byte-identical (every fp32 bit).
// On PASS: per_step_restore correctly mirrors a fresh-decode state.
// On FAIL: restore's reconstructed recurrent / KV state diverges from
//          the fresh decode — Suspect 4 confirmed at the libllama
//          layer. The diff at the first divergent row points at which
//          layer family (target's full-attn vs DeltaNet recurrent)
//          carries the divergence, depending on logit sensitivity.
//
// Note: this test does NOT exercise the DFlash drafter pipeline
// (combine_features / inject_kv_fused / drafter_forward). If L2
// PASSES, the bug needs the drafter pipeline running and we go to L3.
// If L2 FAILS, the bug is in libllama spec-ckpt regardless of DFlash.
//
// Env:
//   LLAMA_TEST_TARGET — target GGUF (skip with 77 if unset).
//
// Optional env:
//   LLAMA_TEST_PROMPT — override prompt string.

#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Decode batch helper — wraps llama_batch_init / common_batch_add /
// llama_decode / llama_batch_free. Sets logits=true on every row.
static int decode_tokens(llama_context * ctx,
                         const std::vector<llama_token> & toks,
                         llama_pos start_pos,
                         llama_seq_id seq_id) {
    const int n = (int) toks.size();
    if (n == 0) return 0;
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        common_batch_add(b, toks[i], start_pos + i, {seq_id}, true);
    }
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc;
}

// Build a fresh llama_context with the same cparams used by the
// dflash-speculative-simple CLI on production/2026-q2-next.
static llama_context * fresh_ctx(llama_model * model) {
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
    return llama_init_from_model(model, cparams);
}

// Capture the logits for row 0 (the only row in a 1-token decode).
static std::vector<float> capture_row0_logits(llama_context * ctx, int n_vocab) {
    const float * gl = llama_get_logits(ctx);
    std::vector<float> out;
    if (!gl) return out;
    out.assign(gl, gl + n_vocab);
    return out;
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

    const int n_vocab = llama_n_vocab(model);

    // Tokenize prompt once on a throwaway context. Both paths use the
    // SAME prompt token sequence — this is the input to the prefill.
    const std::string prompt = []() {
        const char * env = std::getenv("LLAMA_TEST_PROMPT");
        if (env && *env) return std::string(env);
        // Production-style prompt with enough tokens to fill the
        // DeltaNet conv window and exercise both recurrent and
        // full-attn pathways.
        return std::string(
            "The capital of France is Paris. The capital of Germany is");
    }();

    std::vector<llama_token> prompt_tokens;
    {
        llama_context * tmp = fresh_ctx(model);
        if (!tmp) {
            std::fprintf(stderr, "tmp ctx init failed\n");
            llama_free_model(model);
            return 77;
        }
        prompt_tokens = common_tokenize(tmp, prompt, true, true);
        llama_free(tmp);
    }
    if ((int) prompt_tokens.size() < 4) {
        std::fprintf(stderr, "[restore] prompt too short (%zu tokens)\n",
                     prompt_tokens.size());
        llama_free_model(model);
        return 1;
    }
    const llama_pos P = (llama_pos) prompt_tokens.size();
    std::fprintf(stderr, "[restore] prompt P = %d tokens\n", (int) P);

    // Fixed verify-batch token sequence. Token IDs are chosen to be
    // small + likely-valid for a typical BPE vocab — exact identities
    // don't matter for the byte-identity gate; only that both paths
    // see the SAME first 3 tokens.
    //
    // We use [verify_toks[0..4]] in path B and [verify_toks[0..2]]
    // in path A. The bonus decode in both uses verify_toks[5].
    const std::vector<llama_token> verify_toks = {100, 200, 300, 400, 500};
    const llama_token bonus_tok = 600;

    // -------- Path A: fresh 3-token decode, then 1-token bonus.
    std::vector<float> logits_A;
    {
        llama_context * ctx = fresh_ctx(model);
        if (!ctx) { llama_free_model(model); std::fprintf(stderr, "ctxA fail\n"); return 1; }

        if (decode_tokens(ctx, prompt_tokens, 0, 0) != 0) {
            std::fprintf(stderr, "[A] prefill failed\n");
            llama_free(ctx); llama_free_model(model); return 1;
        }
        // 3-token batch at positions [P..P+2]. spec-ckpt is NEVER armed.
        std::vector<llama_token> three = {verify_toks[0], verify_toks[1], verify_toks[2]};
        if (decode_tokens(ctx, three, P, 0) != 0) {
            std::fprintf(stderr, "[A] 3-token decode failed\n");
            llama_free(ctx); llama_free_model(model); return 1;
        }
        // 1-token bonus at P+3.
        std::vector<llama_token> bonus = {bonus_tok};
        if (decode_tokens(ctx, bonus, P + 3, 0) != 0) {
            std::fprintf(stderr, "[A] bonus decode failed\n");
            llama_free(ctx); llama_free_model(model); return 1;
        }
        logits_A = capture_row0_logits(ctx, n_vocab);
        if (logits_A.empty()) {
            std::fprintf(stderr, "[A] logits NULL\n");
            llama_free(ctx); llama_free_model(model); return 1;
        }
        std::fprintf(stderr, "[A] captured %zu fp32 logits at pos P+3\n",
                     logits_A.size());
        llama_free(ctx);
    }

    // -------- Path B: 5-token verify-style decode w/ spec-ckpt armed,
    //                  per_step_restore(2), then 1-token bonus.
    std::vector<float> logits_B;
    {
        llama_context * ctx = fresh_ctx(model);
        if (!ctx) { llama_free_model(model); std::fprintf(stderr, "ctxB fail\n"); return 1; }

        if (decode_tokens(ctx, prompt_tokens, 0, 0) != 0) {
            std::fprintf(stderr, "[B] prefill failed\n");
            llama_free(ctx); llama_free_model(model); return 1;
        }

        // Arm PER_STEP checkpoint sized for the verify batch (5 tokens).
        const int mode = llama_spec_ckpt_init(ctx, LLAMA_SPEC_CKPT_PER_STEP,
                                              /*max_tokens=*/(int) verify_toks.size());
        if (mode != LLAMA_SPEC_CKPT_PER_STEP) {
            std::fprintf(stderr,
                         "[B] spec_ckpt_init returned mode=%d (expected PER_STEP=%d); "
                         "model lacks per-step support — SKIP\n",
                         mode, (int) LLAMA_SPEC_CKPT_PER_STEP);
            llama_free(ctx); llama_free_model(model); llama_backend_free();
            return 77;
        }
        if (!llama_spec_ckpt_save(ctx, /*seq_id=*/0)) {
            std::fprintf(stderr, "[B] spec_ckpt_save failed\n");
            llama_free(ctx); llama_free_model(model); return 1;
        }

        // 5-token verify-style batch at positions [P..P+4].
        if (decode_tokens(ctx, verify_toks, P, 0) != 0) {
            std::fprintf(stderr, "[B] verify-batch decode failed\n");
            llama_free(ctx); llama_free_model(model); return 1;
        }

        // Restore the state to "after [t0..t2]" (accepted_step=2 means
        // 3 tokens accepted out of 5). cells[0].pos becomes P+2; cells
        // past P+2 are seq_rm'd.
        if (!llama_spec_ckpt_restore(ctx, /*seq_id=*/0, /*n_past=*/P,
                                     /*accepted_step=*/2)) {
            std::fprintf(stderr, "[B] spec_ckpt_restore failed\n");
            llama_free(ctx); llama_free_model(model); return 1;
        }

        // 1-token bonus at position P+3.
        std::vector<llama_token> bonus = {bonus_tok};
        if (decode_tokens(ctx, bonus, P + 3, 0) != 0) {
            std::fprintf(stderr, "[B] bonus decode failed\n");
            llama_free(ctx); llama_free_model(model); return 1;
        }
        logits_B = capture_row0_logits(ctx, n_vocab);
        if (logits_B.empty()) {
            std::fprintf(stderr, "[B] logits NULL\n");
            llama_free(ctx); llama_free_model(model); return 1;
        }
        std::fprintf(stderr, "[B] captured %zu fp32 logits at pos P+3\n",
                     logits_B.size());

        llama_spec_ckpt_discard(ctx);
        llama_free(ctx);
    }

    if (logits_A.size() != logits_B.size()) {
        std::fprintf(stderr, "[FAIL] logits sizes differ: A=%zu B=%zu\n",
                     logits_A.size(), logits_B.size());
        llama_free_model(model); llama_backend_free();
        return 1;
    }

    // Byte-identity check.
    int n_diff = 0;
    int first_diff = -1;
    float max_abs = 0.0f;
    for (size_t i = 0; i < logits_A.size(); ++i) {
        uint32_t a, b;
        std::memcpy(&a, &logits_A[i], 4);
        std::memcpy(&b, &logits_B[i], 4);
        if (a != b) {
            if (first_diff < 0) first_diff = (int) i;
            ++n_diff;
            const float d = std::abs(logits_A[i] - logits_B[i]);
            if (d > max_abs) max_abs = d;
        }
    }

    // Also compare argmax — the CLI-visible decision.
    int argmax_A = 0, argmax_B = 0;
    {
        float vA = logits_A[0], vB = logits_B[0];
        for (size_t i = 1; i < logits_A.size(); ++i) {
            if (logits_A[i] > vA) { vA = logits_A[i]; argmax_A = (int) i; }
            if (logits_B[i] > vB) { vB = logits_B[i]; argmax_B = (int) i; }
        }
    }

    int rc = 0;
    if (n_diff == 0) {
        std::printf("[PASS] per_step_restore byte-identity: bonus-decode logits "
                    "at pos P+3 are byte-identical between fresh-3-decode path "
                    "(A) and 5-token-verify + restore(2) path (B). %zu fp32 "
                    "floats match. argmax A=%d B=%d.\n",
                    logits_A.size(), argmax_A, argmax_B);
    } else {
        std::printf("[FAIL] per_step_restore NOT byte-identical to fresh decode: "
                    "%d/%zu logit floats differ (max |Δ|=%.3e, first diff at "
                    "vocab idx %d: A=%+.6f B=%+.6f). argmax A=%d B=%d %s\n",
                    n_diff, logits_A.size(), max_abs, first_diff,
                    logits_A[first_diff], logits_B[first_diff],
                    argmax_A, argmax_B,
                    argmax_A == argmax_B ? "(argmax agrees — drift below "
                                           "argmax-flip threshold)"
                                         : "(argmax FLIPS)");
        rc = 1;
    }

    llama_free_model(model);
    llama_backend_free();
    return rc;
}
