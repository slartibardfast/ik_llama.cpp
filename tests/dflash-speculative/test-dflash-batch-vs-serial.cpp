// test-dflash-batch-vs-serial.cpp
//
// DFlash multi-slot Phase 4 binding gate: confirms that
// `llama_dflash_draft_batch(n_slots=2, seq_ids=[0,1])` produces
// byte-identical candidates to two serial `n_slots=1` calls — one with
// seq_id=0 and one with seq_id=1 — over the same prefilled state.
//
// Binding semantics:
//   A0 = draft_batch(n_slots=1, seq_id=0)   — 4 candidates for seq 0
//   A1 = draft_batch(n_slots=1, seq_id=1)   — 4 candidates for seq 1
//   B  = draft_batch(n_slots=2, [0,1])      — 8 candidates, slot-major
//   PASS iff B[0..4) == A0 AND B[4..8) == A1
//
// The serial calls already trim the per-seq extract buffer to MAL rows
// each via stage_target_hiddens, so the third call re-reads the same
// preserved rows. inject_kv_fused is idempotent at the same anchor
// positions, so re-injection across calls doesn't perturb the cache.
//
// Env:
//   LLAMA_TEST_TARGET  — target GGUF (skip 77 if unset)
//   LLAMA_TEST_DRAFTER — drafter GGUF (skip 77 if unset)

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

    // Load drafter weights and bind to ctx — installs cb_eval hook and
    // allocates per-slot scratch sized to cparams.n_seq_max.
    llama_dflash_drafter * drafter = llama_dflash_drafter_load(drafter_path);
    if (!drafter) { fprintf(stderr, "drafter load failed: %s\n", drafter_path); return 77; }
    if (llama_set_dflash(ctx, drafter) != LLAMA_DFLASH_OK) {
        fprintf(stderr, "llama_set_dflash failed\n");
        return 1;
    }

    // Prefill the same prompt on both seq_ids — gives matching per-seq
    // residual streams in the cb_eval per-seq buffers.
    const std::string prompt = "The capital of France is Paris.";
    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true, true);
    const int n_prompt = (int) tokens.size();
    if (n_prompt < 2) {
        fprintf(stderr, "[phase4-batch-vs-serial] prompt too short: %d\n", n_prompt);
        return 1;
    }

    llama_kv_cache_seq_rm(ctx, 0, -1, -1);
    llama_kv_cache_seq_rm(ctx, 1, -1, -1);
    {
        llama_batch batch = llama_batch_init(2 * n_prompt, 0, 1);
        for (int sid = 0; sid < 2; ++sid) {
            for (int i = 0; i < n_prompt; ++i) {
                const bool last = (i == n_prompt - 1);
                common_batch_add(batch, tokens[i], i, {(llama_seq_id) sid}, last);
            }
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[phase4-batch-vs-serial] prefill decode FAILED\n");
            llama_batch_free(batch);
            return 1;
        }
        llama_batch_free(batch);
    }

    const llama_token anchor_id = tokens.back();
    const int32_t anchor_pos    = n_prompt - 1;
    const int32_t BS            = 4;

    // ── A0: serial n_slots=1, seq_id=0 ──
    std::vector<llama_token> A0(BS, -1);
    {
        const llama_seq_id sid = 0;
        int32_t got = llama_dflash_draft_batch(ctx, 1, &anchor_id, &anchor_pos,
                                                &sid, A0.data(), BS);
        if (got != BS) {
            fprintf(stderr, "[FAIL] A0 draft_batch(n_slots=1, seq_id=0) returned %d\n", got);
            return 1;
        }
    }

    // ── A1: serial n_slots=1, seq_id=1 ──
    std::vector<llama_token> A1(BS, -1);
    {
        const llama_seq_id sid = 1;
        int32_t got = llama_dflash_draft_batch(ctx, 1, &anchor_id, &anchor_pos,
                                                &sid, A1.data(), BS);
        if (got != BS) {
            fprintf(stderr, "[FAIL] A1 draft_batch(n_slots=1, seq_id=1) returned %d\n", got);
            return 1;
        }
    }

    // ── B: multi-slot n_slots=2, [seq 0, seq 1] ──
    std::vector<llama_token> B(2 * BS, -1);
    {
        llama_token   anchor_ids[2] = {anchor_id, anchor_id};
        int32_t       anchor_ps[2]  = {anchor_pos, anchor_pos};
        llama_seq_id  sids[2]       = {0, 1};
        int32_t got = llama_dflash_draft_batch(ctx, 2, anchor_ids, anchor_ps,
                                                sids, B.data(), 2 * BS);
        if (got != 2 * BS) {
            fprintf(stderr, "[FAIL] B draft_batch(n_slots=2) returned %d\n", got);
            return 1;
        }
    }

    // ── Compare ──
    int failures = 0;
    fprintf(stderr, "[phase4-batch-vs-serial] A0: %d %d %d %d\n", A0[0], A0[1], A0[2], A0[3]);
    fprintf(stderr, "[phase4-batch-vs-serial] A1: %d %d %d %d\n", A1[0], A1[1], A1[2], A1[3]);
    fprintf(stderr, "[phase4-batch-vs-serial] B0: %d %d %d %d\n", B[0], B[1], B[2], B[3]);
    fprintf(stderr, "[phase4-batch-vs-serial] B1: %d %d %d %d\n", B[4], B[5], B[6], B[7]);
    for (int b = 0; b < BS; ++b) {
        if (A0[b] != B[b]) {
            fprintf(stderr, "[FAIL] slot 0 candidate %d: serial=%d batch=%d\n",
                    b, A0[b], B[b]);
            failures++;
        }
        if (A1[b] != B[BS + b]) {
            fprintf(stderr, "[FAIL] slot 1 candidate %d: serial=%d batch=%d\n",
                    b, A1[b], B[BS + b]);
            failures++;
        }
    }

    if (failures == 0) {
        std::printf("[PASS] DFlash Phase 4 batch-vs-serial: n_slots=2 byte-identical to two serial n_slots=1 calls (BS=%d × 2 slots)\n", BS);
    } else {
        std::printf("[FAIL] %d candidate mismatch(es)\n", failures);
    }

    llama_set_dflash(ctx, nullptr);
    llama_dflash_drafter_free(drafter);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return failures == 0 ? 0 : 1;
}
