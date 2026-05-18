// test-dflash-np-multislot.cpp
//
// DFlash Phase 6 multi-slot harness. Drives the full
// llama_dflash_draft_batch pipeline at NP ∈ {1, 2, 4, 8} with
// identical per-seq content, asserts slot-0 byte-identity across NP
// (orchestrator-level np-invariance complementing T7's kernel-level
// probe), and reports per-NP dispatch latency and aggregate t/s as a
// measurement of record for the multi-slot decode path.
//
// Binding gates:
//   Gate 2 (per Phase 6 plan): np=1 → np=8 byte-identical per slot.
//   Gate 3 quality: every NP returns non-zero candidates.
//
// Env:
//   LLAMA_TEST_TARGET  — target GGUF (skip 77 if unset)
//   LLAMA_TEST_DRAFTER — drafter GGUF (skip 77 if unset)
//   LLAMA_TEST_CYCLES  — number of draft cycles to time per NP (default 16)

#include "common.h"
#include "llama.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static bool prefill_n_seqs(llama_context * ctx, int n_slots,
                           const std::vector<llama_token> & tokens) {
    const int n = (int) tokens.size();
    for (int sid = 0; sid < n_slots; ++sid) llama_kv_cache_seq_rm(ctx, sid, -1, -1);
    llama_batch batch = llama_batch_init(n_slots * n, 0, 1);
    for (int sid = 0; sid < n_slots; ++sid) {
        for (int i = 0; i < n; ++i) {
            const bool last = (i == n - 1);
            common_batch_add(batch, tokens[i], i, {(llama_seq_id) sid}, last);
        }
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
    const int n_cycles = std::getenv("LLAMA_TEST_CYCLES") ? std::atoi(std::getenv("LLAMA_TEST_CYCLES")) : 16;

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    mparams.split_mode   = LLAMA_SPLIT_MODE_GRAPH;
    static const char * dev_csv = "CUDA0,CUDA1";
    mparams.devices = dev_csv;
    llama_model * model = llama_model_load_from_file(target_path, mparams);
    if (!model) { fprintf(stderr, "load target failed: %s\n", target_path); return 77; }

    const int n_slots_cap = 8;
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx       = 4096 * 2;
    cparams.n_batch     = 2048;
    cparams.n_ubatch    = 2048;
    cparams.n_seq_max   = n_slots_cap;
    cparams.type_k      = GGML_TYPE_Q4_0;
    cparams.type_v      = GGML_TYPE_Q4_0;
    cparams.flash_attn  = true;
    cparams.mla_attn    = 3;
    cparams.k_cache_hadamard = true;
    cparams.v_cache_hadamard = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { llama_free_model(model); fprintf(stderr, "ctx init failed\n"); return 77; }

    llama_dflash_drafter * drafter = llama_dflash_drafter_load(drafter_path);
    if (!drafter) { fprintf(stderr, "drafter load failed\n"); return 77; }
    if (llama_set_dflash(ctx, drafter) != LLAMA_DFLASH_OK) {
        fprintf(stderr, "llama_set_dflash failed\n");
        return 1;
    }
    const int32_t BS_max = llama_dflash_block_size(drafter);

    // Single prompt content replicated across all seqs — gives every
    // slot byte-identical cb_eval extract input, so slot-0 output at
    // NP=N must match slot-0 output at NP=1.
    const std::string prompt = "The capital of France is Paris.";
    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true, true);
    const int n_prompt = (int) tokens.size();
    if (n_prompt < 2) {
        fprintf(stderr, "[multislot] prompt too short: %d\n", n_prompt);
        return 1;
    }
    const llama_token anchor_id = tokens.back();
    const int32_t     anchor_pos = n_prompt - 1;

    const std::vector<int> NPs = {1, 2, 4, 8};
    std::vector<llama_token> slot0_ref;
    int total_failures = 0;

    fprintf(stderr, "[multislot] n_cycles=%d  BS_max=%d  prompt_len=%d\n",
            n_cycles, BS_max, n_prompt);

    for (int N : NPs) {
        if (!prefill_n_seqs(ctx, N, tokens)) {
            fprintf(stderr, "[FAIL] prefill N=%d failed\n", N);
            return 1;
        }

        std::vector<llama_token>   anchor_ids(N, anchor_id);
        std::vector<int32_t>       anchor_ps (N, anchor_pos);
        std::vector<llama_seq_id>  seq_ids   (N);
        for (int i = 0; i < N; ++i) seq_ids[i] = (llama_seq_id) i;
        std::vector<llama_token>   flat_out  ((size_t) N * (size_t) BS_max, 0);

        // Warm-up call (lazy GPU graph capture, alloc warmup).
        const int32_t rc_warm = llama_dflash_draft_batch(
                ctx, N, anchor_ids.data(), anchor_ps.data(), seq_ids.data(),
                flat_out.data(), (int32_t) flat_out.size());
        if (rc_warm < 0) {
            fprintf(stderr, "[FAIL] N=%d warm-up rc=%d\n", N, rc_warm);
            return 1;
        }
        const int32_t per_slot = rc_warm / N;
        if (per_slot <= 0) {
            fprintf(stderr, "[FAIL] N=%d per_slot=%d (rc=%d)\n", N, per_slot, rc_warm);
            return 1;
        }

        // Capture slot 0's first-cycle output for cross-NP byte-identity.
        std::vector<llama_token> slot0(flat_out.begin(),
                                       flat_out.begin() + per_slot);
        if (N == 1) {
            slot0_ref = slot0;
        } else {
            int diffs = 0;
            for (int b = 0; b < per_slot && b < (int) slot0_ref.size(); ++b) {
                if (slot0[b] != slot0_ref[b]) diffs++;
            }
            if (diffs == 0) {
                fprintf(stderr, "[multislot] N=%d slot-0 BYTE-IDENTICAL to N=1 (per_slot=%d)\n",
                        N, per_slot);
            } else {
                fprintf(stderr, "[FAIL] N=%d slot-0 differs from N=1: %d/%d candidate mismatches\n",
                        N, diffs, per_slot);
                fprintf(stderr, "       ref: ");
                for (int b = 0; b < per_slot; ++b) fprintf(stderr, "%d ", slot0_ref[b]);
                fprintf(stderr, "\n       got: ");
                for (int b = 0; b < per_slot; ++b) fprintf(stderr, "%d ", slot0[b]);
                fprintf(stderr, "\n");
                total_failures++;
            }
        }

        // Timed loop: n_cycles draft cycles back-to-back; same anchor
        // each time. Measures dispatch-only latency at this NP.
        const auto t0 = std::chrono::steady_clock::now();
        for (int c = 0; c < n_cycles; ++c) {
            const int32_t rc = llama_dflash_draft_batch(
                    ctx, N, anchor_ids.data(), anchor_ps.data(), seq_ids.data(),
                    flat_out.data(), (int32_t) flat_out.size());
            if (rc < 0) {
                fprintf(stderr, "[FAIL] N=%d cycle %d rc=%d\n", N, c, rc);
                return 1;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double per_cycle_ms = ms / (double) n_cycles;
        const double tokens_emitted = (double) n_cycles * (double) N * (double) per_slot;
        const double tps_total = tokens_emitted / (ms / 1000.0);
        const double tps_per_slot = tps_total / (double) N;
        fprintf(stderr,
                "[multislot] N=%d  cycles=%d  per_cycle=%.3f ms  tokens_emitted=%.0f  "
                "aggregate=%.1f tok/s  per_slot=%.1f tok/s\n",
                N, n_cycles, per_cycle_ms, tokens_emitted, tps_total, tps_per_slot);
    }

    llama_set_dflash(ctx, nullptr);
    llama_dflash_drafter_free(drafter);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    if (total_failures == 0) {
        std::printf("[PASS] DFlash multi-slot harness: slot-0 byte-identical across NP ∈ {1,2,4,8}\n");
        return 0;
    }
    std::printf("[FAIL] DFlash multi-slot harness: %d slot-0 byte-identity break(s)\n", total_failures);
    return 1;
}
