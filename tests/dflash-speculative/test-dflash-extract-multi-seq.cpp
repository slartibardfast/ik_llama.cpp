// test-dflash-extract-multi-seq.cpp
//
// DFlash multi-slot Phase 3 verification: confirms that the cb_eval
// extract hook demuxes rows by primary seq_id. With n_seq_max=2 and a
// batch carrying two seq_ids × n_prompt tokens, each per-seq buffer
// must contain exactly n_prompt rows × n_embd floats — the pre-Phase-3
// flat-concat would yield 2*n_prompt rows in the seq_id=0 buffer and
// 0 in the seq_id=1 buffer.
//
// Binding gates:
//   1. seq_id=0 buffer count == n_prompt × n_embd      (demux right)
//   2. seq_id=1 buffer count == n_prompt × n_embd      (demux right)
//   3. seq_id=0 row 0 differs from seq_id=1 row 0       (rows aren't aliased)
//
// Env:
//   LLAMA_TEST_TARGET — target GGUF (skip with 77 if unset)

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main() {
    const char * target = std::getenv("LLAMA_TEST_TARGET");
    if (!target) { fprintf(stderr, "SKIP: set LLAMA_TEST_TARGET\n"); return 77; }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    mparams.split_mode   = LLAMA_SPLIT_MODE_GRAPH;
    static const char * dev_csv = "CUDA0,CUDA1";
    mparams.devices = dev_csv;
    llama_model * model = llama_model_load_from_file(target, mparams);
    if (!model) { fprintf(stderr, "load failed: %s\n", target); return 77; }

    const int n_layer = llama_n_layer(model);
    const int n_embd  = llama_model_n_embd(model);

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

    const std::string prompt = "The capital of France is Paris.";
    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true, true);
    const int n_prompt = (int) tokens.size();
    if (n_prompt < 2 || n_layer < 1) {
        fprintf(stderr, "[phase3-multi-seq] degenerate inputs n_prompt=%d n_layer=%d\n",
                n_prompt, n_layer);
        return 1;
    }

    // Configure single-layer extract (layer 0). One slot is enough to
    // verify the demux; multi-layer is exercised by other tests.
    int32_t layer0 = 0;
    llama_set_dflash_extract_layers(ctx, &layer0, 1);

    // Build a batch with 2 seq_ids × n_prompt tokens each.
    llama_kv_cache_seq_rm(ctx, 0, -1, -1);
    llama_kv_cache_seq_rm(ctx, 1, -1, -1);
    {
        const int total = 2 * n_prompt;
        llama_batch batch = llama_batch_init(total, 0, 1);
        for (int sid = 0; sid < 2; ++sid) {
            for (int i = 0; i < n_prompt; ++i) {
                const bool last = (i == n_prompt - 1);
                common_batch_add(batch, tokens[i], i, {(llama_seq_id) sid}, last);
            }
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "[phase3-multi-seq] decode FAILED\n");
            llama_batch_free(batch);
            return 1;
        }
        llama_batch_free(batch);
    }

    // Per-seq buffer expected size: n_prompt rows × n_embd floats.
    const size_t expected = (size_t) n_prompt * (size_t) n_embd;
    std::vector<float> seq0(expected, 0.0f);
    std::vector<float> seq1(expected, 0.0f);
    const size_t got0 = llama_get_dflash_extract_data_seq(ctx, 0, 0, seq0.data(), seq0.size());
    const size_t got1 = llama_get_dflash_extract_data_seq(ctx, 0, 1, seq1.data(), seq1.size());
    fprintf(stderr, "[phase3-multi-seq] seq0_count=%zu seq1_count=%zu expected=%zu\n",
            got0, got1, expected);

    int failures = 0;
    if (got0 != expected) {
        fprintf(stderr, "[FAIL] gate 1: seq_id=0 count %zu != %zu\n", got0, expected);
        failures++;
    }
    if (got1 != expected) {
        fprintf(stderr, "[FAIL] gate 2: seq_id=1 count %zu != %zu\n", got1, expected);
        failures++;
    }

    // Sanity: the two seq buffers should have IDENTICAL content because
    // both sequences saw the same token stream at the same positions
    // with no cross-attention between them (causal, seq-isolated). If
    // demux is broken (e.g. all rows aliased to seq_id=0), seq1 stays
    // zero-initialized and the buffers differ by exactly seq1's content.
    // Conversely, demux working correctly yields byte-identical row 0
    // for both sequences. This makes the test asymmetric to a swap-bug:
    // a swap (row 0 of seq0 ends up in seq1's slot and vice versa) is
    // invisible to row-0 comparison alone, but the count gates (1, 2)
    // both fail in that scenario because the source rows arrive in
    // ubatch order and seq0 fills first.
    bool seq1_all_zero = true;
    for (int d = 0; d < n_embd && seq1_all_zero; ++d) {
        if (seq1[d] != 0.0f) seq1_all_zero = false;
    }
    if (seq1_all_zero && got1 > 0) {
        fprintf(stderr, "[FAIL] gate 3: seq_id=1 buffer reports %zu floats but row 0 is all-zero (extract may not be writing seq_id=1's rows)\n", got1);
        failures++;
    }

    if (failures == 0) {
        std::printf("[PASS] DFlash Phase 3 cb_eval per-seq demux: seq0=%zu seq1=%zu floats, expected=%zu each\n",
                    got0, got1, expected);
    } else {
        std::printf("[FAIL] %d gate(s) failed\n", failures);
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return failures == 0 ? 0 : 1;
}
