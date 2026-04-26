// Regression for `llama_get_mtp_draft_token_at(ctx, i)`.
//
// Asserts for every valid position i in a multi-token decode batch:
//   argmax(llama_get_mtp_logits_ith(ctx, i))  ==  llama_get_mtp_draft_token_at(ctx, i)
// (modulo EOG tokens, which llama_get_mtp_draft_token_at should return
// LLAMA_TOKEN_NULL for — tested separately.)
//
// Catches regressions where the "at i" API reads the wrong position, wrong
// stride, or the wrong buffer.
//
// Usage:
//   test-35b-mtp-draft-token-at -m <model.gguf> [--batch-size 4] [-mtp]

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// Mirror llama_get_mtp_draft_token_at's exact logic: argmax across ALL
// positions (no EOG skip), then return LLAMA_TOKEN_NULL if best is EOG.
static llama_token ref_draft_at(const float * logits, int64_t n_vocab,
                                const struct llama_vocab * vocab) {
    llama_token best = 0;
    float best_l = logits[0];
    for (int64_t j = 1; j < n_vocab; j++) {
        if (logits[j] > best_l) { best_l = logits[j]; best = (llama_token)j; }
    }
    if (llama_vocab_is_eog(vocab, best)) return LLAMA_TOKEN_NULL;
    return best;
}

int main(int argc, char ** argv) {
    gpt_params params;
    params.prompt = "The capital of France is";

    int batch_size = 4;
    std::vector<char *> forward;
    forward.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            batch_size = std::atoi(argv[++i]);
        } else {
            forward.push_back(argv[i]);
        }
    }

    if (!gpt_params_parse((int) forward.size(), forward.data(), params)) {
        fprintf(stderr, "Failed to parse params\n");
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);
    llama_init_result init = llama_init_from_gpt_params(params);
    llama_model   * model  = init.model;
    llama_context * ctx    = init.context;
    if (!model || !ctx) { fprintf(stderr, "Load failed\n"); return 1; }

    const int vocab_n = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    // MTP availability check.
    if (llama_model_n_nextn_layer(model) == 0) {
        fprintf(stderr, "SKIP: model has no NextN (MTP) layer\n");
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 0;
    }

    std::vector<llama_token> prompt = ::common_tokenize(ctx, params.prompt, true);
    const int n_prompt = (int) prompt.size();
    fprintf(stderr, "Prompt: '%s' (%d tokens), batch_size=%d\n",
            params.prompt.c_str(), n_prompt, batch_size);

    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        common_batch_add(batch, prompt[i], i, {0}, i == n_prompt - 1);
    }
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Prompt decode failed\n"); return 1; }

    // Sample some tokens to build up context, then do an N-token decode.
    std::vector<llama_token> toks;
    toks.reserve(batch_size);
    std::mt19937 rng(0xBADBEEF);
    for (int k = 0; k < batch_size; k++) {
        llama_token t;
        do {
            t = (llama_token)(rng() % vocab_n);
        } while (llama_vocab_is_eog(vocab, t));
        toks.push_back(t);
    }

    common_batch_clear(batch);
    for (int i = 0; i < batch_size; i++) {
        common_batch_add(batch, toks[i], n_prompt + i, {0}, true);
    }
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "N-token decode failed\n"); return 1; }

    // Check each position: at-i API must match argmax-ith.
    int mismatches = 0;
    printf("\n%-5s %-12s %-12s %s\n", "pos", "argmax(ith)", "draft_at(i)", "match");
    for (int i = 0; i < batch_size; i++) {
        const float * log_i = llama_get_mtp_logits_ith(ctx, i);
        if (log_i == nullptr) {
            fprintf(stderr, "FAIL: llama_get_mtp_logits_ith(%d) returned nullptr\n", i);
            mismatches++;
            continue;
        }
        const int64_t mtp_vocab   = llama_get_mtp_n_vocab(ctx);
        const llama_token expected = ref_draft_at(log_i, mtp_vocab, vocab);
        const llama_token actual   = llama_get_mtp_draft_token_at(ctx, i);
        const bool ok = (actual == expected);
        if (!ok) mismatches++;
        printf("%-5d %-12d %-12d %s\n", i, expected, actual, ok ? "SAME" : "DIFF");
    }

    printf("\nmismatches = %d / %d\n", mismatches, batch_size);
    printf("%s\n", mismatches == 0 ? "PASS" : "FAIL");

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return mismatches == 0 ? 0 : 1;
}
