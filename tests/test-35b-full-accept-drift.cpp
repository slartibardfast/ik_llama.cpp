// Full-accept drift probe.
//
// Complement to test-35b-trajectory-drift. That test uses the always-reject
// pattern (batch=[cur, DUMMY], reject DUMMY, rollback to token_idx=0). This
// test uses the always-accept-BOTH pattern (batch=[cur, predicted_next],
// accept both, NO rollback). The SSM/delta-net state is left at "end of
// batch=2 decode" and the next cycle proceeds from there.
//
// Trajectory A (ground truth): batch=1 greedy.
// Trajectory B (full-accept spec): each cycle, use argmax-of-batch-pos-0 as
// the draft, so the spec always accepts both. Equivalent to a speculator
// that always guesses correctly. Output tokens = ids[0] + ids[1] per cycle.
//
// PASS iff traj_A == traj_B byte-identical.
// FAIL means the batch=2 end-state diverges from sequential batch=1
// end-state, and that divergence propagates through subsequent cycles.
//
// Unlike trajectory-drift, this test does NOT call rollback — the point is
// to isolate "does continuous batch=2 decoding (no rollback) drift vs
// sequential batch=1".
//
// Usage:
//   test-35b-full-accept-drift -m <model.gguf> [--prompt "..."] [--n-predict 20]

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static llama_token argmax(const float * logits, int n_vocab) {
    llama_token best = 0;
    float best_l = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_l) { best_l = logits[i]; best = i; }
    }
    return best;
}

int main(int argc, char ** argv) {
    gpt_params params;
    params.prompt = "The capital of France is";

    int n_predict = 20;
    std::vector<char *> forward;
    forward.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n-predict") == 0 && i + 1 < argc) {
            n_predict = std::atoi(argv[++i]);
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

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token> prompt = ::common_tokenize(ctx, params.prompt, true);
    const int n_prompt = (int) prompt.size();
    fprintf(stderr, "Prompt: '%s' (%d tokens), n_predict=%d\n",
            params.prompt.c_str(), n_prompt, n_predict);

    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        common_batch_add(batch, prompt[i], i, {0}, i == n_prompt - 1);
    }
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Prompt decode failed\n"); return 1; }

    const llama_token first = argmax(
            llama_get_logits_ith(ctx, batch.n_tokens - 1), n_vocab);

    const size_t snap_size = llama_state_seq_get_size(ctx, 0, 0);
    std::vector<uint8_t> snapshot(snap_size);
    if (llama_state_seq_get_data(ctx, snapshot.data(), snap_size, 0, 0) == 0) {
        fprintf(stderr, "Snapshot failed\n"); return 1;
    }

    auto restore = [&]() {
        llama_kv_cache_clear(ctx);
        llama_state_seq_set_data(ctx, snapshot.data(), snap_size, 0, 0);
    };

    // --- Trajectory A: batch=1 greedy ground truth ---
    restore();
    std::vector<llama_token> traj_A;
    traj_A.push_back(first);
    {
        llama_token cur = first;
        for (int i = 1; i < n_predict; i++) {
            common_batch_clear(batch);
            common_batch_add(batch, cur, n_prompt + (i - 1), {0}, true);
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "A step %d decode failed\n", i); return 1; }
            cur = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
            traj_A.push_back(cur);
        }
    }

    // --- Trajectory B: full-accept spec, no rollback ---
    //
    // To make the spec always accept, we use the ORACLE draft: for each
    // cycle, the draft is simply the next token from traj_A. That way S0 ==
    // draft always, and both get accepted. This isolates "batch=2 end-state
    // drift" without confounding with bad drafts causing rejects.
    restore();
    std::vector<llama_token> traj_B;
    traj_B.push_back(first);
    {
        llama_token sampled = first;
        int n_past = n_prompt;
        int out_idx = 1;  // next traj_A index we need to cover
        while ((int) traj_B.size() < n_predict) {
            // Oracle draft from traj_A. If we've already covered all, stop.
            if (out_idx >= (int) traj_A.size()) break;
            const llama_token draft = traj_A[out_idx];

            common_batch_clear(batch);
            common_batch_add(batch, sampled, n_past,     {0}, true);
            common_batch_add(batch, draft,   n_past + 1, {0}, true);
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "B verify decode failed\n"); return 1; }

            const llama_token S0 = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
            const llama_token S1 = argmax(llama_get_logits_ith(ctx, 1), n_vocab);

            // Oracle-guaranteed S0 == draft (== traj_A[out_idx]) — if not,
            // we've already diverged at step out_idx. Record and continue
            // with what we sampled (so downstream divergences can be seen).
            if ((int) traj_B.size() < n_predict) traj_B.push_back(S0);
            if ((int) traj_B.size() < n_predict) traj_B.push_back(S1);

            // Next cycle starts with S1 as the sampled token. n_past
            // advances by 2 (both accepted, no rollback, no seq_rm needed).
            sampled = S1;
            n_past += 2;
            out_idx += 2;
        }
    }

    // --- Compare ---
    int first_diff = -1;
    for (int i = 0; i < n_predict; i++) {
        if (traj_A[i] != traj_B[i]) { first_diff = i; break; }
    }

    printf("\n%-4s %-24s %-24s %s\n", "step", "ground-truth (A)", "full-accept (B)", "");
    for (int i = 0; i < n_predict; i++) {
        const char * marker = (traj_A[i] == traj_B[i]) ? "SAME" : "DIFF";
        printf("%-4d %-24s %-24s %s\n", i,
               common_token_to_piece(ctx, traj_A[i]).c_str(),
               common_token_to_piece(ctx, traj_B[i]).c_str(),
               marker);
    }
    if (first_diff < 0) {
        printf("\nPASS: full-accept trajectory byte-identical over %d steps\n", n_predict);
    } else {
        printf("\nFAIL: first divergence at step %d ('%s' vs '%s')\n",
               first_diff,
               common_token_to_piece(ctx, traj_A[first_diff]).c_str(),
               common_token_to_piece(ctx, traj_B[first_diff]).c_str());
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return first_diff < 0 ? 0 : 1;
}
