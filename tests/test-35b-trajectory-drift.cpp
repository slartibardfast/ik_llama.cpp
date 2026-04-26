// Multi-step trajectory drift probe.
//
// Compares two decode trajectories from the same post-prompt state:
//   A (no-spec)        : for i=1..N, decode batch=[cur] at pos n_past+(i-1),
//                         cur = argmax(main_logits[0]).
//   B (spec-always-rej): for i=1..N, decode batch=[cur, DUMMY] at pos i-1, i,
//                         cur = argmax(main_logits[0]),
//                         seq_rm DUMMY position (always-reject semantics).
//
// Both commit the same position-0 argmax per step. A correct batch=2
// implementation must give byte-identical trajectories. Any divergence
// means batch=2 writes position-(i-1) KV slightly different from batch=1,
// and per-step errors compound — which is what breaks speculation on 35B
// MoE even when single-step argmax is stable (verified in
// test-35b-batch-invariance-sweep).
//
// PASS iff trajectory_A == trajectory_B over N steps.
// Reports the divergence step and the diverging tokens.
//
// Usage:
//   test-35b-trajectory-drift -m <model.gguf> [-n-predict 20] [-c 256 -b 16 -ub 16]
//
// NOTE on -mtp: if set, inline MTP graph nodes run at every decode. This is
// the setting the server uses. Compare with-mtp vs without-mtp to isolate
// whether MTP compute itself contributes to drift.

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
        if (logits[i] > best_l) {
            best_l = logits[i];
            best = i;
        }
    }
    return best;
}

int main(int argc, char ** argv) {
    gpt_params params;
    params.prompt  = "The capital of France is";

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
    llama_model   * model = init.model;
    llama_context * ctx   = init.context;
    if (!model || !ctx) { fprintf(stderr, "Load failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token> prompt = ::common_tokenize(ctx, params.prompt, true);
    const int n_prompt = (int) prompt.size();
    fprintf(stderr, "Prompt: '%s' (%d tokens), n_predict=%d\n",
            params.prompt.c_str(), n_prompt, n_predict);

    // Prompt eval.
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        common_batch_add(batch, prompt[i], i, {0}, i == n_prompt - 1);
    }
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Prompt decode failed\n"); return 1; }

    const llama_token first = argmax(
            llama_get_logits_ith(ctx, batch.n_tokens - 1), n_vocab);

    // Snapshot post-prompt state.
    const size_t snap_size = llama_state_seq_get_size(ctx, 0, 0);
    std::vector<uint8_t> snapshot(snap_size);
    if (llama_state_seq_get_data(ctx, snapshot.data(), snap_size, 0, 0) == 0) {
        fprintf(stderr, "Snapshot failed\n"); return 1;
    }
    fprintf(stderr, "Snapshot: %zu bytes. First sampled token = %d ('%s')\n\n",
            snap_size, first, common_token_to_piece(ctx, first).c_str());

    auto restore = [&]() {
        llama_kv_cache_clear(ctx);
        llama_state_seq_set_data(ctx, snapshot.data(), snap_size, 0, 0);
    };

    // --- Trajectory A: no-spec ---
    restore();
    std::vector<llama_token> traj_A;
    traj_A.push_back(first);
    llama_token cur = first;
    for (int i = 1; i < n_predict; i++) {
        common_batch_clear(batch);
        common_batch_add(batch, cur, n_prompt + (i - 1), {0}, true);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "A decode failed at step %d\n", i); return 1; }
        cur = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
        traj_A.push_back(cur);
    }

    // --- Trajectory B: spec-always-reject ---
    restore();
    std::vector<llama_token> traj_B;
    traj_B.push_back(first);
    cur = first;
    // Pick a stable dummy draft — avoid EOG, avoid being same as any plausible
    // argmax. Token id 0 is often BOS or padding; using a mid-vocab token.
    llama_token DUMMY = (llama_token) (n_vocab / 2);
    if (llama_vocab_is_eog(llama_model_get_vocab(model), DUMMY)) DUMMY++;

    for (int i = 1; i < n_predict; i++) {
        common_batch_clear(batch);
        common_batch_add(batch, cur,   n_prompt + (i - 1), {0}, true);
        common_batch_add(batch, DUMMY, n_prompt + i,       {0}, true);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "B decode failed at step %d\n", i); return 1; }
        cur = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
        traj_B.push_back(cur);
        // Always-reject: on hybrid models, seq_rm alone does not restore
        // the SSM recurrent state (it only adjusts KV-cell metadata and
        // refuses partial erasure for recurrent caches). Use the MTP-IR
        // rollback API to restore per-token-intermediate state back to
        // token_idx=0 (= state after only the first batched token, i.e.
        // `cur` committed, DUMMY rejected).
        const bool rollback_ok = llama_rollback_delta_net_state(
                ctx, /*token_idx=*/0, /*seq_id=*/0, /*target_pos=*/n_prompt + (i - 1));
        if (!rollback_ok) {
            // No intermediates available — model lacks delta-net OR this was
            // a single-token decode path. Fall back to seq_rm and note it.
            if (i == 1) fprintf(stderr,
                    "(note: rollback returned false at step 1; "
                    "traj B will diverge on hybrid models)\n");
        }
        llama_kv_cache_seq_rm(ctx, 0, n_prompt + i, -1);
    }

    // Compare.
    int first_diff = -1;
    for (int i = 0; i < n_predict; i++) {
        if (traj_A[i] != traj_B[i]) { first_diff = i; break; }
    }

    printf("\n%-4s %-24s %-24s %s\n", "step", "no-spec (A)", "spec-reject (B)", "");
    for (int i = 0; i < n_predict; i++) {
        const char * marker = (traj_A[i] == traj_B[i]) ? "SAME" : "DIFF";
        printf("%-4d %-24s %-24s %s\n",
               i,
               common_token_to_piece(ctx, traj_A[i]).c_str(),
               common_token_to_piece(ctx, traj_B[i]).c_str(),
               marker);
    }

    if (first_diff < 0) {
        printf("\nPASS: trajectories byte-identical over %d steps\n", n_predict);
    } else {
        printf("\nFAIL: first divergence at step %d ('%s' vs '%s')\n",
               first_diff,
               common_token_to_piece(ctx, traj_A[first_diff]).c_str(),
               common_token_to_piece(ctx, traj_B[first_diff]).c_str());
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return (first_diff < 0) ? 0 : 1;
}
