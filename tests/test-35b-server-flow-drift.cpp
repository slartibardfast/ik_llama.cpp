// Server-flow drift probe.
//
// Simulates the server's speculative MTP cycle EXACTLY on 35B so we can
// bisect where residual 1-2 token drift enters.
//
// Trajectory A (ground truth): batch=1 greedy argmax trajectory of length N.
// Trajectory B (server flow):
//   for cycle = 0..:
//     1. Draft from mtp_logits_buf (at position n_accepted-1 if prior partial
//        reject, else last position).
//     2. Verify batch=[sampled, draft] at positions [P, P+1].
//     3. Decode.
//     4. Greedy sample:
//          S0 = argmax(logits[0]); accept S0.
//          If S0 == draft: S1 = argmax(logits[1]); accept S1 too.
//          ids = accepted outputs (length 1 or 2).
//     5. cache ← cache + ids[0..ids.size()-1), slot.sampled = ids.back().
//        n_past = cache.size(). (slot.sampled NOT yet in cache.)
//     6. If partial reject: call llama_rollback_delta_net_state(
//            token_idx = n_accepted - 1,
//            target_pos = n_past - 1).
//        Set last_verify_n_accepted = n_accepted.
//     7. llama_kv_cache_seq_rm(seq, n_past, -1).
//
// Record output sequence over B and compare to first-N tokens of A.
//
// Diagnostic toggles:
//   --no-mtp-draft   : use a fixed dummy draft instead of MTP (tests whether
//                       MTP draft-position is the drift source).
//   --no-rollback    : skip llama_rollback_delta_net_state entirely (shows
//                       baseline drift without the rollback).
//   --always-accept-both : force-accept both batch positions regardless of
//                       sample comparison (never enters rollback path).
//
// Usage:
//   test-35b-server-flow-drift -m <model.gguf> [--prompt "..."]
//                              [--n-predict 20] [-c 256 -b 16 -ub 16 -mtp]

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
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

    int  n_predict        = 20;
    bool opt_no_mtp_draft = false;
    bool opt_no_rollback  = false;
    bool opt_always_accept_both = false;
    bool opt_always_rollback    = false;
    bool opt_always_reject_second = false;

    std::vector<char *> forward;
    forward.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n-predict") == 0 && i + 1 < argc) {
            n_predict = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-mtp-draft") == 0) {
            opt_no_mtp_draft = true;
        } else if (strcmp(argv[i], "--no-rollback") == 0) {
            opt_no_rollback = true;
        } else if (strcmp(argv[i], "--always-accept-both") == 0) {
            opt_always_accept_both = true;
        } else if (strcmp(argv[i], "--always-rollback") == 0) {
            opt_always_rollback = true;
        } else if (strcmp(argv[i], "--always-reject-second") == 0) {
            opt_always_reject_second = true;
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

    // A dummy fallback draft (only used with --no-mtp-draft).
    llama_token DUMMY = (llama_token) (n_vocab / 2);
    if (llama_vocab_is_eog(llama_model_get_vocab(model), DUMMY)) DUMMY++;

    // Tokenize and prompt-eval.
    std::vector<llama_token> prompt = ::common_tokenize(ctx, params.prompt, true);
    const int n_prompt = (int) prompt.size();
    fprintf(stderr, "Prompt: '%s' (%d tokens), n_predict=%d\n"
                    "Toggles: no-mtp-draft=%d no-rollback=%d always-accept-both=%d\n",
            params.prompt.c_str(), n_prompt, n_predict,
            (int)opt_no_mtp_draft, (int)opt_no_rollback, (int)opt_always_accept_both);

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

    // ---- Trajectory A: batch=1 greedy ground truth ----
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

    // ---- Trajectory B: server flow simulation ----
    restore();
    std::vector<llama_token> traj_B;
    traj_B.push_back(first);
    int last_verify_n_accepted = -1;  // slot state mirror
    llama_token sampled         = first;
    int n_past                  = n_prompt;  // cache length; `sampled` NOT yet in cache

    while ((int) traj_B.size() < n_predict) {
        // (1) Draft extraction.
        llama_token draft;
        if (opt_no_mtp_draft) {
            draft = DUMMY;
        } else {
            if (last_verify_n_accepted > 0) {
                draft = llama_get_mtp_draft_token_at(ctx, last_verify_n_accepted - 1);
            } else {
                draft = llama_get_mtp_draft_token(ctx);
            }
            if (draft == LLAMA_TOKEN_NULL) draft = DUMMY;
        }
        last_verify_n_accepted = -1;  // consumed

        // (2) Verify batch [sampled @ n_past, draft @ n_past+1].
        common_batch_clear(batch);
        common_batch_add(batch, sampled, n_past,     {0}, true);
        common_batch_add(batch, draft,   n_past + 1, {0}, true);

        // (3) Decode.
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "B verify decode failed\n"); return 1; }

        // (4) Greedy sample+accept.
        const llama_token S0 = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
        const llama_token S1 = argmax(llama_get_logits_ith(ctx, 1), n_vocab);

        std::vector<llama_token> ids;
        ids.push_back(S0);
        const bool accept_both = !opt_always_reject_second &&
                                 (opt_always_accept_both || (S0 == draft));
        if (accept_both) ids.push_back(S1);
        const int n_accepted    = (int) ids.size();
        const int n_verify_batch = 2;

        // (5) Cache update: cache grows by (n_accepted - 1), sampled = ids.back().
        //     Before: cache has n_past tokens, `sampled` is held out.
        //     keep_first(n_past + 1 - n_draft=1) == n_past, then insert ids[0..n_accepted-1).
        //     Here we don't track an actual cache vector — only n_past counts.
        //     Conceptually cache had (n_past + sampled) after add_sampled_tokens;
        //     keep_first drops the draft; then insert n_accepted-1 tokens; sampled is last accepted.
        const int n_past_before = n_past;
        // Before this cycle, cache length (incl. sampled that was batched) was n_past + 2 (verify batch).
        // keep_first(len - n_draft) with n_draft=1: cache len = n_past + 1 (sampled retained).
        // insert(n_accepted - 1) tokens: cache len = n_past + 1 + (n_accepted - 1) = n_past + n_accepted.
        // slot.n_past := cache.size() == n_past + n_accepted.
        n_past = n_past_before + n_accepted;
        sampled = ids.back();
        // Record output tokens that were emitted into the stream this cycle.
        // Server emits all n_accepted tokens; we've already stored `first` as
        // the very first output, corresponding to "before cycle 0 started",
        // so each cycle's ids.front() is the "produced" token from sampling
        // the old `sampled`'s continuation, and ids.back() becomes the new
        // `sampled`. Both are emitted to the user. Align with traj_A:
        for (llama_token t : ids) {
            if ((int) traj_B.size() < n_predict) traj_B.push_back(t);
        }

        // (6) Rollback.
        const bool do_rollback =
                !opt_no_rollback &&
                (n_accepted < n_verify_batch || opt_always_rollback);
        if (do_rollback) {
            const bool ok = llama_rollback_delta_net_state(
                    ctx, /*token_idx=*/n_accepted - 1, /*seq_id=*/0,
                    /*target_pos=*/n_past - 1);
            if (!ok && (int) traj_B.size() == 2) {
                fprintf(stderr, "(note: rollback returned false — intermediates unavailable)\n");
            }
            last_verify_n_accepted = n_accepted;
        } else {
            last_verify_n_accepted = -1;
        }

        // (7) Trim KV cells past n_past.
        llama_kv_cache_seq_rm(ctx, 0, n_past, -1);
    }

    // ---- Compare ----
    int first_diff = -1;
    for (int i = 0; i < n_predict; i++) {
        if (traj_A[i] != traj_B[i]) { first_diff = i; break; }
    }

    printf("\n%-4s %-24s %-24s %s\n", "step", "ground-truth (A)", "server-flow (B)", "");
    for (int i = 0; i < n_predict; i++) {
        const char * marker = (traj_A[i] == traj_B[i]) ? "SAME" : "DIFF";
        printf("%-4d %-24s %-24s %s\n", i,
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
