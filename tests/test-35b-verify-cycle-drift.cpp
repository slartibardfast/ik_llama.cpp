// Simulates the server's reject-cycle flow exactly:
//   cycle i: decode batch=[cur, DUMMY] at positions N, N+1
//            cur_next = argmax(main_logits[0])
//            rollback SSM to token_idx=0 (= state after cur)
//            seq_rm from N (removes BOTH cur and DUMMY non-recurrent KV)
//            cur := cur_next; N stays (server sets n_past back to pre-cycle)
//
// Compares to no-spec: decode batch=[cur] at N, advance cur, N++.
//
// If server flow stays byte-identical to no-spec, the rollback wire-up fix is
// complete. If it diverges, we've identified a state bookkeeping bug beyond
// just the SSM rollback (specifically: double-advancement of SSM state when
// cur is re-fed into the next batch after rollback-to-post-cur).
//
// Usage: test-35b-verify-cycle-drift -m MODEL -c 512 -b 32 -ub 32 --n-predict 30

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

    int n_predict = 30;
    std::vector<char *> forward;
    forward.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n-predict") == 0 && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else { forward.push_back(argv[i]); }
    }

    if (!gpt_params_parse((int) forward.size(), forward.data(), params)) {
        fprintf(stderr, "Failed to parse params\n"); return 1;
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

    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        common_batch_add(batch, prompt[i], i, {0}, i == n_prompt - 1);
    }
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "prompt decode failed\n"); return 1; }

    const llama_token first = argmax(
            llama_get_logits_ith(ctx, batch.n_tokens - 1), n_vocab);

    const size_t snap_size = llama_state_seq_get_size(ctx, 0, 0);
    std::vector<uint8_t> snapshot(snap_size);
    if (llama_state_seq_get_data(ctx, snapshot.data(), snap_size, 0, 0) == 0) {
        fprintf(stderr, "snap failed\n"); return 1;
    }
    auto restore = [&]() {
        llama_kv_cache_clear(ctx);
        llama_state_seq_set_data(ctx, snapshot.data(), snap_size, 0, 0);
    };

    // --- No-spec trajectory ---
    restore();
    std::vector<llama_token> traj_A;
    traj_A.push_back(first);
    llama_token cur = first;
    for (int i = 1; i < n_predict; i++) {
        common_batch_clear(batch);
        common_batch_add(batch, cur, n_prompt + (i - 1), {0}, true);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "A fail step %d\n", i); return 1; }
        cur = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
        traj_A.push_back(cur);
    }

    // --- Server-mimic trajectory: always-reject, re-feed cur ---
    restore();
    std::vector<llama_token> traj_B;
    traj_B.push_back(first);
    cur = first;
    llama_token DUMMY = (llama_token) (n_vocab / 2);
    if (llama_vocab_is_eog(llama_model_get_vocab(model), DUMMY)) DUMMY++;
    int pos = n_prompt;  // server keeps n_past at pre-cycle position on reject

    for (int i = 1; i < n_predict; i++) {
        common_batch_clear(batch);
        common_batch_add(batch, cur,   pos,     {0}, true);
        common_batch_add(batch, DUMMY, pos + 1, {0}, true);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "B fail step %d\n", i); return 1; }
        llama_token next = argmax(llama_get_logits_ith(ctx, 0), n_vocab);
        traj_B.push_back(next);
        // Rollback SSM to "post-cur" (token_idx=0). Server does this on reject.
        llama_rollback_delta_net_state(ctx, /*token_idx=*/0, 0, pos);
        // seq_rm removes KV at positions >= pos (BOTH cur and DUMMY cells).
        llama_kv_cache_seq_rm(ctx, 0, pos, -1);
        // Next cycle's cur = argmax from this cycle's pos-0 logits.
        // Server keeps pos unchanged (slot.n_past = pre-cycle n_past on reject)
        cur = next;
    }

    int first_diff = -1;
    for (int i = 0; i < n_predict; i++) {
        if (traj_A[i] != traj_B[i]) { first_diff = i; break; }
    }

    printf("\n%-4s %-24s %-24s %s\n", "step", "no-spec (A)", "server-reject (B)", "");
    for (int i = 0; i < n_predict; i++) {
        const char * marker = (traj_A[i] == traj_B[i]) ? "SAME" : "DIFF";
        printf("%-4d %-24s %-24s %s\n", i,
                common_token_to_piece(ctx, traj_A[i]).c_str(),
                common_token_to_piece(ctx, traj_B[i]).c_str(),
                marker);
    }
    if (first_diff < 0) {
        printf("\nPASS: server-reject trajectory byte-identical to no-spec over %d steps\n", n_predict);
    } else {
        printf("\nFAIL: first divergence at step %d\n", first_diff);
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return (first_diff < 0) ? 0 : 1;
}
