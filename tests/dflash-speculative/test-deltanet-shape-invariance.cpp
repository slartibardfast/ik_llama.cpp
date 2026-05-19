// test-deltanet-shape-invariance.cpp
//
// Probes whether the CUDA delta_net op produces byte-identical slot-0 output
// AND byte-identical slot-0 new-state across n_seqs ∈ {1, 2, 4, 8}. Each
// seq's recurrent state is algebraically independent of other seqs; this
// test binds that fact at the bit level for production geometry.
//
// Production geometry (Qwen 3.6 27B Linear-Attention layers):
//   head_dim = 128
//   H_v      = 16    (V/Q heads)
//   H_k      = 2     (KQ heads, gqa_ratio = H_v / H_k = 8)
//   n_tokens = 1     (decode-step, the harness-failing path)
//
// Layout per ggml_delta_net contract:
//   Q,K: [head_dim, n_tokens, H_k, n_seqs]
//   V:   [head_dim, n_tokens, H_v, n_seqs]
//   G:   [n_tokens, 1,        H_v, n_seqs]
//   B:   [1,        n_tokens, H_v, n_seqs]
//   S:   [head_dim, head_dim*H_v, 1, n_seqs]
//
// Returns: 0 = PASS, 1 = FAIL, 77 = SKIP (no CUDA).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static constexpr int HEAD_DIM = 128;
static constexpr int H_V      = 16;
static constexpr int H_K      = 2;     // gqa_ratio = 8
static constexpr int N_TOKENS = 1;

struct slot0_result {
    std::vector<float> output;   // HEAD_DIM * H_V * N_TOKENS floats
    std::vector<float> state;    // HEAD_DIM * HEAD_DIM * H_V floats
};

static void fill_slot(std::vector<float> & buf, size_t per_seq_floats, int seq, uint64_t base_seed) {
    std::mt19937_64 rng(base_seed ^ (uint64_t)(seq + 1) * 0x9E3779B97F4A7C15ULL);
    for (size_t i = 0; i < per_seq_floats; ++i) {
        const uint32_t r = (uint32_t)(rng() & 0xffffffffULL);
        buf[i] = ((int32_t)(r & 0xffff) - 32768) / 32768.0f * 0.5f;
    }
}

// Slot 0 inputs are fixed (independent of n_seqs); slot k>=1 inputs are
// reproducible-random with a seed that depends on slot index but not n_seqs.
static bool run_one(ggml_backend_t backend, int n_seqs, slot0_result & out) {
    static const size_t mem_size = 16 * 1024 * 1024;
    ggml_init_params params = { mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, N_TOKENS, H_K, n_seqs);
    ggml_tensor * k     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, N_TOKENS, H_K, n_seqs);
    ggml_tensor * v     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, N_TOKENS, H_V, n_seqs);
    ggml_tensor * g     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, N_TOKENS, 1, H_V, n_seqs);
    ggml_tensor * b     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, N_TOKENS, H_V, n_seqs);
    ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, HEAD_DIM * H_V, 1, n_seqs);

    ggml_tensor * out_t = ggml_delta_net(ctx, q, k, v, g, b, state, /*save_all_steps=*/false);
    ggml_set_name(out_t, "delta_net_out");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_t);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }

    const size_t qk_per_seq    = (size_t)HEAD_DIM * N_TOKENS * H_K;
    const size_t v_per_seq     = (size_t)HEAD_DIM * N_TOKENS * H_V;
    const size_t g_per_seq     = (size_t)N_TOKENS * H_V;
    const size_t b_per_seq     = (size_t)N_TOKENS * H_V;
    const size_t state_per_seq = (size_t)HEAD_DIM * HEAD_DIM * H_V;

    std::vector<float> Q_all((size_t)n_seqs * qk_per_seq);
    std::vector<float> K_all((size_t)n_seqs * qk_per_seq);
    std::vector<float> V_all((size_t)n_seqs * v_per_seq);
    std::vector<float> G_all((size_t)n_seqs * g_per_seq);
    std::vector<float> B_all((size_t)n_seqs * b_per_seq);
    std::vector<float> S_all((size_t)n_seqs * state_per_seq);

    for (int s = 0; s < n_seqs; ++s) {
        std::vector<float> q_s(qk_per_seq), k_s(qk_per_seq), v_s(v_per_seq);
        std::vector<float> g_s(g_per_seq),  b_s(b_per_seq),  st_s(state_per_seq);
        fill_slot(q_s,  qk_per_seq,    s, 0xA1ULL);
        fill_slot(k_s,  qk_per_seq,    s, 0xA2ULL);
        fill_slot(v_s,  v_per_seq,     s, 0xA3ULL);
        fill_slot(g_s,  g_per_seq,     s, 0xA4ULL);
        fill_slot(b_s,  b_per_seq,     s, 0xA5ULL);
        fill_slot(st_s, state_per_seq, s, 0xA6ULL);
        std::memcpy(&Q_all[s*qk_per_seq],    q_s.data(),  qk_per_seq*sizeof(float));
        std::memcpy(&K_all[s*qk_per_seq],    k_s.data(),  qk_per_seq*sizeof(float));
        std::memcpy(&V_all[s*v_per_seq],     v_s.data(),  v_per_seq*sizeof(float));
        std::memcpy(&G_all[s*g_per_seq],     g_s.data(),  g_per_seq*sizeof(float));
        std::memcpy(&B_all[s*b_per_seq],     b_s.data(),  b_per_seq*sizeof(float));
        std::memcpy(&S_all[s*state_per_seq], st_s.data(), state_per_seq*sizeof(float));
    }

    ggml_backend_tensor_set(q,     Q_all.data(), 0, Q_all.size()*sizeof(float));
    ggml_backend_tensor_set(k,     K_all.data(), 0, K_all.size()*sizeof(float));
    ggml_backend_tensor_set(v,     V_all.data(), 0, V_all.size()*sizeof(float));
    ggml_backend_tensor_set(g,     G_all.data(), 0, G_all.size()*sizeof(float));
    ggml_backend_tensor_set(b,     B_all.data(), 0, B_all.size()*sizeof(float));
    ggml_backend_tensor_set(state, S_all.data(), 0, S_all.size()*sizeof(float));

    const auto status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "graph_compute failed (status=%d) at n_seqs=%d\n", (int)status, n_seqs);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    const size_t out_floats   = (size_t)HEAD_DIM * H_V * N_TOKENS * n_seqs;
    const size_t state_floats = state_per_seq * n_seqs;
    std::vector<float> full(out_floats + state_floats);
    ggml_backend_tensor_get(out_t, full.data(), 0, full.size()*sizeof(float));

    // Output layout per kernel comment: [HEAD_DIM, H_V, N_TOKENS, n_seqs] — slot 0 is first
    // (HEAD_DIM * H_V * N_TOKENS) floats. State for slot 0 is the first state_per_seq floats
    // after the output region.
    out.output.assign(full.begin(), full.begin() + (size_t)HEAD_DIM * H_V * N_TOKENS);
    out.state.assign(full.begin() + out_floats, full.begin() + out_floats + state_per_seq);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

static int count_diffs(const std::vector<float> & a, const std::vector<float> & b, float & max_abs) {
    int n = 0;
    max_abs = 0.0f;
    for (size_t j = 0; j < a.size(); ++j) {
        uint32_t ai, bi;
        std::memcpy(&ai, &a[j], 4);
        std::memcpy(&bi, &b[j], 4);
        if (ai != bi) {
            ++n;
            const float d = std::abs(a[j] - b[j]);
            if (d > max_abs) max_abs = d;
        }
    }
    return n;
}

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed; SKIP\n");
        return 77;
    }

    const std::vector<int> n_seqs_list = {1, 2, 4, 8};
    std::vector<slot0_result> results(n_seqs_list.size());

    for (size_t i = 0; i < n_seqs_list.size(); ++i) {
        if (!run_one(backend, n_seqs_list[i], results[i])) {
            ggml_backend_free(backend);
            return 1;
        }
    }
    ggml_backend_free(backend);

    fprintf(stderr, "slot-0 output first 8 floats per n_seqs:\n");
    for (size_t i = 0; i < n_seqs_list.size(); ++i) {
        fprintf(stderr, "  n_seqs=%d:", n_seqs_list[i]);
        for (int k = 0; k < 8; ++k) fprintf(stderr, " %+10.6f", results[i].output[k]);
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "slot-0 new-state first 8 floats per n_seqs:\n");
    for (size_t i = 0; i < n_seqs_list.size(); ++i) {
        fprintf(stderr, "  n_seqs=%d:", n_seqs_list[i]);
        for (int k = 0; k < 8; ++k) fprintf(stderr, " %+10.6f", results[i].state[k]);
        fprintf(stderr, "\n");
    }

    int total = 0;
    const auto & ref = results[0];
    for (size_t i = 1; i < results.size(); ++i) {
        const auto & cur = results[i];
        float mo = 0, ms = 0;
        const int do_ = count_diffs(ref.output, cur.output, mo);
        const int ds  = count_diffs(ref.state,  cur.state,  ms);
        if (do_ == 0 && ds == 0) {
            fprintf(stderr, "  n_seqs=%d: byte-identical to n_seqs=1 baseline (output + state)\n",
                    n_seqs_list[i]);
        } else {
            fprintf(stderr, "  n_seqs=%d: output diffs=%d/%zu (max |Δ|=%.3e), state diffs=%d/%zu (max |Δ|=%.3e)\n",
                    n_seqs_list[i],
                    do_, ref.output.size(), mo,
                    ds,  ref.state.size(),  ms);
            total += do_ + ds;
        }
    }

    if (total == 0) {
        fprintf(stderr, "OVERALL: PASS — slot 0 output + new-state byte-identical across n_seqs ∈ {1,2,4,8}\n");
        return 0;
    }
    fprintf(stderr, "OVERALL: FAIL — %d byte mismatches in DeltaNet slot-0 across n_seqs\n", total);
    return 1;
}
