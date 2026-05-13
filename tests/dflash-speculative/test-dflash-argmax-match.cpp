// test-dflash-argmax-match.cpp
//
// Unit test for dflash_argmax_match — exercises each invariant binding
// via planted synthetic logit patterns.
//
// @witnesses: LongestPrefixMatchUnderArgmax
// @witnesses: NAcceptedWithinBound
// @witnesses: BonusIsArgmaxAtFirstUnacceptedRow
// @witnesses: BonusPosIsAnchorPlusNAcceptedPlusOne
// @witnesses: DeterminismUnderFixedInputs
// @witnesses: ProbabilisticVerifyOutOfScope
//
// Sub-tests:
//   1. all-accept: every drafter argmax equals target argmax → n_accepted = BLOCK_SIZE
//   2. zero-accept: drafter argmax differs at position 0 → n_accepted = 0
//   3. partial-accept: agreement up to position k, then mismatch → n_accepted = k
//   4. tie-break: planted ties in target logits → lowest token id wins
//   5. determinism: same inputs 3 times → byte-identical output (3-run check)
//   6. bonus_pos formula: known anchor_pos → bonus_pos == anchor_pos + n_accepted + 1
//
// All sub-tests use a small V (e.g. 32) so we can plant exact patterns
// in CPU code and inspect failures. Production V=248320 is exercised
// via a final sweep with large V at random patterns.

#include "ggml-cuda/dflash/dflash-argmax-match.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_CHECK(stmt)                                                   \
    do {                                                                   \
        cudaError_t _e = (stmt);                                           \
        if (_e != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                         __FILE__, __LINE__, cudaGetErrorString(_e));      \
            return 1;                                                      \
        }                                                                  \
    } while (0)

namespace {

constexpr int N_SLOTS_SMALL = 4;
constexpr int V_SMALL       = 32;

struct Result {
    std::vector<int> n_accepted;
    std::vector<int> bonus_token;
    std::vector<int> bonus_pos;
};

// Run the kernel with given logits + anchor_pos, return outputs.
Result run_kernel(
    const std::vector<float> & drafter_logits,    // [N_slots, BS, V]
    const std::vector<float> & target_logits,     // [N_slots, BS+1, V]
    const std::vector<int>   & anchor_pos,        // [N_slots]
    int N_slots, int BLOCK_SIZE, int V)
{
    float * d_drafter = nullptr;
    float * d_target  = nullptr;
    int   * d_anchor  = nullptr;
    int   * d_na      = nullptr;
    int   * d_bt      = nullptr;
    int   * d_bp      = nullptr;

    cudaMalloc(&d_drafter, drafter_logits.size() * sizeof(float));
    cudaMalloc(&d_target,  target_logits.size() * sizeof(float));
    cudaMalloc(&d_anchor,  anchor_pos.size() * sizeof(int));
    cudaMalloc(&d_na,      N_slots * sizeof(int));
    cudaMalloc(&d_bt,      N_slots * sizeof(int));
    cudaMalloc(&d_bp,      N_slots * sizeof(int));

    cudaMemcpy(d_drafter, drafter_logits.data(), drafter_logits.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target,  target_logits.data(),  target_logits.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_anchor,  anchor_pos.data(),     anchor_pos.size() * sizeof(int),       cudaMemcpyHostToDevice);

    dflash_argmax_match_launch(
        d_drafter, d_target, d_anchor,
        d_na, d_bt, d_bp,
        N_slots, BLOCK_SIZE, V, 0);
    cudaDeviceSynchronize();

    Result r;
    r.n_accepted.resize(N_slots);
    r.bonus_token.resize(N_slots);
    r.bonus_pos.resize(N_slots);
    cudaMemcpy(r.n_accepted.data(),  d_na, N_slots * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(r.bonus_token.data(), d_bt, N_slots * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(r.bonus_pos.data(),   d_bp, N_slots * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_drafter);
    cudaFree(d_target);
    cudaFree(d_anchor);
    cudaFree(d_na);
    cudaFree(d_bt);
    cudaFree(d_bp);
    return r;
}

// Plant a peaked logit: row[v] = -1.0f except row[argmax_id] = 1.0f.
void plant_peak(float * row, int V, int argmax_id) {
    for (int v = 0; v < V; ++v) row[v] = -1.0f;
    row[argmax_id] = 1.0f;
}

// Plant a tied logit: two equal max values at id_a and id_b.
void plant_tie(float * row, int V, int id_a, int id_b) {
    for (int v = 0; v < V; ++v) row[v] = -1.0f;
    row[id_a] = 1.0f;
    row[id_b] = 1.0f;
}

int sub_all_accept() {
    constexpr int BS = 4;
    const int N = N_SLOTS_SMALL;
    const int V = V_SMALL;
    std::vector<float> drafter(N * BS * V);
    std::vector<float> target(N * (BS + 1) * V);
    std::vector<int>   anchor(N);

    // Slot 0: tokens [5,6,7,8] match, target row 4 → token 9
    // Slot 1: tokens [1,2,3,4] match, target row 4 → token 10
    // Slot 2..3: similar
    for (int s = 0; s < N; ++s) {
        anchor[s] = 100 + s;
        for (int i = 0; i < BS; ++i) {
            const int tok = 5 + s + i;
            plant_peak(&drafter[(s * BS + i) * V], V, tok);
            plant_peak(&target [(s * (BS + 1) + i) * V], V, tok);
        }
        plant_peak(&target[(s * (BS + 1) + BS) * V], V, 11 + s);
    }
    Result r = run_kernel(drafter, target, anchor, N, BS, V);
    for (int s = 0; s < N; ++s) {
        if (r.n_accepted[s] != BS) {
            std::fprintf(stderr, "[FAIL all-accept] slot %d n_accepted=%d expected %d\n",
                         s, r.n_accepted[s], BS); return 1;
        }
        if (r.bonus_token[s] != 11 + s) {
            std::fprintf(stderr, "[FAIL all-accept] slot %d bonus_token=%d expected %d\n",
                         s, r.bonus_token[s], 11 + s); return 1;
        }
        if (r.bonus_pos[s] != anchor[s] + BS + 1) {
            std::fprintf(stderr, "[FAIL all-accept] slot %d bonus_pos=%d expected %d\n",
                         s, r.bonus_pos[s], anchor[s] + BS + 1); return 1;
        }
    }
    std::printf("[PASS] all-accept\n");
    return 0;
}

int sub_zero_accept() {
    constexpr int BS = 4;
    const int N = 2;
    const int V = V_SMALL;
    std::vector<float> drafter(N * BS * V);
    std::vector<float> target(N * (BS + 1) * V);
    std::vector<int>   anchor(N);

    for (int s = 0; s < N; ++s) {
        anchor[s] = 50 + s;
        plant_peak(&drafter[(s * BS + 0) * V], V, 5);  // drafter says 5
        plant_peak(&target [(s * (BS + 1) + 0) * V], V, 7);  // target says 7
        for (int i = 1; i < BS; ++i) {
            plant_peak(&drafter[(s * BS + i) * V], V, 1);
            plant_peak(&target [(s * (BS + 1) + i) * V], V, 1);
        }
        plant_peak(&target[(s * (BS + 1) + BS) * V], V, 13);
    }
    Result r = run_kernel(drafter, target, anchor, N, BS, V);
    for (int s = 0; s < N; ++s) {
        if (r.n_accepted[s] != 0) {
            std::fprintf(stderr, "[FAIL zero-accept] slot %d n_accepted=%d expected 0\n",
                         s, r.n_accepted[s]); return 1;
        }
        if (r.bonus_token[s] != 7) {
            std::fprintf(stderr, "[FAIL zero-accept] slot %d bonus_token=%d expected 7\n",
                         s, r.bonus_token[s]); return 1;
        }
        if (r.bonus_pos[s] != anchor[s] + 0 + 1) {
            std::fprintf(stderr, "[FAIL zero-accept] slot %d bonus_pos=%d expected %d\n",
                         s, r.bonus_pos[s], anchor[s] + 1); return 1;
        }
    }
    std::printf("[PASS] zero-accept\n");
    return 0;
}

int sub_partial_accept() {
    constexpr int BS = 6;
    const int N = 2;
    const int V = V_SMALL;
    std::vector<float> drafter(N * BS * V);
    std::vector<float> target(N * (BS + 1) * V);
    std::vector<int>   anchor(N);

    // Slot 0: accept up to k=3, mismatch at row 3.
    // Slot 1: accept up to k=5, mismatch at row 5.
    const int planned_k[] = {3, 5};
    const int bonus_tok[] = {17, 23};
    for (int s = 0; s < N; ++s) {
        anchor[s] = 200 + 10 * s;
        for (int i = 0; i < BS; ++i) {
            if (i < planned_k[s]) {
                plant_peak(&drafter[(s * BS + i) * V], V, 5 + i);
                plant_peak(&target [(s * (BS + 1) + i) * V], V, 5 + i);
            } else {
                // mismatch
                plant_peak(&drafter[(s * BS + i) * V], V, 1);
                plant_peak(&target [(s * (BS + 1) + i) * V], V, 2);
            }
        }
        // bonus row
        plant_peak(&target[(s * (BS + 1) + BS) * V], V, 31 + s);
        // target_argmax[planned_k[s]] is 2; bonus_token after acceptance
        // = target_argmax[n_accepted = planned_k[s]] = 2
    }
    Result r = run_kernel(drafter, target, anchor, N, BS, V);
    for (int s = 0; s < N; ++s) {
        if (r.n_accepted[s] != planned_k[s]) {
            std::fprintf(stderr, "[FAIL partial] slot %d n_accepted=%d expected %d\n",
                         s, r.n_accepted[s], planned_k[s]); return 1;
        }
        // bonus = target_argmax[n_accepted] = 2 (we planted the mismatch as
        // target=2 in the first unaccepted row).
        if (r.bonus_token[s] != 2) {
            std::fprintf(stderr, "[FAIL partial] slot %d bonus_token=%d expected 2\n",
                         s, r.bonus_token[s]); return 1;
        }
        if (r.bonus_pos[s] != anchor[s] + planned_k[s] + 1) {
            std::fprintf(stderr, "[FAIL partial] slot %d bonus_pos=%d expected %d\n",
                         s, r.bonus_pos[s], anchor[s] + planned_k[s] + 1); return 1;
        }
    }
    (void) bonus_tok;
    std::printf("[PASS] partial-accept\n");
    return 0;
}

int sub_tie_break() {
    constexpr int BS = 1;
    const int N = 1;
    const int V = V_SMALL;
    std::vector<float> drafter(N * BS * V);
    std::vector<float> target(N * (BS + 1) * V);
    std::vector<int>   anchor(N, 7);

    // Target row 0 has tied values at id 3 and id 9 → lowest wins (3).
    plant_tie(&target[0 * V], V, 3, 9);
    // drafter says 3 → should accept (match the tie-winner)
    plant_peak(&drafter[0 * V], V, 3);
    // bonus row also tied: id 5 and id 11 → bonus_token = 5
    plant_tie(&target[1 * V], V, 5, 11);

    Result r = run_kernel(drafter, target, anchor, N, BS, V);
    if (r.n_accepted[0] != 1) {
        std::fprintf(stderr, "[FAIL tie] n_accepted=%d expected 1\n", r.n_accepted[0]); return 1;
    }
    if (r.bonus_token[0] != 5) {
        std::fprintf(stderr, "[FAIL tie] bonus_token=%d expected 5\n", r.bonus_token[0]); return 1;
    }
    std::printf("[PASS] tie-break (lowest id wins)\n");
    return 0;
}

int sub_determinism() {
    constexpr int BS = 4;
    const int N = 4;
    const int V = 1024;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> drafter(N * BS * V);
    std::vector<float> target(N * (BS + 1) * V);
    std::vector<int>   anchor(N);
    for (auto & x : drafter) x = dist(rng);
    for (auto & x : target)  x = dist(rng);
    for (int i = 0; i < N; ++i) anchor[i] = 1000 + 17 * i;

    Result r1 = run_kernel(drafter, target, anchor, N, BS, V);
    Result r2 = run_kernel(drafter, target, anchor, N, BS, V);
    Result r3 = run_kernel(drafter, target, anchor, N, BS, V);
    for (int s = 0; s < N; ++s) {
        if (r1.n_accepted[s] != r2.n_accepted[s] || r1.n_accepted[s] != r3.n_accepted[s]) {
            std::fprintf(stderr, "[FAIL det] slot %d n_accepted r1=%d r2=%d r3=%d\n",
                         s, r1.n_accepted[s], r2.n_accepted[s], r3.n_accepted[s]); return 1;
        }
        if (r1.bonus_token[s] != r2.bonus_token[s] || r1.bonus_token[s] != r3.bonus_token[s]) {
            std::fprintf(stderr, "[FAIL det] slot %d bonus_token r1=%d r2=%d r3=%d\n",
                         s, r1.bonus_token[s], r2.bonus_token[s], r3.bonus_token[s]); return 1;
        }
        if (r1.bonus_pos[s] != r2.bonus_pos[s] || r1.bonus_pos[s] != r3.bonus_pos[s]) {
            std::fprintf(stderr, "[FAIL det] slot %d bonus_pos r1=%d r2=%d r3=%d\n",
                         s, r1.bonus_pos[s], r2.bonus_pos[s], r3.bonus_pos[s]); return 1;
        }
    }
    std::printf("[PASS] determinism (3-run byte-identical)\n");
    return 0;
}

int sub_production_V() {
    constexpr int BS = 4;
    const int N = 2;
    const int V = 248320;  // production drafter vocab
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    std::vector<float> drafter(static_cast<std::size_t>(N) * BS * V);
    std::vector<float> target(static_cast<std::size_t>(N) * (BS + 1) * V);
    std::vector<int>   anchor(N);
    for (auto & x : drafter) x = dist(rng);
    for (auto & x : target)  x = dist(rng);
    for (int i = 0; i < N; ++i) anchor[i] = 500 + i;

    Result r = run_kernel(drafter, target, anchor, N, BS, V);
    // Sanity: n_accepted in [0, BS], bonus_token in [0, V).
    for (int s = 0; s < N; ++s) {
        if (r.n_accepted[s] < 0 || r.n_accepted[s] > BS) {
            std::fprintf(stderr, "[FAIL prodV] slot %d n_accepted=%d out of bounds\n",
                         s, r.n_accepted[s]); return 1;
        }
        if (r.bonus_token[s] < 0 || r.bonus_token[s] >= V) {
            std::fprintf(stderr, "[FAIL prodV] slot %d bonus_token=%d out of bounds\n",
                         s, r.bonus_token[s]); return 1;
        }
        if (r.bonus_pos[s] != anchor[s] + r.n_accepted[s] + 1) {
            std::fprintf(stderr, "[FAIL prodV] slot %d bonus_pos=%d expected %d\n",
                         s, r.bonus_pos[s], anchor[s] + r.n_accepted[s] + 1); return 1;
        }
    }
    std::printf("[PASS] production-V sanity (V=%d, random inputs, NAcceptedWithinBound)\n", V);
    return 0;
}

} // anonymous namespace

int main() {
    std::printf("=== test-dflash-argmax-match ===\n");
    int fails = 0;
    fails += sub_all_accept();
    fails += sub_zero_accept();
    fails += sub_partial_accept();
    fails += sub_tie_break();
    fails += sub_determinism();
    fails += sub_production_V();
    if (fails > 0) {
        std::fprintf(stderr, "[OVERALL] %d failures\n", fails);
        return 1;
    }
    std::printf("[OVERALL] all PASS\n");
    return 0;
}
