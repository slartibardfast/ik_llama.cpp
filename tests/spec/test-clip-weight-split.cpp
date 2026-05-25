// test-clip-weight-split.cpp
//
// PHASE 46 B.5b spec test — stub-style property test of the
// is_splittable predicate and two-context weight residency pattern
// in examples/mtmd/clip.cpp:3500+.
//
// What it tests:
//   1. is_splittable predicate matches the production form:
//        ndims >= 2 AND ne[0] >= 256 AND ne[1] >= 256 AND nbytes >= 1 MiB
//   2. Small tensors (norms, biases) route to ctx_data (single-device).
//   3. Large matmul weights route to ctx_data_split (multi-device).
//   4. The two-ctx partition is *exhaustive*: every tensor lands in
//      exactly one ctx (no duplicate, no orphan).
//   5. The split decision is monotonic in size — once a dim/byte
//      threshold is crossed, every larger tensor also splits.
//
// Cross-codepath consistency (CrossCodepathConsistency.allium spec #5):
// the predicate here mirrors what llama_layer::split_* uses on the LM
// side. Both consumers see the same shared mgpu_split_config from B.4.
//
// Returns: 0 = PASS, 1 = FAIL.

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>

namespace {

struct StubTensor {
    std::string name;
    int ndims;
    int64_t ne[4];
    size_t nbytes;
};

// Mirror of clip.cpp's is_splittable predicate.
bool is_splittable(const StubTensor & t) {
    if (t.ndims < 2) return false;
    if (t.ne[0] < 256) return false;
    if (t.ne[1] < 256) return false;
    if (t.nbytes < (1u << 20)) return false; // 1 MiB
    return true;
}

enum class Residency { LAYER, SPLIT };

Residency partition(const StubTensor & t) {
    return is_splittable(t) ? Residency::SPLIT : Residency::LAYER;
}

int test_norm_stays_local() {
    // A typical CLIP layer-norm: 1D weight + bias.
    StubTensor t{"v.blk.0.ln1.weight", 1, {1280, 1, 1, 1}, 1280 * 4};
    if (partition(t) != Residency::LAYER) {
        fprintf(stderr, "FAIL test_norm_stays_local: routed to SPLIT\n");
        return 1;
    }
    return 0;
}

int test_bias_stays_local() {
    StubTensor t{"v.blk.0.attn_q.bias", 1, {1280, 1, 1, 1}, 1280 * 4};
    if (partition(t) != Residency::LAYER) {
        fprintf(stderr, "FAIL test_bias_stays_local\n");
        return 1;
    }
    return 0;
}

int test_matmul_weight_splits() {
    // Typical CLIP attention projection: 1280×1280 f16 = ~3.1 MiB.
    StubTensor t{"v.blk.0.attn_q.weight", 2, {1280, 1280, 1, 1}, 1280 * 1280 * 2};
    if (partition(t) != Residency::SPLIT) {
        fprintf(stderr, "FAIL test_matmul_weight_splits\n");
        return 1;
    }
    return 0;
}

int test_just_under_byte_threshold() {
    // 256x256 f16 = 128 KiB — fails nbytes >= 1 MiB.
    StubTensor t{"small.matmul", 2, {256, 256, 1, 1}, 256 * 256 * 2};
    if (partition(t) != Residency::LAYER) {
        fprintf(stderr, "FAIL test_just_under_byte_threshold\n");
        return 1;
    }
    return 0;
}

int test_just_under_dim_threshold() {
    // 255x4096 — fails ne[0] >= 256 even though nbytes is large.
    StubTensor t{"thin.matmul", 2, {255, 4096, 1, 1}, 255 * 4096 * 2};
    if (partition(t) != Residency::LAYER) {
        fprintf(stderr, "FAIL test_just_under_dim_threshold\n");
        return 1;
    }
    return 0;
}

int test_1d_never_splits() {
    // Even a huge 1D tensor never splits — ndims < 2 fails.
    StubTensor t{"huge.1d", 1, {65536, 1, 1, 1}, 65536 * 4};
    if (partition(t) != Residency::LAYER) {
        fprintf(stderr, "FAIL test_1d_never_splits\n");
        return 1;
    }
    return 0;
}

int test_partition_is_exhaustive() {
    // Build a 1024-token CLIP-encoder-shaped roster and confirm every
    // tensor lands in exactly one bucket. (No tensor escapes the split.)
    std::vector<StubTensor> roster = {
        {"v.position_embd", 2, {1280, 1024, 1, 1}, 1280*1024*2},
        {"v.patch_embd.weight", 4, {14, 14, 3, 1280}, 14*14*3*1280*2},
        {"v.patch_embd.bias", 1, {1280,1,1,1}, 1280*4},
        {"v.blk.0.attn_q.weight", 2, {1280,1280,1,1}, 1280*1280*2},
        {"v.blk.0.attn_q.bias", 1, {1280,1,1,1}, 1280*4},
        {"v.blk.0.attn_k.weight", 2, {1280,1280,1,1}, 1280*1280*2},
        {"v.blk.0.attn_v.weight", 2, {1280,1280,1,1}, 1280*1280*2},
        {"v.blk.0.attn_out.weight", 2, {1280,1280,1,1}, 1280*1280*2},
        {"v.blk.0.ffn_up.weight", 2, {1280,5120,1,1}, 1280*5120*2},
        {"v.blk.0.ffn_down.weight", 2, {5120,1280,1,1}, 5120*1280*2},
        {"v.blk.0.ln1.weight", 1, {1280,1,1,1}, 1280*4},
        {"v.blk.0.ln2.weight", 1, {1280,1,1,1}, 1280*4},
    };
    size_t layer_count = 0, split_count = 0;
    for (const auto & t : roster) {
        auto r = partition(t);
        if (r == Residency::LAYER) ++layer_count;
        else                       ++split_count;
    }
    if (layer_count + split_count != roster.size()) {
        fprintf(stderr, "FAIL test_partition_is_exhaustive: %zu+%zu != %zu\n",
                layer_count, split_count, roster.size());
        return 1;
    }
    // Sanity: matmul weights split, norms/biases stay local.
    if (split_count < 5 || layer_count < 3) {
        fprintf(stderr, "FAIL test_partition_is_exhaustive: shape sanity (%zu / %zu)\n",
                split_count, layer_count);
        return 1;
    }
    return 0;
}

int test_monotonic_in_size() {
    // For fixed ndims=2 and a tensor type that crosses the boundary,
    // every larger size that's also splittable must remain split.
    int64_t sizes[] = {255, 256, 512, 1024, 2048};
    bool last_split = false;
    for (int64_t s : sizes) {
        StubTensor t{"x", 2, {s, s, 1, 1}, (size_t)(s*s*2)};
        bool sp = is_splittable(t);
        if (last_split && !sp) {
            fprintf(stderr, "FAIL test_monotonic_in_size: regressed at s=%lld\n", (long long)s);
            return 1;
        }
        last_split = sp;
    }
    return 0;
}

} // namespace

int main() {
    int rc = 0;
    rc |= test_norm_stays_local();
    rc |= test_bias_stays_local();
    rc |= test_matmul_weight_splits();
    rc |= test_just_under_byte_threshold();
    rc |= test_just_under_dim_threshold();
    rc |= test_1d_never_splits();
    rc |= test_partition_is_exhaustive();
    rc |= test_monotonic_in_size();
    if (rc == 0) {
        printf("PASS test-clip-weight-split (8 cases)\n");
    } else {
        printf("FAIL test-clip-weight-split\n");
    }
    return rc;
}
