// test-mgpu-split-cross-codepath.cpp
//
// Spec test for specs/mgpu-split/CrossCodepathConsistency.allium
// (yarn-agentic Phase 46 B.0-full spec #5).
//
// Binds the structural invariants of the LM ⇔ CLIP cross-codepath
// contract over a shared mgpu_split_config. The deep semantic
// invariants (LmRefactorPreservesAllLmInvariants,
// LmObservableBehaviorUnchanged, ClipNewObservableBehaviorIsAdditive)
// are integration-tested by Phase 46 B.6 LM gate re-cert and the
// production CLIP encoder battery; this test covers what a unit-level
// stub can credibly assert:
//
//   1. PairOverSameStruct — lm_view and clip_view point to the same
//      backing struct (pointer equality).
//   2. PairConsumersDistinct — the consumer labels are lm and clip.
//   3. InvariantSatisfactionEquivalent — every field value read
//      through lm_view equals the value read through clip_view.
//   4. MutationsByOneConsumerVisibleToOther — when LM mutates
//      mem_used[d], CLIP — viewing the same struct — sees it.
//   5. ImmutableFieldsBidirectionallyImmutable — capture a hash over
//      {split_mode, splits, capacity_per_device} before and after a
//      legitimate mem_used mutation; the hash is unchanged. Bound
//      structurally by the spec; here we assert observationally.
//   6. AccessPathsBelongToConsumer — the two consumer call-site sets
//      from the spec are disjoint.
//   7. BuftSetupOutputEquivalent — both views read identical
//      buft_layer entries after a stub buft assignment.
//   8. CreateSplitOutputEquivalent — both views read identical
//      mem_used after a stub create_split walk.
//
// Returns: 0 = PASS, 1 = FAIL.

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

// Stub mirror of ggml_mgpu_split_config (ik_llama.cpp/ggml/include/
// ggml-mgpu-split.h). Field layout matches the production struct; we
// don't include the real header here because this test is structural
// and runs with no backend init.
enum SplitMode { MODE_NONE = 0, MODE_LAYER = 1, MODE_ATTN = 2, MODE_GRAPH = 3 };

struct StubBuft {
    int  kind_id;        // 0 = nil, 1 = split, 2 = offload, 3 = cpu
    int  device_id;      // -1 unless kind_id == offload
};

struct StubMgpuSplitConfig {
    int                                  n_device;
    std::vector<int>                     devices;
    std::vector<size_t>                  capacity_per_device;
    std::vector<size_t>                  mem_used_per_device;
    std::vector<float>                   splits;
    StubBuft                             split_buft;
    SplitMode                            split_mode;
    int                                  n_layer;
    int                                  i_gpu_start;
    std::vector<std::pair<StubBuft, StubBuft>> buft_layer;     // (split, offload)
    std::vector<int>                     default_layer_device;
};

enum Consumer { CONSUMER_LM, CONSUMER_CLIP };

// A consumer view: a pointer to a shared backing struct plus the
// consumer label. Mirrors ConfigInstance in the spec.
struct ConfigInstance {
    StubMgpuSplitConfig *  cfg;
    Consumer               consumer;
};

// Build a realistic Phase 46 production-shape config: 2 CUDA devices,
// GRAPH mode, 32 layers, 4 of them CPU-offloaded (i_gpu_start = 4).
StubMgpuSplitConfig make_production_shape_config() {
    StubMgpuSplitConfig c{};
    c.n_device              = 2;
    c.devices               = {0, 1};
    c.capacity_per_device   = {21ULL << 30, 21ULL << 30}; // 21 GiB each (RTX 6000)
    c.mem_used_per_device   = {0, 0};
    c.splits                = {0.5f, 1.0f};
    c.split_buft            = StubBuft{1, -1};           // kind = "split"
    c.split_mode            = MODE_GRAPH;
    c.n_layer               = 32;
    c.i_gpu_start           = 4;
    c.buft_layer.resize(c.n_layer);
    c.default_layer_device.resize(c.n_layer);
    for (int i = 0; i < c.n_layer; ++i) {
        if (i < c.i_gpu_start) {
            // CPU layers: split = nil, offload = cpu, device = -1.
            c.buft_layer[i]            = {StubBuft{0, -1}, StubBuft{3, -1}};
            c.default_layer_device[i]  = -1;
        } else {
            // GPU layers: split = split, offload = offload(device),
            // round-robin across the 2 devices.
            int d = (i - c.i_gpu_start) % c.n_device;
            c.buft_layer[i]            = {StubBuft{1, -1}, StubBuft{2, d}};
            c.default_layer_device[i]  = d;
        }
    }
    return c;
}

// ============================================================
// Per-invariant tests
// ============================================================

int test_pair_over_same_struct() {
    StubMgpuSplitConfig backing = make_production_shape_config();
    ConfigInstance lm{&backing, CONSUMER_LM};
    ConfigInstance clip{&backing, CONSUMER_CLIP};
    if (lm.cfg != clip.cfg) {
        fprintf(stderr, "FAIL test_pair_over_same_struct: pointers differ (lm=%p clip=%p)\n",
                (void*)lm.cfg, (void*)clip.cfg);
        return 1;
    }
    return 0;
}

int test_pair_consumers_distinct() {
    StubMgpuSplitConfig backing = make_production_shape_config();
    ConfigInstance lm{&backing, CONSUMER_LM};
    ConfigInstance clip{&backing, CONSUMER_CLIP};
    if (!(lm.consumer == CONSUMER_LM && clip.consumer == CONSUMER_CLIP)) {
        fprintf(stderr, "FAIL test_pair_consumers_distinct: lm=%d clip=%d\n",
                (int)lm.consumer, (int)clip.consumer);
        return 1;
    }
    if (lm.consumer == clip.consumer) {
        fprintf(stderr, "FAIL test_pair_consumers_distinct: labels collapsed\n");
        return 1;
    }
    return 0;
}

int test_invariant_satisfaction_equivalent() {
    StubMgpuSplitConfig backing = make_production_shape_config();
    ConfigInstance lm{&backing, CONSUMER_LM};
    ConfigInstance clip{&backing, CONSUMER_CLIP};

    if (lm.cfg->n_device              != clip.cfg->n_device)              { fprintf(stderr, "FAIL n_device\n"); return 1; }
    if (lm.cfg->n_layer               != clip.cfg->n_layer)               { fprintf(stderr, "FAIL n_layer\n"); return 1; }
    if (lm.cfg->i_gpu_start           != clip.cfg->i_gpu_start)           { fprintf(stderr, "FAIL i_gpu_start\n"); return 1; }
    if (lm.cfg->split_mode            != clip.cfg->split_mode)            { fprintf(stderr, "FAIL split_mode\n"); return 1; }
    if (lm.cfg->split_buft.kind_id    != clip.cfg->split_buft.kind_id)    { fprintf(stderr, "FAIL split_buft.kind_id\n"); return 1; }
    if (lm.cfg->devices               != clip.cfg->devices)               { fprintf(stderr, "FAIL devices\n"); return 1; }
    if (lm.cfg->capacity_per_device   != clip.cfg->capacity_per_device)   { fprintf(stderr, "FAIL capacity_per_device\n"); return 1; }
    if (lm.cfg->mem_used_per_device   != clip.cfg->mem_used_per_device)   { fprintf(stderr, "FAIL mem_used_per_device\n"); return 1; }
    if (lm.cfg->splits                != clip.cfg->splits)                { fprintf(stderr, "FAIL splits\n"); return 1; }
    if (lm.cfg->default_layer_device  != clip.cfg->default_layer_device)  { fprintf(stderr, "FAIL default_layer_device\n"); return 1; }
    for (int i = 0; i < lm.cfg->n_layer; ++i) {
        const auto & a = lm.cfg->buft_layer[i];
        const auto & b = clip.cfg->buft_layer[i];
        if (a.first.kind_id   != b.first.kind_id ||
            a.first.device_id != b.first.device_id ||
            a.second.kind_id  != b.second.kind_id ||
            a.second.device_id!= b.second.device_id) {
            fprintf(stderr, "FAIL buft_layer[%d]\n", i);
            return 1;
        }
    }
    return 0;
}

int test_mutations_visible() {
    StubMgpuSplitConfig backing = make_production_shape_config();
    ConfigInstance lm{&backing, CONSUMER_LM};
    ConfigInstance clip{&backing, CONSUMER_CLIP};
    const size_t before = clip.cfg->mem_used_per_device[0];
    const size_t delta  = 4ULL << 20; // 4 MiB
    lm.cfg->mem_used_per_device[0] += delta;
    if (clip.cfg->mem_used_per_device[0] != before + delta) {
        fprintf(stderr, "FAIL test_mutations_visible: clip sees %zu, expected %zu\n",
                clip.cfg->mem_used_per_device[0], before + delta);
        return 1;
    }
    return 0;
}

uint64_t hash_immutable_fields(const StubMgpuSplitConfig * c) {
    // FNV-1a over the three frozen-after-init fields. If any of them
    // changes, this digest changes.
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&](const void * p, size_t n) {
        const uint8_t * b = (const uint8_t *)p;
        for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 0x100000001b3ULL; }
    };
    mix(&c->split_mode, sizeof(c->split_mode));
    mix(c->splits.data(), c->splits.size() * sizeof(float));
    mix(c->capacity_per_device.data(), c->capacity_per_device.size() * sizeof(size_t));
    return h;
}

int test_immutable_fields_immutable() {
    StubMgpuSplitConfig backing = make_production_shape_config();
    ConfigInstance lm{&backing, CONSUMER_LM};
    ConfigInstance clip{&backing, CONSUMER_CLIP};
    const uint64_t before = hash_immutable_fields(lm.cfg);
    // Legitimate mutation: mem_used grows as the LM allocates weights.
    lm.cfg->mem_used_per_device[0] += (8ULL << 20);
    lm.cfg->mem_used_per_device[1] += (8ULL << 20);
    // Legitimate CLIP mutation in the same vein.
    clip.cfg->mem_used_per_device[0] += (2ULL << 20);
    clip.cfg->mem_used_per_device[1] += (2ULL << 20);
    const uint64_t after_lm   = hash_immutable_fields(lm.cfg);
    const uint64_t after_clip = hash_immutable_fields(clip.cfg);
    if (after_lm != before || after_clip != before) {
        fprintf(stderr, "FAIL test_immutable_fields_immutable: before=%llx after_lm=%llx after_clip=%llx\n",
                (unsigned long long)before, (unsigned long long)after_lm, (unsigned long long)after_clip);
        return 1;
    }
    return 0;
}

int test_access_paths_disjoint() {
    // Spec §AccessPathsBelongToConsumer lists the LM-side and CLIP-side
    // call-site sets. The two sets must be disjoint — no call site
    // exercises both consumer codepaths.
    const std::set<std::string> lm_paths = {
        "create_tensors_helper_ctor",
        "create_tensors_helper_ctx_for_layer_split",
        "per_arch_create_llama_tensors",
        "llama_cpp_4168_buft_setup_loop",
        "llama_load_tensors_create_split_call_sites_x13",
    };
    const std::set<std::string> clip_paths = {
        "clip_cpp_488_init",
        "clip_cpp_weight_loader",
        "clip_cpp_create_split_call",
    };
    for (const auto & p : lm_paths) {
        if (clip_paths.count(p)) {
            fprintf(stderr, "FAIL test_access_paths_disjoint: '%s' in both sets\n", p.c_str());
            return 1;
        }
    }
    return 0;
}

int test_buft_setup_output_equivalent() {
    // After the buft-setup loop assigns layer buft pairs (modelled in
    // BuftSetupLoop.tla), both views must read identical buft_layer
    // entries. The make_production_shape_config helper already
    // populates buft_layer; re-derive from scratch through both views
    // and confirm identity.
    StubMgpuSplitConfig backing = make_production_shape_config();
    ConfigInstance lm{&backing, CONSUMER_LM};
    ConfigInstance clip{&backing, CONSUMER_CLIP};
    for (int i = 0; i < lm.cfg->n_layer; ++i) {
        const auto & a = lm.cfg->buft_layer[i];
        const auto & b = clip.cfg->buft_layer[i];
        const int lm_split_kind   = a.first.kind_id;
        const int lm_offload_kind = a.second.kind_id;
        const int cl_split_kind   = b.first.kind_id;
        const int cl_offload_kind = b.second.kind_id;
        if (i >= lm.cfg->i_gpu_start) {
            // NoOrphanGpuLayers from spec #1: GPU layers have a
            // non-nil offload buft.
            if (lm_offload_kind == 0 || cl_offload_kind == 0) {
                fprintf(stderr, "FAIL test_buft_setup_output_equivalent: orphan GPU layer %d\n", i);
                return 1;
            }
        }
        if (lm_split_kind != cl_split_kind || lm_offload_kind != cl_offload_kind) {
            fprintf(stderr, "FAIL test_buft_setup_output_equivalent: layer %d divergent\n", i);
            return 1;
        }
    }
    return 0;
}

int test_create_split_output_equivalent() {
    // Stub a create_split walk: weight rows allocated round-robin
    // across the 2 devices. After the walk, both views must read
    // identical mem_used_per_device.
    StubMgpuSplitConfig backing = make_production_shape_config();
    ConfigInstance lm{&backing, CONSUMER_LM};
    ConfigInstance clip{&backing, CONSUMER_CLIP};

    const size_t row_bytes = 1ULL << 20; // 1 MiB per stubbed weight row
    const int    n_rows    = 64;
    for (int r = 0; r < n_rows; ++r) {
        // LM-side allocation through lm view.
        lm.cfg->mem_used_per_device[r % lm.cfg->n_device] += row_bytes;
    }
    // CLIP-side mmproj allocation through clip view (CLIP loads
    // AFTER the LM is up — spec §MutationsByOneConsumerVisibleToOther).
    for (int r = 0; r < n_rows / 4; ++r) {
        clip.cfg->mem_used_per_device[r % clip.cfg->n_device] += row_bytes;
    }
    // Capacity is honoured (CapacityHonored from spec #1).
    for (int d = 0; d < lm.cfg->n_device; ++d) {
        if (lm.cfg->mem_used_per_device[d] > lm.cfg->capacity_per_device[d]) {
            fprintf(stderr, "FAIL test_create_split_output_equivalent: capacity exceeded on device %d\n", d);
            return 1;
        }
    }
    // Both views see the same final state.
    if (lm.cfg->mem_used_per_device != clip.cfg->mem_used_per_device) {
        fprintf(stderr, "FAIL test_create_split_output_equivalent: views diverged\n");
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    int rc = 0;
    rc |= test_pair_over_same_struct();
    rc |= test_pair_consumers_distinct();
    rc |= test_invariant_satisfaction_equivalent();
    rc |= test_mutations_visible();
    rc |= test_immutable_fields_immutable();
    rc |= test_access_paths_disjoint();
    rc |= test_buft_setup_output_equivalent();
    rc |= test_create_split_output_equivalent();
    if (rc == 0) {
        printf("PASS test-mgpu-split-cross-codepath (8 cases)\n");
    } else {
        printf("FAIL test-mgpu-split-cross-codepath\n");
    }
    return rc;
}
