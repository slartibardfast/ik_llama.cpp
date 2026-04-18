// Isolate the intermediate-rollback bug in MTP MTP-IR with a deterministic
// three-path state comparison on a real hybrid model.
//
// Kernel intermediate test (test-gdn-intermediate-state.cpp) passes 24/24 at
// the 0.8B model's dimensions on ik_llama.cpp. But we still need a graph-level
// verification that llama_rollback_delta_net_state correctly restores the
// post-T recurrent state from per-token intermediates captured during a 2-token
// [T, D] batch. This test is that verification.
//
// Ported from polaris /home/llm/src/qwen35-mtp/tests/test-intermediate-rollback.cpp.
// Adaptations for ik_llama:
//   - llama_state_seq_{get,set}_{size,data}_ext → no _ext suffix
//   - llama_memory_seq_rm(llama_get_memory(ctx), ...) → llama_kv_cache_seq_rm(ctx, ...)
//   - common_params → gpt_params; common_init_from_params → llama_init_from_gpt_params
//   - params.sampling.seed → params.seed; params.model.path → params.model
//
// Three paths, all starting from the same post-prompt baseline S0:
//   Path A (ground truth)   : decode [T]                               → S_A
//   Path B (intermediate)   : decode [T, D] + rollback(token=0)        → S_B (claimed = S_A)
//   Path C (snapshot+rerun) : decode [T, D] + restore(S0) + decode [T] → S_C
//
// Outcome decoder:
//   A == B on both backends → rollback correct, server bug elsewhere
//   A != B on Vulkan only   → Vulkan-specific (conv_input_keep, scheduler, MMVQ)
//   A != B on CPU + Vulkan  → generic rollback logic bug
//   A != C (either backend) → snapshot mechanism broken (unlikely)

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static llama_token greedy(llama_context * ctx) {
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
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

struct diff_report {
    float    max_diff        = 0.0f;
    size_t   first_diff_at   = (size_t) -1;
    size_t   n_diff_floats   = 0;
    size_t   total_floats    = 0;
};

static diff_report compare_states(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    diff_report r;
    const size_t n = std::min(a.size(), b.size());
    r.total_floats = n / sizeof(float);
    for (size_t i = 0; i + sizeof(float) <= n; i += sizeof(float)) {
        float fa, fb;
        std::memcpy(&fa, a.data() + i, sizeof(float));
        std::memcpy(&fb, b.data() + i, sizeof(float));
        const float d = std::fabs(fa - fb);
        if (d > 1e-9f) {
            r.n_diff_floats++;
            if (r.first_diff_at == (size_t) -1) {
                r.first_diff_at = i / sizeof(float);
            }
        }
        if (d > r.max_diff) {
            r.max_diff = d;
        }
    }
    if (a.size() != b.size()) {
        fprintf(stderr, "  [warn] size mismatch: %zu vs %zu\n", a.size(), b.size());
    }
    return r;
}

// --- PARTIAL_ONLY serialization parser for ik_llama ---
// Format (from src/llama.cpp write_kv_cache_data, flags=PARTIAL_ONLY):
//   cell_count(u32)
//   per cell: pos(i32) + n_seq_id(u32) + seq_ids[n_seq_id](i32)
//   v_state(u32) + n_layer(u32)
//   K section: per layer: k_type(i32) + k_size_row(u64) + [data if k_type != -1]
//     (PARTIAL_ONLY → has_k_cache=false → k_type=-1, k_size_row=0, no data)
//   V section (v_state=0): per layer: v_type(i32) + v_size_row(u64) + [data]
//     (PARTIAL_ONLY → same: v_type=-1, v_size_row=0, no data)
//   V section (v_state=1): per layer: v_type(i32) + v_size_el(u32) + n_embd_v_gqa(u32) + [data]
//   V section (v_state=2): no V section written
//   qnext_state(u32)
//   if qnext_state: per layer: s_type(i32) + s_size_row(u64) + s_rows(u32) + [data if has_s_cache && s_rows>0]
struct layer_entry {
    size_t data_off;
    size_t data_size;
    char   kind;  // 'r' or 's' (ik_llama has only 's', no 'r' — SSM unified cache)
};

static std::vector<layer_entry> parse_partial_only(const std::vector<uint8_t> & buf) {
    std::vector<layer_entry> out;
    size_t off = 0;
    if (off + 4 > buf.size()) return out;
    uint32_t cell_count;
    std::memcpy(&cell_count, buf.data() + off, 4); off += 4;

    // Cell metadata
    for (uint32_t c = 0; c < cell_count; c++) {
        if (off + 8 > buf.size()) return out;
        off += 4;  // pos
        uint32_t n_seq_id;
        std::memcpy(&n_seq_id, buf.data() + off, 4); off += 4;
        off += (size_t) n_seq_id * 4;
    }

    // v_state + n_layer
    if (off + 8 > buf.size()) return out;
    uint32_t v_state, n_layer;
    std::memcpy(&v_state, buf.data() + off, 4); off += 4;
    std::memcpy(&n_layer, buf.data() + off, 4); off += 4;

    // K section — skip (PARTIAL_ONLY → all type=-1, size_row=0, no data)
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (off + 12 > buf.size()) return out;
        int32_t k_type;  std::memcpy(&k_type, buf.data() + off, 4); off += 4;
        uint64_t k_size_row; std::memcpy(&k_size_row, buf.data() + off, 8); off += 8;
        if (k_type != -1) {
            // unexpected under PARTIAL_ONLY but handle defensively
            off += (size_t) k_size_row * cell_count;
        }
    }

    // V section
    if (v_state == 0) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (off + 12 > buf.size()) return out;
            int32_t v_type; std::memcpy(&v_type, buf.data() + off, 4); off += 4;
            uint64_t v_size_row; std::memcpy(&v_size_row, buf.data() + off, 8); off += 8;
            if (v_type != -1) off += (size_t) v_size_row * cell_count;
        }
    } else if (v_state == 1) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (off + 12 > buf.size()) return out;
            int32_t v_type; std::memcpy(&v_type, buf.data() + off, 4); off += 4;
            off += 4;  // v_size_el
            uint32_t n_embd_v_gqa; std::memcpy(&n_embd_v_gqa, buf.data() + off, 4); off += 4;
            if (v_type != -1) {
                // skip is complex; PARTIAL_ONLY shouldn't hit this path
                fprintf(stderr, "[parse_partial_only] unexpected v_state=1 with data\n");
                return out;
            }
        }
    }

    // qnext_state flag
    if (off + 4 > buf.size()) return out;
    uint32_t qnext_state;
    std::memcpy(&qnext_state, buf.data() + off, 4); off += 4;

    if (qnext_state) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (off + 16 > buf.size()) return out;
            int32_t s_type;  std::memcpy(&s_type, buf.data() + off, 4); off += 4;
            uint64_t s_size_row; std::memcpy(&s_size_row, buf.data() + off, 8); off += 8;
            uint32_t s_rows; std::memcpy(&s_rows, buf.data() + off, 4); off += 4;
            if (s_type != -1 && s_rows > 0) {
                const size_t data_size = (size_t) s_rows * (size_t) s_size_row;
                layer_entry e;
                e.data_off  = off;
                e.data_size = data_size;
                e.kind      = 's';
                out.push_back(e);
                off += data_size;
            }
        }
    }
    return out;
}

static void report_per_layer_diff(const char * tag, const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    const auto ea = parse_partial_only(a);
    const auto eb = parse_partial_only(b);
    if (ea.size() != eb.size() || ea.empty()) {
        fprintf(stderr, "[%s] per-layer: entry count mismatch or empty (%zu vs %zu)\n", tag, ea.size(), eb.size());
        return;
    }
    fprintf(stderr, "[%s] per-layer diff (layer index = position within valid-recurrent group):\n", tag);
    int r_idx = 0, s_idx = 0;
    float worst_r = 0.0f, worst_s = 0.0f;
    int   worst_r_at = -1, worst_s_at = -1;
    size_t worst_r_first_byte = (size_t)-1;
    size_t worst_s_first_byte = (size_t)-1;
    const std::vector<uint8_t> * worst_s_a = nullptr;
    const std::vector<uint8_t> * worst_s_b = nullptr;
    size_t worst_s_data_off_a = 0;
    size_t worst_s_data_off_b = 0;
    for (size_t i = 0; i < ea.size(); i++) {
        if (ea[i].kind != eb[i].kind || ea[i].data_size != eb[i].data_size) {
            fprintf(stderr, "  entry %zu: structural mismatch\n", i);
            continue;
        }
        const size_t n_floats = ea[i].data_size / sizeof(float);
        float max_d = 0.0f;
        size_t n_diff = 0;
        size_t first_diff = (size_t)-1;
        for (size_t f = 0; f < n_floats; f++) {
            float va, vb;
            std::memcpy(&va, a.data() + ea[i].data_off + f * sizeof(float), sizeof(float));
            std::memcpy(&vb, b.data() + eb[i].data_off + f * sizeof(float), sizeof(float));
            const float d = std::fabs(va - vb);
            if (d > 1e-9f) { n_diff++; if (first_diff == (size_t)-1) first_diff = f; }
            if (d > max_d) max_d = d;
        }
        const int layer_i = (ea[i].kind == 'r') ? r_idx++ : s_idx++;
        fprintf(stderr, "  %c[%2d]: max_diff=%.3e  n_diff=%zu / %zu  first_at_float=%zd\n",
                ea[i].kind, layer_i, max_d, n_diff, n_floats,
                first_diff == (size_t)-1 ? (ssize_t)-1 : (ssize_t)first_diff);
        if (ea[i].kind == 'r' && max_d > worst_r) { worst_r = max_d; worst_r_at = layer_i; worst_r_first_byte = first_diff; }
        if (ea[i].kind == 's' && max_d > worst_s) {
            worst_s = max_d; worst_s_at = layer_i; worst_s_first_byte = first_diff;
            worst_s_a = &a; worst_s_b = &b;
            worst_s_data_off_a = ea[i].data_off; worst_s_data_off_b = eb[i].data_off;
        }
    }
    fprintf(stderr, "[%s] worst R: layer %d max_diff=%.3e | worst S: layer %d max_diff=%.3e\n",
            tag, worst_r_at, worst_r, worst_s_at, worst_s);
    (void)worst_r_first_byte;
    if (worst_s_a && worst_s_first_byte != (size_t)-1) {
        const size_t off_a = worst_s_data_off_a + worst_s_first_byte * sizeof(float);
        const size_t off_b = worst_s_data_off_b + worst_s_first_byte * sizeof(float);
        fprintf(stderr, "[%s] S-region first-diff dump (layer %d, float %zu):\n", tag, worst_s_at, worst_s_first_byte);
        fprintf(stderr, "  A: "); for (int k = 0; k < 8; k++) { float v; std::memcpy(&v, worst_s_a->data() + off_a + k*sizeof(float), sizeof(float)); fprintf(stderr, "%9.4f ", v); } fprintf(stderr, "\n");
        fprintf(stderr, "  B: "); for (int k = 0; k < 8; k++) { float v; std::memcpy(&v, worst_s_b->data() + off_b + k*sizeof(float), sizeof(float)); fprintf(stderr, "%9.4f ", v); } fprintf(stderr, "\n");
    }
}

// Snapshot the sequence with a freshly-queried size. ik_llama's serialized
// PARTIAL_ONLY format size can vary between calls (more cells after a decode
// → more cell metadata), so S0's size isn't valid for later snapshots.
static std::vector<uint8_t> snap(llama_context * ctx, llama_state_seq_flags flags) {
    const size_t sz = llama_state_seq_get_size(ctx, 0, flags);
    std::vector<uint8_t> buf(sz);
    const size_t got = llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0, flags);
    if (got != sz) {
        fprintf(stderr, "  [warn] snap: wrote %zu of %zu\n", got, sz);
    }
    return buf;
}

static void print_report(const char * name, const diff_report & r, float tolerance) {
    const bool ok = r.max_diff <= tolerance;
    fprintf(stderr, "[%s] max_diff=%.6e n_diff=%zu / %zu first_at=",
            name, r.max_diff, r.n_diff_floats, r.total_floats);
    if (r.first_diff_at == (size_t) -1) {
        fprintf(stderr, "none");
    } else {
        fprintf(stderr, "%zu", r.first_diff_at);
    }
    fprintf(stderr, " → %s\n", ok ? "PASS" : "FAIL");
}

int main(int argc, char ** argv) {
    gpt_params params;
    params.prompt    = "The capital of France is";
    params.n_predict = 0;
    params.seed      = 42;
    params.n_ctx     = 2048;
    params.n_parallel = 1;
    params.has_mtp   = true;   // ik_llama: enable MTP head at context init
    // Disable graph reuse: the rollback test's A-vs-B comparison requires Path
    // A's 1-token decode and Path B's 2-token decode to both land exactly on
    // "state after T" (no padding). Graph reuse defaults to a worst-case ubatch
    // (e.g. 32) and the DeltaNet kernel has no mask → padding positions
    // contaminate the state. Forcing per-batch-size graph rebuild eliminates
    // padding. See plan /home/llm/.claude/plans/rosy-churning-dove.md.
    params.graph_reuse = false;

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    llama_init_result init = llama_init_from_gpt_params(params);
    llama_model   * model = init.model;
    llama_context * ctx   = init.context;
    if (!model || !ctx) {
        fprintf(stderr, "failed to init\n");
        return 1;
    }

    fprintf(stderr, "=== test-intermediate-rollback ===\n");
    fprintf(stderr, "model: %s\n", params.model.c_str());
    fprintf(stderr, "prompt: %s\n", params.prompt.c_str());
    fprintf(stderr, "n_gpu_layers: %d (0 = CPU, 999 = offload all)\n", params.n_gpu_layers);

    // --- 1. Process the prompt to establish a non-trivial baseline state ---
    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, params.prompt, true);
    const int n_prompt = (int) prompt_tokens.size();
    fprintf(stderr, "prompt tokens: %d\n", n_prompt);

    llama_batch batch = llama_batch_init(params.n_ctx, 0, 1);
    common_batch_clear(batch);
    for (int i = 0; i < n_prompt; i++) {
        common_batch_add(batch, prompt_tokens[i], i, {0}, i == n_prompt - 1);
    }
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "FAIL: prompt decode\n");
        return 2;
    }

    const int pos_of_T = n_prompt;
    const llama_token T = greedy(ctx);
    const llama_token D = 42;
    fprintf(stderr, "T (greedy next after prompt) = %d\n", T);
    fprintf(stderr, "D (fixed unrelated draft)    = %d\n", D);
    fprintf(stderr, "pos_of_T = %d\n", pos_of_T);

    // --- 2. Snapshot the post-prompt (baseline S0) state ---
    const llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    std::vector<uint8_t> S0 = snap(ctx, flags);
    if (S0.empty()) { fprintf(stderr, "FAIL: baseline snapshot\n"); return 4; }
    fprintf(stderr, "baseline snapshot: %zu bytes\n", S0.size());

    // --- 3. Path A: ground truth = decode [T] from S0 ---
    fprintf(stderr, "\n--- Path A: ground truth 1-token decode [T=%d] ---\n", T);
    common_batch_clear(batch);
    common_batch_add(batch, T, pos_of_T, {0}, true);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "FAIL: Path A decode\n"); return 10; }

    std::vector<uint8_t> S_A = snap(ctx, flags);
    if (S_A.empty()) { fprintf(stderr, "FAIL: S_A snapshot\n"); return 11; }

    // Restore S0 and trim KV back to pos_of_T
    if (llama_state_seq_set_data(ctx, S0.data(), S0.size(), 0, flags) == 0) {
        fprintf(stderr, "FAIL: restore S0 before path B\n"); return 12;
    }
    llama_kv_cache_seq_rm(ctx, 0, pos_of_T, -1);

    // --- 4. Path B: 2-token batch [T, D] + intermediate rollback(token=0) ---
    fprintf(stderr, "\n--- Path B: 2-token batch + llama_rollback_delta_net_state ---\n");
    common_batch_clear(batch);
    common_batch_add(batch, T, pos_of_T,     {0}, true);
    common_batch_add(batch, D, pos_of_T + 1, {0}, true);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "FAIL: Path B decode\n"); return 20; }

    const int token_idx_rb = std::getenv("ROLLBACK_IDX") ? std::atoi(std::getenv("ROLLBACK_IDX")) : 0;
    const bool rb_ok = llama_rollback_delta_net_state(ctx, token_idx_rb, /*seq_id=*/0, /*target_pos=*/pos_of_T);
    fprintf(stderr, "llama_rollback_delta_net_state: %s\n", rb_ok ? "ok" : "FAILED");
    llama_kv_cache_seq_rm(ctx, 0, pos_of_T + 1, -1);

    std::vector<uint8_t> S_B = snap(ctx, flags);
    if (S_B.empty()) { fprintf(stderr, "FAIL: S_B snapshot\n"); return 21; }

    // Restore S0 for path D (null-rollback diagnostic)
    if (llama_state_seq_set_data(ctx, S0.data(), S0.size(), 0, flags) == 0) {
        fprintf(stderr, "FAIL: restore S0 before path D\n"); return 22;
    }
    llama_kv_cache_seq_rm(ctx, 0, pos_of_T, -1);

    // --- 4b. Path D: 2-token batch WITHOUT rollback (diagnostic) ---
    fprintf(stderr, "\n--- Path D: 2-token batch (NO rollback) — diagnostic baseline ---\n");
    common_batch_clear(batch);
    common_batch_add(batch, T, pos_of_T,     {0}, true);
    common_batch_add(batch, D, pos_of_T + 1, {0}, true);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "FAIL: Path D decode\n"); return 23; }

    std::vector<uint8_t> S_D = snap(ctx, flags);
    if (S_D.empty()) { fprintf(stderr, "FAIL: S_D snapshot\n"); return 24; }

    // Restore S0 for path C
    if (llama_state_seq_set_data(ctx, S0.data(), S0.size(), 0, flags) == 0) {
        fprintf(stderr, "FAIL: restore S0 before path C\n"); return 25;
    }
    llama_kv_cache_seq_rm(ctx, 0, pos_of_T, -1);

    // --- 5. Path C: snapshot+rerun control (known-good) ---
    fprintf(stderr, "\n--- Path C: 2-token batch + snapshot restore + rerun [T] ---\n");
    common_batch_clear(batch);
    common_batch_add(batch, T, pos_of_T,     {0}, true);
    common_batch_add(batch, D, pos_of_T + 1, {0}, true);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "FAIL: Path C first decode\n"); return 30; }

    if (llama_state_seq_set_data(ctx, S0.data(), S0.size(), 0, flags) == 0) {
        fprintf(stderr, "FAIL: Path C restore\n"); return 31;
    }
    llama_kv_cache_seq_rm(ctx, 0, pos_of_T, -1);

    common_batch_clear(batch);
    common_batch_add(batch, T, pos_of_T, {0}, true);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "FAIL: Path C rerun\n"); return 32; }

    std::vector<uint8_t> S_C = snap(ctx, flags);
    if (S_C.empty()) { fprintf(stderr, "FAIL: S_C snapshot\n"); return 33; }

    // --- 6. Compare ---
    fprintf(stderr, "\n=== Results ===\n");
    // CPU is bit-identical (max_diff=0). Vulkan accumulates ~5e-6 per-layer kernel
    // noise across 18 layers → residual ~1e-3 even with GGML_VK_DISABLE_MMVQ=1 on
    // NAVI21. Pre-Step-8 bug in polaris was 19.0; without MMVQ workaround it's 0.131.
    const float tol = 5e-3f;
    const diff_report ac = compare_states(S_A, S_C);
    const diff_report ab = compare_states(S_A, S_B);
    const diff_report bc = compare_states(S_B, S_C);
    const diff_report ad = compare_states(S_A, S_D);
    const diff_report bd = compare_states(S_B, S_D);

    print_report("A vs C (snapshot+rerun sanity)",        ac, tol);
    print_report("A vs B (intermediate vs ground-truth)", ab, tol);
    print_report("B vs C (intermediate vs snapshot+rerun)", bc, tol);
    print_report("A vs D (ground-truth vs no-rollback)",  ad, tol);
    print_report("B vs D (intermediate vs no-rollback)",  bd, tol);

    // Per-layer diagnostic for the "A vs B" comparison — which layer diverges first?
    fprintf(stderr, "\n");
    report_per_layer_diff("A vs B", S_A, S_B);
    fprintf(stderr, "\n");
    report_per_layer_diff("B vs D", S_B, S_D);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);

    int rc = 0;
    if (ac.max_diff > tol) { fprintf(stderr, "\nFAIL: snapshot+rerun doesn't match ground truth — snapshot mechanism broken\n"); rc = 40; }
    if (ab.max_diff > tol) { fprintf(stderr, "\nFAIL: intermediate rollback doesn't match ground truth — rollback has a bug\n"); rc = 41; }
    if (rc == 0) { fprintf(stderr, "\nALL PASSES ✓ — intermediate rollback is correct on this backend\n"); }
    return rc;
}
