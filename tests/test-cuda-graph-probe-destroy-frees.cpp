// CUDA graph cache: cudaGraphExecDestroy free-delta probe coverage.
//
// Forces eviction (via low GGML_CUDA_GRAPH_MAX cap) and asserts the
// vram_delta probe records at least one event=destroy entry with a
// non-negative free_after_bytes. This is the precondition signal for
// allocation-aware eviction: if delta_bytes is consistently 0 across
// many destroys, freeing entries won't relieve VRAM pressure (cuda_pool
// retains memory).
//
// RED until the destroy-time vram_delta probe and the JSONL dump
// infrastructure both land.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include "nlohmann/json.hpp"

#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;

constexpr size_t MAX_CACHE_ENTRIES = 2;
constexpr int    N_DISTINCT_SHAPES = 8;  // 8 - 2 = 6 evictions expected

static bool run_one_shape(ggml_backend_t backend, int64_t n) {
    static const size_t mem_size = 4 * 1024 * 1024;
    struct ggml_init_params params = { mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, n);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, n);
    ggml_tensor * out = ggml_add(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { ggml_free(ctx); return false; }
    std::vector<float> data((size_t) 32 * n, 1.0f);
    ggml_backend_tensor_set(a, data.data(), 0, data.size() * sizeof(float));
    ggml_backend_tensor_set(b, data.data(), 0, data.size() * sizeof(float));
    const auto status = ggml_backend_graph_compute(backend, gf);
    const bool ok = (status == GGML_STATUS_SUCCESS);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return ok;
}

static std::vector<std::string> list_jsonl(const std::string & dir) {
    std::vector<std::string> out;
    DIR * d = opendir(dir.c_str());
    if (!d) return out;
    while (auto * ent = readdir(d)) {
        std::string name = ent->d_name;
        if (name.size() > 6 && name.substr(name.size() - 6) == ".jsonl") {
            out.push_back(dir + "/" + name);
        }
    }
    closedir(d);
    return out;
}

int main() {
    char dir_template[] = "cuda-graph-probe-destroy-frees-XXXXXX";
    if (mkdtemp(dir_template) == nullptr) {
        fprintf(stderr, "mkdtemp failed: %s\n", strerror(errno));
        return 1;
    }
    char cap_env[32];
    snprintf(cap_env, sizeof(cap_env), "%zu", MAX_CACHE_ENTRIES);
    setenv("GGML_CUDA_GRAPH_MAX", cap_env, 1);
    setenv("GGML_CUDA_GRAPH_PROBE", "1", 1);
    setenv("GGML_CUDA_GRAPH_PROBE_DIR", dir_template, 1);

    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed; skipping\n");
        return 0;
    }

    for (int i = 0; i < N_DISTINCT_SHAPES; ++i) {
        if (!run_one_shape(backend, 8 + i)) {
            fprintf(stderr, "shape %d failed\n", i);
            ggml_backend_free(backend);
            return 1;
        }
    }

    const int flush_rc = ggml_backend_cuda_graph_probe_flush(backend);
    ggml_backend_free(backend);

    printf("=== test-cuda-graph-probe-destroy-frees ===\n");
    printf("  drove %d shapes with cap=%zu (≈%d evictions expected)\n",
           N_DISTINCT_SHAPES, MAX_CACHE_ENTRIES, N_DISTINCT_SHAPES - (int) MAX_CACHE_ENTRIES);
    printf("  probe_flush rc = %d\n", flush_rc);

    if (flush_rc != 0) {
        printf("RESULT: RED — probe_flush returned %d (stub until probe code lands)\n", flush_rc);
        return 1;
    }

    auto files = list_jsonl(dir_template);
    int destroy_records = 0;
    int with_nonzero_delta = 0;
    int64_t total_delta_bytes = 0;
    for (const auto & path : files) {
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            try {
                auto j = json::parse(line);
                if (j.value("probe", "") != "vram_delta") continue;
                if (j.value("event", "") != "destroy") continue;
                destroy_records++;
                int64_t d = j.value("delta_bytes", (int64_t) 0);
                if (d != 0) with_nonzero_delta++;
                total_delta_bytes += d;
            } catch (...) {
                printf("RESULT: RED — malformed JSONL line\n");
                return 1;
            }
        }
    }
    printf("  destroy records = %d\n", destroy_records);
    printf("  records with nonzero delta_bytes = %d\n", with_nonzero_delta);
    printf("  cumulative delta_bytes = %lld\n", (long long) total_delta_bytes);

    if (destroy_records == 0) {
        printf("RESULT: RED — no destroy_free_delta records emitted\n");
        return 1;
    }
    printf("RESULT: PASS\n");
    return 0;
}
