// CUDA graph cache: distinct shape coverage in hit_counter dump.
//
// Drives N distinct shapes and asserts the hit_counter probe dump
// surfaces >= N distinct shape_key values.
//
// RED until the hit counter on cache entries and the JSONL dump
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
#include <set>
#include <string>
#include <vector>

using json = nlohmann::json;

constexpr int N_DISTINCT_SHAPES = 100;

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
    char dir_template[] = "cuda-graph-probe-distinct-shapes-XXXXXX";
    if (mkdtemp(dir_template) == nullptr) {
        fprintf(stderr, "mkdtemp failed: %s\n", strerror(errno));
        return 1;
    }
    // Force a high cap so we don't evict before the assertion runs.
    setenv("GGML_CUDA_GRAPH_MAX", "256", 1);
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

    printf("=== test-cuda-graph-probe-distinct-shapes ===\n");
    printf("  drove %d distinct shapes\n", N_DISTINCT_SHAPES);
    printf("  probe_flush rc = %d\n", flush_rc);

    if (flush_rc != 0) {
        printf("RESULT: RED — probe_flush returned %d (stub until probe code lands)\n", flush_rc);
        return 1;
    }

    auto files = list_jsonl(dir_template);
    if (files.empty()) {
        printf("RESULT: RED — no JSONL files written\n");
        return 1;
    }

    std::set<std::string> distinct_shape_keys;
    for (const auto & path : files) {
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            try {
                auto j = json::parse(line);
                if (j.value("probe", "") != "hit_counter") continue;
                std::string sk = j.value("shape_key", "");
                if (!sk.empty()) distinct_shape_keys.insert(sk);
            } catch (...) {
                printf("RESULT: RED — malformed JSONL line\n");
                return 1;
            }
        }
    }

    printf("  distinct shape_keys observed = %zu (expected >= %d)\n",
           distinct_shape_keys.size(), N_DISTINCT_SHAPES);
    if ((int) distinct_shape_keys.size() < N_DISTINCT_SHAPES) {
        printf("RESULT: RED — fewer distinct shape_keys than driven shapes\n");
        return 1;
    }
    printf("RESULT: PASS\n");
    return 0;
}
