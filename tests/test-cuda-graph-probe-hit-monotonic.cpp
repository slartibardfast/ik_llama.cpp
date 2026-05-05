// CUDA graph cache: per-entry hit counter monotonicity.
//
// Drives N repeated submissions of the same shape, flushes the probe
// dump, and asserts that some hit_counter record reports hits_total >= N.
//
// RED until the hit counter on cache entries and the JSONL dump
// infrastructure both land.
//
// Build: gated on GGML_CUDA via tests/CMakeLists.txt.

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
#include <sys/stat.h>
#include <vector>

using json = nlohmann::json;

constexpr int N_REPEATS = 100;

static bool run_one_shape(ggml_backend_t backend, int64_t n) {
    static const size_t mem_size = 4 * 1024 * 1024;
    struct ggml_init_params params = { mem_size, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, n);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, n);
    ggml_tensor * out = ggml_add(ctx, a, b);
    ggml_set_name(out, "out");

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
    char dir_template[] = "cuda-graph-probe-hit-monotonic-XXXXXX";
    if (mkdtemp(dir_template) == nullptr) {
        fprintf(stderr, "mkdtemp failed: %s\n", strerror(errno));
        return 1;
    }
    setenv("GGML_CUDA_GRAPH_PROBE", "1", 1);
    setenv("GGML_CUDA_GRAPH_PROBE_DIR", dir_template, 1);

    ggml_backend_t backend = ggml_backend_cuda_init(0, nullptr);
    if (!backend) {
        fprintf(stderr, "ggml_backend_cuda_init failed; skipping\n");
        return 0;
    }

    for (int i = 0; i < N_REPEATS; ++i) {
        if (!run_one_shape(backend, /*n=*/8)) {
            fprintf(stderr, "submit %d failed\n", i);
            ggml_backend_free(backend);
            return 1;
        }
    }

    const int flush_rc = ggml_backend_cuda_graph_probe_flush(backend);
    ggml_backend_free(backend);

    printf("=== test-cuda-graph-probe-hit-monotonic ===\n");
    printf("  drove %d same-shape submits (n=8)\n", N_REPEATS);
    printf("  probe_flush rc = %d\n", flush_rc);

    if (flush_rc != 0) {
        printf("RESULT: RED — probe_flush did not write a dump (stub returns -1 until probe code lands)\n");
        return 1;
    }

    auto files = list_jsonl(dir_template);
    if (files.empty()) {
        printf("RESULT: RED — no JSONL files written under %s\n", dir_template);
        return 1;
    }

    int64_t max_hits = 0;
    for (const auto & path : files) {
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            try {
                auto j = json::parse(line);
                if (j.value("probe", "") != "hit_counter") continue;
                int64_t h = j.value("hits_total", (int64_t) 0);
                if (h > max_hits) max_hits = h;
            } catch (...) {
                printf("RESULT: RED — malformed JSONL line in %s: %s\n", path.c_str(), line.c_str());
                return 1;
            }
        }
    }

    printf("  max hits_total across all entries = %lld (expected >= %d)\n", (long long) max_hits, N_REPEATS);
    if (max_hits < N_REPEATS) {
        printf("RESULT: RED — hit counter not monotonic enough\n");
        return 1;
    }
    printf("RESULT: PASS\n");
    return 0;
}
