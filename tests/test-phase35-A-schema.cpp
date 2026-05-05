// Phase 35 A.T2: probe dump schema validation.
//
// Drives a small workload, flushes, and walks every JSONL record asserting
// the per-probe required fields per PHASE35-GRAPH-CACHE-REDESIGN.md §3.3.
// Common required fields: ts_ns, runid, backend, probe.
//
// RED at A.0 (probe_flush stub returns -1; no dump exists).
// GREEN after A.1 + A.7.

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

static bool record_has(const json & j, const char * field) {
    return j.contains(field) && !j[field].is_null();
}

// Returns "" on valid record, otherwise a human-readable error string naming
// the missing/wrong field.
static std::string validate(const json & j) {
    if (!record_has(j, "ts_ns")  || !j["ts_ns"].is_number_integer())  return "ts_ns missing/non-int";
    if (!record_has(j, "runid")  || !j["runid"].is_string())          return "runid missing/non-string";
    if (!record_has(j, "backend")|| !j["backend"].is_string())        return "backend missing/non-string";
    if (!record_has(j, "probe")  || !j["probe"].is_string())          return "probe missing/non-string";

    const std::string probe = j["probe"];
    if (probe == "hit_counter") {
        if (!record_has(j, "topology_key")) return "hit_counter missing topology_key";
        if (!record_has(j, "shape_key"))    return "hit_counter missing shape_key";
        if (!record_has(j, "hits_total") || !j["hits_total"].is_number_integer()) return "hit_counter missing hits_total";
    } else if (probe == "timing") {
        if (!record_has(j, "event"))        return "timing missing event";
        if (!record_has(j, "duration_us") || !j["duration_us"].is_number()) return "timing missing duration_us";
    } else if (probe == "vram_delta") {
        if (!record_has(j, "event"))        return "vram_delta missing event";
        if (!record_has(j, "delta_bytes") || !j["delta_bytes"].is_number_integer()) return "vram_delta missing delta_bytes";
    } else if (probe == "update_failures") {
        if (!record_has(j, "topology_key"))    return "update_failures missing topology_key";
        if (!record_has(j, "shape_key_old"))   return "update_failures missing shape_key_old";
        if (!record_has(j, "shape_key_new"))   return "update_failures missing shape_key_new";
    } else if (probe == "disable_too_many") {
        if (!record_has(j, "topology_key"))            return "disable_too_many missing topology_key";
        if (!record_has(j, "consecutive_updates") || !j["consecutive_updates"].is_number_integer())
            return "disable_too_many missing consecutive_updates";
    } else {
        return "unknown probe type: " + probe;
    }
    return "";
}

int main() {
    char dir_template[] = "phase35-A-schema-XXXXXX";
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

    for (int i = 0; i < 4; ++i) {
        if (!run_one_shape(backend, 8 + i)) {
            fprintf(stderr, "shape %d failed\n", i);
            ggml_backend_free(backend);
            return 1;
        }
    }

    const int flush_rc = ggml_backend_cuda_graph_probe_flush(backend);
    ggml_backend_free(backend);

    printf("=== test-phase35-A-schema ===\n");
    printf("  probe_flush rc = %d\n", flush_rc);

    if (flush_rc != 0) {
        printf("RESULT: RED — probe_flush returned %d (stub until A.7)\n", flush_rc);
        return 1;
    }

    auto files = list_jsonl(dir_template);
    if (files.empty()) {
        printf("RESULT: RED — no JSONL files written\n");
        return 1;
    }

    int n_records = 0;
    std::set<std::string> probe_types;
    for (const auto & path : files) {
        std::ifstream f(path);
        std::string line;
        int line_no = 0;
        while (std::getline(f, line)) {
            line_no++;
            if (line.empty()) continue;
            try {
                auto j = json::parse(line);
                std::string err = validate(j);
                if (!err.empty()) {
                    printf("RESULT: RED — %s:%d: %s\n", path.c_str(), line_no, err.c_str());
                    return 1;
                }
                n_records++;
                probe_types.insert(j.value("probe", ""));
            } catch (const std::exception & e) {
                printf("RESULT: RED — %s:%d: parse error: %s\n", path.c_str(), line_no, e.what());
                return 1;
            }
        }
    }
    printf("  validated %d records across %zu files\n", n_records, files.size());
    printf("  probe types observed:");
    for (const auto & p : probe_types) printf(" %s", p.c_str());
    printf("\n");

    if (n_records == 0) {
        printf("RESULT: RED — zero records emitted\n");
        return 1;
    }
    printf("RESULT: PASS\n");
    return 0;
}
