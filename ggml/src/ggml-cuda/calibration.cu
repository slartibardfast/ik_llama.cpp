// ggml-cuda/calibration.cu
//
// PHASE_CUDA_NATIVE_DISPATCH commit C0 — Calibrated dispatch framework.
// See calibration.h for the public API and design rationale.

#include "ggml-cuda-calibration.h"
#include "ggml-cuda/common.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cinttypes>
#include <ctime>
#include <algorithm>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#define gmtime_r(t, tm) gmtime_s((tm), (t))
#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#endif
#else
#include <sys/types.h>
#include <unistd.h>
#include <pwd.h>
#endif
#include <fstream>
#include <sstream>

// =============================================================================
// §1 — Bucket constants
// =============================================================================

const size_t ggml_cal_buckets[6] = {
    0,
    1ULL  * 1024 * 1024,           //   1 MB
    10ULL * 1024 * 1024,           //  10 MB
    100ULL * 1024 * 1024,          // 100 MB
    1024ULL * 1024 * 1024,         //   1 GB
    SIZE_MAX,
};
const int ggml_cal_n_buckets = 6;

// Bump when the calibration code itself changes in a way that
// invalidates previously cached thresholds (e.g., probe algo change,
// bucket-set change, equivalence-criterion change).
#define GGML_CUDA_CALIBRATION_SCHEMA_VERSION 1

// =============================================================================
// §2 — Global probe registry
//
// Probe functions live in the .cu files that implement each op. They
// register themselves via ggml_cuda_calibration_register_op() typically
// from a static initializer. The framework keeps a global table indexed
// by op_id.
// =============================================================================

namespace {

struct ggml_cal_registered_op {
    const char *       name = nullptr;
    ggml_cuda_probe_fn probe = nullptr;
};

static ggml_cal_registered_op g_registry[GGML_CAL_OP_COUNT_] = {};

const char * g_default_op_names[GGML_CAL_OP_COUNT_] = {
    "REDUCE_CROSS_DEVICE",
    "MATMUL_STREAM_SPLIT",
    "PEER_COPY",
    "GRAPH_CAPTURE",
};

}  // namespace

extern "C" const char * ggml_cal_op_name(ggml_cuda_calibrated_op op) {
    const int i = (int) op;
    if (i < 0 || i >= GGML_CAL_OP_COUNT_) return "<invalid>";
    if (g_registry[i].name) return g_registry[i].name;
    return g_default_op_names[i];
}

extern "C" void ggml_cuda_calibration_register_op(
        ggml_cuda_calibrated_op op,
        const char *            name,
        ggml_cuda_probe_fn      probe) {
    const int i = (int) op;
    if (i < 0 || i >= GGML_CAL_OP_COUNT_) {
        fprintf(stderr, "ggml_cuda_calibration_register_op: invalid op id %d\n", i);
        return;
    }
    g_registry[i].name  = name;
    g_registry[i].probe = probe;
}

extern "C" void ggml_cuda_calibration_reset_registry_for_tests(void) {
    for (int i = 0; i < GGML_CAL_OP_COUNT_; ++i) {
        g_registry[i].name  = nullptr;
        g_registry[i].probe = nullptr;
    }
}

// =============================================================================
// §3 — Cache key derivation
//
// Key inputs: schema_version, cuda_runtime_version, gpu_uuids[].
// We deliberately omit ggml_commit — calibration depends on CUDA driver
// + hardware, not on ggml's own code. Bump GGML_CUDA_CALIBRATION_SCHEMA_
// VERSION manually when calibration code changes invalidate prior caches.
//
// The filename uses an FNV-1a 64-bit hash of the key string for a
// stable, length-bounded filename. The key string itself is also stored
// in the JSON so cache hits can be verified.
// =============================================================================

namespace {

uint64_t fnv1a64(const char * s, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint8_t) s[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

uint64_t fnv1a64_string(const std::string & s) {
    return fnv1a64(s.data(), s.size());
}

std::string ggml_cal_derive_cache_key(int n_devices) {
    std::ostringstream ss;
    ss << "v" << GGML_CUDA_CALIBRATION_SCHEMA_VERSION;

    int cuda_runtime = 0;
    cudaRuntimeGetVersion(&cuda_runtime);
    ss << "-cuda" << cuda_runtime;

    int driver_version = 0;
    cudaDriverGetVersion(&driver_version);
    ss << "-drv" << driver_version;

    ss << "-gpus";
    for (int d = 0; d < n_devices; ++d) {
        cudaDeviceProp prop = {};
        if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) {
            ss << ":err";
            continue;
        }
        ss << ":";
        for (int i = 0; i < 16; ++i) {
            char hex[3];
            snprintf(hex, sizeof(hex), "%02x", (unsigned char) prop.uuid.bytes[i]);
            ss << hex;
        }
    }
    return ss.str();
}

}  // namespace

// =============================================================================
// §4 — Cache path resolution
//
// Order: $XDG_CACHE_HOME/ggml → $HOME/.cache/ggml → /tmp/ggml-cache.
// If none is writable, in-memory mode.
// =============================================================================

namespace {

bool ggml_cal_mkdir_p(const std::string & path) {
    if (path.empty()) return false;
    // Try create. If it exists, success.
    if (mkdir(path.c_str(), 0700) == 0) return true;
    if (errno == EEXIST) {
        struct stat st;
        return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
    }
    // Parent might not exist — try mkdir on parent first.
    size_t slash = path.find_last_of('/');
    if (slash == std::string::npos || slash == 0) return false;
    if (!ggml_cal_mkdir_p(path.substr(0, slash))) return false;
    return mkdir(path.c_str(), 0700) == 0 || errno == EEXIST;
}

std::string ggml_cal_resolve_cache_dir() {
    const char * xdg = std::getenv("XDG_CACHE_HOME");
    if (xdg && *xdg) {
        std::string p = std::string(xdg) + "/ggml";
        if (ggml_cal_mkdir_p(p)) return p;
    }
    const char * home = std::getenv("HOME");
#ifndef _WIN32
    if (!home || !*home) {
        struct passwd * pw = getpwuid(getuid());
        if (pw && pw->pw_dir) home = pw->pw_dir;
    }
#else
    if (!home || !*home) {
        home = std::getenv("USERPROFILE");
    }
#endif
    if (home && *home) {
        std::string p = std::string(home) + "/.cache/ggml";
        if (ggml_cal_mkdir_p(p)) return p;
    }
    // Fallback /tmp.
    if (ggml_cal_mkdir_p("/tmp/ggml-cache")) {
        return "/tmp/ggml-cache";
    }
    return "";  // signals in-memory mode
}

std::string ggml_cal_cache_path(const std::string & cache_key) {
    std::string dir = ggml_cal_resolve_cache_dir();
    if (dir.empty()) return "";
    char filename[64];
    snprintf(filename, sizeof(filename),
             "cuda-calibration-%016" PRIx64 ".json",
             fnv1a64_string(cache_key));
    return dir + "/" + filename;
}

}  // namespace

// =============================================================================
// §5 — Minimal JSON parser / writer
//
// Schema is fixed; the parser only handles what we write. It is
// fault-tolerant: any parse error returns false from the loader, which
// triggers re-calibration. The writer always emits valid JSON for our
// schema.
// =============================================================================

namespace {

// Write the threshold table to JSON.
bool ggml_cal_write_json(
        const std::string &                   path,
        const std::string &                   cache_key,
        const ggml_cuda_calibration_table &   table,
        int                                   n_devices) {
    std::ofstream f(path, std::ios::trunc);
    if (!f.good()) return false;

    f << "{\n";
    f << "  \"schema_version\": " << GGML_CUDA_CALIBRATION_SCHEMA_VERSION << ",\n";
    f << "  \"cache_key\": \"" << cache_key << "\",\n";

    char ts[32] = {};
    time_t now = time(nullptr);
    struct tm tm_utc;
    gmtime_r(&now, &tm_utc);
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    f << "  \"calibrated_at\": \"" << ts << "\",\n";

    f << "  \"host_info\": {\n";
    f << "    \"n_gpus\": " << n_devices << ",\n";
    f << "    \"gpu_models\": [";
    for (int d = 0; d < n_devices; ++d) {
        cudaDeviceProp prop = {};
        if (cudaGetDeviceProperties(&prop, d) == cudaSuccess) {
            if (d > 0) f << ", ";
            f << "\"" << prop.name << "\"";
        }
    }
    f << "]\n";
    f << "  },\n";

    f << "  \"thresholds\": {\n";
    for (int op = 0; op < GGML_CAL_OP_COUNT_; ++op) {
        f << "    \"" << ggml_cal_op_name((ggml_cuda_calibrated_op) op) << "\": ";
        const size_t v = table.thresholds[op];
        if (v == SIZE_MAX) {
            f << "-1";
        } else {
            f << v;
        }
        if (op < GGML_CAL_OP_COUNT_ - 1) f << ",";
        f << "\n";
    }
    f << "  }\n";
    f << "}\n";
    return f.good();
}

// Whitespace skipping helper for the parser.
void skip_ws(const std::string & s, size_t & i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
}

// Returns false on parse error.
bool match_lit(const std::string & s, size_t & i, char c) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != c) return false;
    ++i;
    return true;
}

// Read a JSON string literal into out. Assumes opening quote not yet
// consumed.
bool parse_string(const std::string & s, size_t & i, std::string & out) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '"') return false;
    ++i;
    out.clear();
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\' && i + 1 < s.size()) {
            // Minimal escape handling: just copy through.
            out.push_back(s[i + 1]);
            i += 2;
        } else {
            out.push_back(s[i]);
            ++i;
        }
    }
    if (i >= s.size()) return false;
    ++i;  // consume closing quote
    return true;
}

// Read a JSON integer (possibly negative). Returns false on parse error.
bool parse_int64(const std::string & s, size_t & i, int64_t & out) {
    skip_ws(s, i);
    size_t start = i;
    if (i < s.size() && s[i] == '-') ++i;
    if (i >= s.size() || !isdigit((unsigned char) s[i])) return false;
    while (i < s.size() && isdigit((unsigned char) s[i])) ++i;
    out = (int64_t) std::strtoll(s.substr(start, i - start).c_str(), nullptr, 10);
    return true;
}

// Skip over an arbitrary JSON value (object, array, string, number,
// or literal). Robust against the field types we emit. Returns false
// only on outright malformed input.
bool skip_value(const std::string & s, size_t & i) {
    skip_ws(s, i);
    if (i >= s.size()) return false;
    if (s[i] == '"') {
        std::string tmp;
        return parse_string(s, i, tmp);
    }
    if (s[i] == '{' || s[i] == '[') {
        char open  = s[i];
        char close = (open == '{') ? '}' : ']';
        int  depth = 1;
        ++i;
        while (i < s.size() && depth > 0) {
            if (s[i] == '"') {
                std::string tmp;
                if (!parse_string(s, i, tmp)) return false;
                continue;
            }
            if (s[i] == open)  ++depth;
            if (s[i] == close) --depth;
            ++i;
        }
        return depth == 0;
    }
    // number / true / false / null
    while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']' &&
           s[i] != ' ' && s[i] != '\n' && s[i] != '\t' && s[i] != '\r') {
        ++i;
    }
    return true;
}

// Load the threshold table from JSON. Returns false on parse error or
// cache_key mismatch.
bool ggml_cal_read_json(
        const std::string &              path,
        const std::string &              expected_key,
        ggml_cuda_calibration_table &    out) {
    std::ifstream f(path);
    if (!f.good()) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    if (s.empty()) return false;

    size_t i = 0;
    if (!match_lit(s, i, '{')) return false;

    int         got_schema_version = -1;
    std::string got_cache_key;
    bool        thresholds_seen = false;
    size_t      raw_thresholds[GGML_CAL_OP_COUNT_];
    bool        threshold_seen[GGML_CAL_OP_COUNT_] = {};
    for (int j = 0; j < GGML_CAL_OP_COUNT_; ++j) raw_thresholds[j] = SIZE_MAX;

    while (true) {
        skip_ws(s, i);
        if (i >= s.size()) return false;
        if (s[i] == '}') { ++i; break; }

        std::string key;
        if (!parse_string(s, i, key)) return false;
        if (!match_lit(s, i, ':')) return false;
        skip_ws(s, i);

        if (key == "schema_version") {
            int64_t v;
            if (!parse_int64(s, i, v)) return false;
            got_schema_version = (int) v;
        } else if (key == "cache_key") {
            if (!parse_string(s, i, got_cache_key)) return false;
        } else if (key == "thresholds") {
            if (!match_lit(s, i, '{')) return false;
            while (true) {
                skip_ws(s, i);
                if (i >= s.size()) return false;
                if (s[i] == '}') { ++i; break; }
                std::string op_name;
                if (!parse_string(s, i, op_name)) return false;
                if (!match_lit(s, i, ':')) return false;
                int64_t v;
                if (!parse_int64(s, i, v)) return false;
                for (int op = 0; op < GGML_CAL_OP_COUNT_; ++op) {
                    if (op_name == g_default_op_names[op]) {
                        raw_thresholds[op] = (v < 0) ? SIZE_MAX : (size_t) v;
                        threshold_seen[op] = true;
                        break;
                    }
                }
                skip_ws(s, i);
                if (i < s.size() && s[i] == ',') ++i;
            }
            thresholds_seen = true;
        } else {
            // Skip unknown field (forward-compat).
            if (!skip_value(s, i)) return false;
        }

        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') ++i;
    }

    if (got_schema_version != GGML_CUDA_CALIBRATION_SCHEMA_VERSION) return false;
    if (got_cache_key != expected_key) return false;
    if (!thresholds_seen) return false;

    for (int op = 0; op < GGML_CAL_OP_COUNT_; ++op) {
        out.thresholds[op] = threshold_seen[op] ? raw_thresholds[op] : SIZE_MAX;
    }
    return true;
}

}  // namespace

// =============================================================================
// §6 — Quantization
// =============================================================================

namespace {

// Returns the smallest bucket >= raw_threshold, or SIZE_MAX. Sentinel
// indices (0 and N-1) are skipped during bucket selection — calibration
// uses indices 1..N-2 as real thresholds, with SIZE_MAX as the "no
// crossover found" sentinel.
size_t ggml_cal_quantize_to_bucket(size_t raw_threshold) {
    if (raw_threshold == SIZE_MAX) return SIZE_MAX;
    for (int b = 1; b < ggml_cal_n_buckets - 1; ++b) {
        if (raw_threshold <= ggml_cal_buckets[b]) return ggml_cal_buckets[b];
    }
    return SIZE_MAX;
}

bool ggml_cal_has_vram_for_probe(int device, size_t payload_bytes) {
    // We need roughly 6× payload (default-path scratch + alt-path
    // scratch + small overhead) per device for the probe to fit
    // comfortably. Be generous with headroom — leave 1 GiB free.
    size_t free_b = 0, total_b = 0;
    cudaSetDevice(device);
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return false;
    const size_t need = 6 * payload_bytes + 1024ULL * 1024 * 1024;
    return free_b >= need;
}

}  // namespace

// =============================================================================
// §7 — Calibration entry point
// =============================================================================

extern "C" void ggml_cuda_calibrate(ggml_backend_cuda_context * ctx) {
    if (!ctx) return;
    if (ctx->calibration_table.calibrated) return;

    // Default to SIZE_MAX for every op (no alt-strategy use).
    for (int op = 0; op < GGML_CAL_OP_COUNT_; ++op) {
        ctx->calibration_table.thresholds[op] = SIZE_MAX;
    }
    ctx->calibration_table.loaded_from_cache = false;

    const bool force_recal = std::getenv("GGML_CALIBRATION_FORCE_RECALIBRATE") != nullptr;
    const bool disable_io  = std::getenv("GGML_CALIBRATION_DISABLE") != nullptr;

    int n_devices = 1;
    cudaGetDeviceCount(&n_devices);

    ctx->calibration_table.cache_key = ggml_cal_derive_cache_key(n_devices);
    const std::string cache_path = ggml_cal_cache_path(ctx->calibration_table.cache_key);

    // Try cache hit.
    if (!force_recal && !disable_io && !cache_path.empty()) {
        if (ggml_cal_read_json(cache_path, ctx->calibration_table.cache_key,
                               ctx->calibration_table)) {
            ctx->calibration_table.loaded_from_cache = true;
            ctx->calibration_table.calibrated        = true;
            fprintf(stderr,
                    "ggml_cuda_calibrate: cache hit %s\n", cache_path.c_str());
            return;
        }
        fprintf(stderr,
                "ggml_cuda_calibrate: cache miss; probing %s\n", cache_path.c_str());
    }

    // Cache miss → probe each registered op.
    for (int op = 0; op < GGML_CAL_OP_COUNT_; ++op) {
        // Per-op env override takes precedence over probing.
        char env_name[128];
        snprintf(env_name, sizeof(env_name),
                 "GGML_CAL_%s_THRESHOLD_BYTES",
                 g_default_op_names[op]);
        if (const char * v = std::getenv(env_name)) {
            const int64_t parsed = std::strtoll(v, nullptr, 10);
            ctx->calibration_table.thresholds[op] =
                (parsed < 0) ? SIZE_MAX : (size_t) parsed;
            fprintf(stderr,
                    "ggml_cuda_calibrate: %s = %s (env override)\n",
                    g_default_op_names[op], v);
            continue;
        }

        if (!g_registry[op].probe) {
            // Not registered yet; will get SIZE_MAX from default init.
            continue;
        }

        const int N_ITERS = 10;
        size_t threshold = SIZE_MAX;
        for (int b = 1; b < ggml_cal_n_buckets - 1; ++b) {
            const size_t sz = ggml_cal_buckets[b];
            if (!ggml_cal_has_vram_for_probe(ctx->device, sz)) {
                fprintf(stderr,
                        "ggml_cuda_calibrate: %s skipping %zu MB probe (insufficient VRAM)\n",
                        g_default_op_names[op], sz / (1024 * 1024));
                break;
            }

            ggml_cuda_probe_result def = g_registry[op].probe(ctx, false, sz, N_ITERS);
            ggml_cuda_probe_result alt = g_registry[op].probe(ctx, true,  sz, N_ITERS);

            // Conservative crossover criterion: alt.p95 < default.p50.
            if (alt.p95_ms < def.p50_ms) {
                threshold = sz;
                break;
            }
        }
        ctx->calibration_table.thresholds[op] = threshold;
        fprintf(stderr,
                "ggml_cuda_calibrate: %s threshold = %s\n",
                g_default_op_names[op],
                (threshold == SIZE_MAX) ? "SIZE_MAX"
                                        : std::to_string(threshold).c_str());
    }

    // Save cache (best-effort).
    if (!disable_io && !cache_path.empty()) {
        if (ggml_cal_write_json(cache_path, ctx->calibration_table.cache_key,
                                ctx->calibration_table, n_devices)) {
            fprintf(stderr,
                    "ggml_cuda_calibrate: cache written to %s\n",
                    cache_path.c_str());
        } else {
            fprintf(stderr,
                    "ggml_cuda_calibrate: cache write failed at %s (continuing in-memory)\n",
                    cache_path.c_str());
        }
    }

    ctx->calibration_table.calibrated = true;
}

// =============================================================================
// §8 — Threshold lookup (hot path)
// =============================================================================

extern "C" size_t ggml_cuda_threshold_for(
        ggml_backend_cuda_context *  ctx,
        ggml_cuda_calibrated_op       op) {
    if (!ctx) return SIZE_MAX;
    const int i = (int) op;
    if (i < 0 || i >= GGML_CAL_OP_COUNT_) return SIZE_MAX;
    return ctx->calibration_table.thresholds[i];
}

// =============================================================================
// §9 — Public test helpers
// =============================================================================

#include "ggml-backend-impl.h"

extern "C" bool ggml_cuda_calibration_was_loaded_from_cache(struct ggml_backend * backend) {
    if (!backend || !backend->context) return false;
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    return ctx->calibration_table.loaded_from_cache;
}

extern "C" size_t ggml_cuda_calibration_threshold_for_backend(
        struct ggml_backend *        backend,
        ggml_cuda_calibrated_op       op) {
    if (!backend || !backend->context) return SIZE_MAX;
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    return ggml_cuda_threshold_for(ctx, op);
}
