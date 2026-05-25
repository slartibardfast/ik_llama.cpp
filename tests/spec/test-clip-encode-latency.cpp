// test-clip-encode-latency.cpp
//
// PHASE 46 B.7 perf gate (HARD, §11.5):
//   median encode latency (multi-GPU CLIP, 1024 tokens) must be
//   ≤ 1.3× the §11.1 single-GPU baseline.
//
// p95 ≤ 1.5× baseline is a SOFT check (warning, not failure).
//
// Like test-clip-encode-equivalence, this is a thin consumer of a JSON
// record produced by scripts/verify-multigpu-clip.sh — the actual
// encode timings come from a real two-GPU host under maintenance window.
//
// Input file (default /tmp/phase46-multigpu-clip/latency.json):
//   { "baseline_ms": 1234,
//     "median_ms":   1499,
//     "p95_ms":      1602,
//     "n_samples":   8 }
//
// (Per §11.5: N=10 encodes, first 2 discarded as warm-up.)
//
// Acceptance:
//   median_ms <= 1.3 * baseline_ms (HARD)
//   p95_ms    <= 1.5 * baseline_ms (SOFT; logs warning)
//
// Exit codes:
//   0  PASS
//   1  FAIL (perf gate breached)
//   77 SKIP (results file missing)

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace {

bool file_exists(const std::string & p) {
    std::ifstream f(p);
    return f.good();
}

std::string slurp(const std::string & p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Extract a numeric value for "key": <number>. Returns -1 on miss.
double extract_number(const std::string & j, const std::string & key) {
    std::string needle = "\"" + key + "\"";
    size_t k = j.find(needle);
    if (k == std::string::npos) return -1.0;
    size_t colon = j.find(':', k);
    if (colon == std::string::npos) return -1.0;
    // Skip whitespace.
    size_t i = colon + 1;
    while (i < j.size() && (j[i] == ' ' || j[i] == '\t')) ++i;
    char * end = nullptr;
    double v = std::strtod(j.c_str() + i, &end);
    if (end == j.c_str() + i) return -1.0;
    return v;
}

} // namespace

int main(int argc, char ** argv) {
    const char * default_path = "/tmp/phase46-multigpu-clip/latency.json";
    std::string path = (argc > 1) ? argv[1] : default_path;
    const char * env = std::getenv("PHASE46_LATENCY_JSON");
    if (env && *env) path = env;

    if (!file_exists(path)) {
        printf("SKIP test-clip-encode-latency: results file missing: %s\n", path.c_str());
        printf("  Capture §11.1 baseline + run scripts/verify-multigpu-clip.sh first.\n");
        return 77;
    }

    std::string j = slurp(path);
    double baseline = extract_number(j, "baseline_ms");
    double median   = extract_number(j, "median_ms");
    double p95      = extract_number(j, "p95_ms");

    if (baseline <= 0.0 || median <= 0.0) {
        fprintf(stderr, "FAIL: results file missing baseline_ms or median_ms\n");
        return 1;
    }

    double hard_ceiling = 1.3 * baseline;
    double soft_ceiling = 1.5 * baseline;

    printf("test-clip-encode-latency:\n");
    printf("  baseline_ms     : %.1f (§11.1 single-GPU reference)\n", baseline);
    printf("  median_ms       : %.1f\n", median);
    printf("  hard ceiling    : %.1f  (1.3× baseline)\n", hard_ceiling);
    if (p95 > 0.0) {
        printf("  p95_ms          : %.1f\n", p95);
        printf("  soft ceiling    : %.1f  (1.5× baseline, warn-only)\n", soft_ceiling);
    }

    if (median > hard_ceiling) {
        fprintf(stderr, "FAIL: median %.1f ms > 1.3× baseline (%.1f ms)\n",
                median, hard_ceiling);
        fprintf(stderr, "  Phase 46 B.7 perf gate BREACHED. Phase stays OPEN.\n");
        fprintf(stderr, "  See PHASE46 §11.5 for diagnostic follow-up paths.\n");
        return 1;
    }

    if (p95 > 0.0 && p95 > soft_ceiling) {
        fprintf(stderr, "WARN: p95 %.1f ms > 1.5× baseline (%.1f ms) — soft gate\n",
                p95, soft_ceiling);
    }

    double headroom_pct = 100.0 * (hard_ceiling - median) / hard_ceiling;
    printf("PASS test-clip-encode-latency (headroom %.1f%% under hard ceiling)\n",
           headroom_pct);
    return 0;
}
