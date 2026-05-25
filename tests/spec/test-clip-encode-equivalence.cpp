// test-clip-encode-equivalence.cpp
//
// PHASE 46 §3 acceptance #4 — vision-encoder output byte-identity check.
//
// The full byte-identity assertion must be done against a live two-GPU
// host with a real CLIP encode (the encoder allocates ~10 GiB per device
// for the 1024-token graph). This test is a thin wrapper that consumes
// a JSON record produced by scripts/verify-multigpu-clip.sh and asserts
// the recorded fields meet the §3 acceptance bar.
//
// Input file (default /tmp/phase46-multigpu-clip/equivalence.json):
//   { "baseline_sha256":  "<hex>",
//     "multigpu_sha256":  "<hex>",
//     "image":            "examples/mtmd/test-1.jpeg",
//     "image_tokens":     1024 }
//
// Acceptance: baseline_sha256 == multigpu_sha256.
//
// Exit codes:
//   0  PASS
//   1  FAIL (sha256 mismatch — Phase 46 acceptance #4 breached)
//   77 SKIP (results file missing — harness has not run)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace {

bool file_exists(const std::string & path) {
    std::ifstream f(path);
    return f.good();
}

std::string slurp(const std::string & path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Extract "key": "value" from a flat JSON blob. Minimal — no escapes,
// matches the format the harness emits.
std::string extract_string(const std::string & json, const std::string & key) {
    std::string needle = "\"" + key + "\"";
    size_t k = json.find(needle);
    if (k == std::string::npos) return {};
    size_t colon = json.find(':', k);
    if (colon == std::string::npos) return {};
    size_t q1 = json.find('"', colon + 1);
    if (q1 == std::string::npos) return {};
    size_t q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos) return {};
    return json.substr(q1 + 1, q2 - q1 - 1);
}

} // namespace

int main(int argc, char ** argv) {
    const char * default_path = "/tmp/phase46-multigpu-clip/equivalence.json";
    std::string path = (argc > 1) ? argv[1] : default_path;
    const char * env = std::getenv("PHASE46_EQUIVALENCE_JSON");
    if (env && *env) path = env;

    if (!file_exists(path)) {
        printf("SKIP test-clip-encode-equivalence: results file missing: %s\n", path.c_str());
        printf("  Run scripts/verify-multigpu-clip.sh during maintenance window first.\n");
        return 77;
    }

    std::string j = slurp(path);
    std::string a = extract_string(j, "baseline_sha256");
    std::string b = extract_string(j, "multigpu_sha256");
    if (a.empty() || b.empty()) {
        fprintf(stderr, "FAIL: results file missing baseline_sha256 or multigpu_sha256\n");
        return 1;
    }
    if (a != b) {
        fprintf(stderr, "FAIL test-clip-encode-equivalence:\n");
        fprintf(stderr, "  baseline_sha256: %s\n", a.c_str());
        fprintf(stderr, "  multigpu_sha256: %s\n", b.c_str());
        fprintf(stderr, "  Phase 46 §3 acceptance #4 BREACHED — byte-identity lost.\n");
        return 1;
    }
    printf("PASS test-clip-encode-equivalence: sha256 %s\n", a.c_str());
    return 0;
}
