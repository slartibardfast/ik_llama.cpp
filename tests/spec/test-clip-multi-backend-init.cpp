// test-clip-multi-backend-init.cpp
//
// PHASE 46 B.5 spec test — stub-style property test of the multi-backend
// init parser in examples/mtmd/clip.cpp:493+.
//
// What it tests:
//   1. Comma-separated MTMD_BACKEND_DEVICE / --mmproj-devices values are
//      split into a list of device tokens with whitespace trimmed.
//   2. Empty / single-device inputs degrade gracefully (single backend).
//   3. The peer-access gate fails closed: if any (i,j) pair lacks
//      cudaDeviceCanAccessPeer, the init refuses to proceed (returns
//      a failure marker instead of silently using one backend).
//   4. Duplicate device strings collapse to a single entry (sane input
//      normalization).
//
// This is a SPEC test: it mirrors the parsing/gating algorithm in clip.cpp
// against a stub that replicates the logic exactly. If the production
// parser diverges from this stub, the spec test will not catch it
// directly — the live binding is verify-multigpu-clip.sh §11.3 against
// a real two-GPU host.
//
// Returns: 0 = PASS, 1 = FAIL.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <set>

namespace {

// Mirror of the comma-split + trim logic in clip.cpp:493-540.
std::vector<std::string> parse_devices(const std::string & csv) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : csv) {
        if (c == ',') {
            // Trim leading/trailing whitespace.
            size_t a = cur.find_first_not_of(" \t");
            size_t b = cur.find_last_not_of(" \t");
            if (a != std::string::npos) {
                out.push_back(cur.substr(a, b - a + 1));
            }
            cur.clear();
        } else {
            cur += c;
        }
    }
    size_t a = cur.find_first_not_of(" \t");
    size_t b = cur.find_last_not_of(" \t");
    if (a != std::string::npos) {
        out.push_back(cur.substr(a, b - a + 1));
    }
    return out;
}

// Mirror of the duplicate-collapsing pass: order-preserving dedup.
std::vector<std::string> dedup(const std::vector<std::string> & in) {
    std::vector<std::string> out;
    std::set<std::string> seen;
    for (const auto & s : in) {
        if (seen.insert(s).second) out.push_back(s);
    }
    return out;
}

// Mirror of the P2 peer-access gate. Given a NxN bool matrix of
// cudaDeviceCanAccessPeer results, return true iff *every* off-diagonal
// (i,j) with i != j is reachable.
bool peer_access_complete(const std::vector<std::vector<bool>> & m) {
    size_t n = m.size();
    for (size_t i = 0; i < n; ++i) {
        if (m[i].size() != n) return false;
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            if (!m[i][j]) return false;
        }
    }
    return true;
}

int test_basic_csv() {
    auto v = parse_devices("CUDA0,CUDA1");
    if (v.size() != 2 || v[0] != "CUDA0" || v[1] != "CUDA1") {
        fprintf(stderr, "FAIL test_basic_csv: got [%zu] '%s' / '%s'\n",
                v.size(), v.empty() ? "" : v[0].c_str(),
                v.size() < 2 ? "" : v[1].c_str());
        return 1;
    }
    return 0;
}

int test_whitespace_trim() {
    auto v = parse_devices(" CUDA0 , CUDA1 ");
    if (v.size() != 2 || v[0] != "CUDA0" || v[1] != "CUDA1") {
        fprintf(stderr, "FAIL test_whitespace_trim\n");
        return 1;
    }
    return 0;
}

int test_single_device() {
    auto v = parse_devices("CUDA0");
    if (v.size() != 1 || v[0] != "CUDA0") {
        fprintf(stderr, "FAIL test_single_device\n");
        return 1;
    }
    return 0;
}

int test_empty_input() {
    auto v = parse_devices("");
    if (!v.empty()) {
        fprintf(stderr, "FAIL test_empty_input: got %zu entries\n", v.size());
        return 1;
    }
    return 0;
}

int test_trailing_comma() {
    auto v = parse_devices("CUDA0,CUDA1,");
    if (v.size() != 2) {
        fprintf(stderr, "FAIL test_trailing_comma: got %zu entries\n", v.size());
        return 1;
    }
    return 0;
}

int test_dedup() {
    auto v = dedup({"CUDA0", "CUDA1", "CUDA0"});
    if (v.size() != 2 || v[0] != "CUDA0" || v[1] != "CUDA1") {
        fprintf(stderr, "FAIL test_dedup\n");
        return 1;
    }
    return 0;
}

int test_peer_access_all_reachable() {
    // 2 GPUs, both can peer-access — gate passes.
    std::vector<std::vector<bool>> m = {{true, true}, {true, true}};
    if (!peer_access_complete(m)) {
        fprintf(stderr, "FAIL test_peer_access_all_reachable\n");
        return 1;
    }
    return 0;
}

int test_peer_access_fail_closed() {
    // 2 GPUs, (0,1) unreachable — gate must refuse.
    std::vector<std::vector<bool>> m = {{true, false}, {true, true}};
    if (peer_access_complete(m)) {
        fprintf(stderr, "FAIL test_peer_access_fail_closed: gate accepted bad matrix\n");
        return 1;
    }
    return 0;
}

int test_peer_access_three_device() {
    // 3 GPUs, all pairs reachable.
    std::vector<std::vector<bool>> m = {
        {true,  true,  true},
        {true,  true,  true},
        {true,  true,  true}
    };
    if (!peer_access_complete(m)) {
        fprintf(stderr, "FAIL test_peer_access_three_device\n");
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    int rc = 0;
    rc |= test_basic_csv();
    rc |= test_whitespace_trim();
    rc |= test_single_device();
    rc |= test_empty_input();
    rc |= test_trailing_comma();
    rc |= test_dedup();
    rc |= test_peer_access_all_reachable();
    rc |= test_peer_access_fail_closed();
    rc |= test_peer_access_three_device();
    if (rc == 0) {
        printf("PASS test-clip-multi-backend-init (9 cases)\n");
    } else {
        printf("FAIL test-clip-multi-backend-init\n");
    }
    return rc;
}
