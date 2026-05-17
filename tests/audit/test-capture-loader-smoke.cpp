// test-capture-loader-smoke: validates the F.2 capture-loader against the
// F.1 smoke-test capture. Used as a sanity gate before any A.x audit test
// is built on top.
//
// Reads CAPTURE_DIR/manifest.json (default ../data/audit-f1-smoke) and:
//   1. confirms every record's bin file exists and is the right size
//   2. confirms byte_identity(a, a) returns identical for one tensor
//   3. confirms byte_identity(a, b) where a != b reports the diff

#include "capture_loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char ** argv) {
    std::string manifest = argc >= 2
        ? argv[1]
        : "/home/llm/yarn-agentic/data/audit-f1-smoke/manifest.json";

    fprintf(stderr, "[audit-F.2] loading %s\n", manifest.c_str());
    audit::Capture cap(manifest);
    fprintf(stderr, "[audit-F.2] manifest has %zu records\n", cap.records().size());
    if (cap.records().empty()) {
        fprintf(stderr, "[audit-F.2] FAIL: empty manifest\n");
        return 1;
    }

    // Load every record — fails-fast if any file is missing/corrupt.
    int loaded = 0;
    for (auto & rec : cap.records()) {
        audit::LoadedTensor t = cap.load(rec.name, rec.ubatch_idx);
        if ((int64_t) t.data.size() != t.nelems()) {
            fprintf(stderr, "[audit-F.2] FAIL %s ub%d size mismatch\n",
                    rec.name.c_str(), rec.ubatch_idx);
            return 1;
        }
        loaded++;
    }
    fprintf(stderr, "[audit-F.2] loaded %d/%zu tensors OK\n", loaded, cap.records().size());

    // Byte-identity self-test: a == a is identical.
    audit::LoadedTensor first = cap.load(cap.records()[0].name, cap.records()[0].ubatch_idx);
    auto self = audit::byte_identity(first.data, first.data);
    audit::print_byte_identity(stderr, "self-cmp", self);
    if (!self.identical) {
        fprintf(stderr, "[audit-F.2] FAIL: self-comparison reported diff\n");
        return 1;
    }

    // Negative test: perturb one float, confirm diff is reported.
    std::vector<float> perturbed = first.data;
    if (!perturbed.empty()) perturbed[0] += 1e-3f;
    auto neg = audit::byte_identity(first.data, perturbed);
    audit::print_byte_identity(stderr, "neg-cmp", neg);
    if (neg.identical) {
        fprintf(stderr, "[audit-F.2] FAIL: negative case reported identical\n");
        return 1;
    }
    if (neg.first_idx != 0) {
        fprintf(stderr, "[audit-F.2] FAIL: negative case first_idx=%lld expected 0\n",
                (long long) neg.first_idx);
        return 1;
    }

    fprintf(stderr, "[audit-F.2] all checks PASS\n");
    return 0;
}
