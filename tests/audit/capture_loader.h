// capture_loader.h — load production-state tensor captures produced by
// llama-state-capture (audit F.1). Reusable across all audit A.x tests
// per yarn-agentic/PLAN_DETERMINISM_AUDIT.md §F.2.
//
// Usage:
//   audit::Capture cap("data/audit-NNN/manifest.json");
//   auto q = cap.load("q-15", 0);                  // first ubatch
//   auto k = cap.load("k-15", 0);
//   // q.data is std::vector<float>, q.shape is [ne0,ne1,ne2,ne3]
//
// Manifest record fields (JSON):
//   prompt_id, name, prefix, layer, shape [4], orig_dtype, n_seq_max,
//   ubatch_idx, file.

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace audit {

struct CaptureRecord {
    std::string         prompt_id;
    std::string         name;
    std::string         prefix;
    int                 layer = -1;
    std::array<int64_t, 4> shape = {1, 1, 1, 1};
    std::string         orig_dtype;
    int                 n_seq_max = 1;
    int                 ubatch_idx = 0;
    std::string         file;
};

struct LoadedTensor {
    std::vector<float>      data;
    std::array<int64_t, 4>  shape{1, 1, 1, 1};
    std::string             orig_dtype;
    int                     layer = -1;
    int                     ubatch_idx = 0;

    int64_t nelems() const { return shape[0] * shape[1] * shape[2] * shape[3]; }
};

// Minimal JSON parser for the manifest format llama-state-capture emits.
// Not a general JSON parser — relies on the well-known shape of the file.
class Capture {
public:
    explicit Capture(const std::string & manifest_path) {
        std::ifstream f(manifest_path);
        if (!f) throw std::runtime_error("capture: cannot open " + manifest_path);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string s = ss.str();
        parse(s);
        // base_dir = dirname(manifest_path)
        size_t slash = manifest_path.find_last_of('/');
        base_dir_ = (slash == std::string::npos) ? "." : manifest_path.substr(0, slash);
    }

    // Number of records matching (name, ubatch_idx>=0 picks specific, =-1 any)
    size_t count(const std::string & name) const {
        size_t n = 0;
        for (auto & r : recs_) if (r.name == name) ++n;
        return n;
    }

    // Load one tensor by name + ubatch_idx. Throws if not found or file
    // shape/size mismatches.
    LoadedTensor load(const std::string & name, int ubatch_idx = 0) const {
        for (auto & r : recs_) {
            if (r.name == name && r.ubatch_idx == ubatch_idx) {
                return load_record(r);
            }
        }
        throw std::runtime_error("capture: not found: " + name + " ub" + std::to_string(ubatch_idx));
    }

    const std::vector<CaptureRecord> & records() const { return recs_; }

private:
    std::vector<CaptureRecord> recs_;
    std::string                base_dir_;

    LoadedTensor load_record(const CaptureRecord & r) const {
        LoadedTensor t;
        t.shape       = r.shape;
        t.orig_dtype  = r.orig_dtype;
        t.layer       = r.layer;
        t.ubatch_idx  = r.ubatch_idx;

        std::string path = base_dir_ + "/" + r.file;
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("capture: cannot open " + path);
        const size_t bytes = (size_t) f.tellg();
        f.seekg(0);
        const size_t expected = (size_t) t.nelems() * sizeof(float);
        if (bytes != expected) {
            throw std::runtime_error("capture: size mismatch for " + path +
                ": file=" + std::to_string(bytes) + " expected=" + std::to_string(expected));
        }
        t.data.resize(t.nelems());
        f.read((char *) t.data.data(), bytes);
        return t;
    }

    void parse(const std::string & s) {
        size_t i = 0;
        skip_ws(s, i);
        if (i >= s.size() || s[i] != '[') throw std::runtime_error("manifest: expected [");
        ++i;
        while (i < s.size()) {
            skip_ws(s, i);
            if (i >= s.size()) break;
            if (s[i] == ']') break;
            if (s[i] == ',') { ++i; continue; }
            if (s[i] != '{') throw std::runtime_error("manifest: expected {");
            CaptureRecord r;
            parse_record(s, i, r);
            recs_.push_back(std::move(r));
        }
    }

    static void skip_ws(const std::string & s, size_t & i) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i;
    }

    void parse_record(const std::string & s, size_t & i, CaptureRecord & r) {
        if (s[i] != '{') throw std::runtime_error("rec: {");
        ++i;
        while (i < s.size()) {
            skip_ws(s, i);
            if (s[i] == '}') { ++i; return; }
            if (s[i] == ',') { ++i; continue; }
            std::string key;
            parse_string(s, i, key);
            skip_ws(s, i);
            if (s[i] != ':') throw std::runtime_error("rec: :");
            ++i;
            skip_ws(s, i);

            if      (key == "prompt_id")   parse_string(s, i, r.prompt_id);
            else if (key == "name")        parse_string(s, i, r.name);
            else if (key == "prefix")      parse_string(s, i, r.prefix);
            else if (key == "layer")       r.layer       = (int) parse_int(s, i);
            else if (key == "orig_dtype")  parse_string(s, i, r.orig_dtype);
            else if (key == "n_seq_max")   r.n_seq_max   = (int) parse_int(s, i);
            else if (key == "ubatch_idx")  r.ubatch_idx  = (int) parse_int(s, i);
            else if (key == "file")        parse_string(s, i, r.file);
            else if (key == "shape") {
                if (s[i] != '[') throw std::runtime_error("rec: shape [");
                ++i;
                for (int k = 0; k < 4; ++k) {
                    skip_ws(s, i);
                    r.shape[k] = parse_int(s, i);
                    skip_ws(s, i);
                    if (s[i] == ',') ++i;
                }
                skip_ws(s, i);
                if (s[i] != ']') throw std::runtime_error("rec: shape ]");
                ++i;
            } else {
                // Unknown key — skip to next , or }
                int depth = 0;
                while (i < s.size()) {
                    if (s[i] == '"') { std::string dummy; parse_string(s, i, dummy); continue; }
                    if (s[i] == '{' || s[i] == '[') { depth++; ++i; continue; }
                    if (s[i] == '}' || s[i] == ']') { if (!depth) break; depth--; ++i; continue; }
                    if (s[i] == ',' && !depth) break;
                    ++i;
                }
            }
        }
    }

    static void parse_string(const std::string & s, size_t & i, std::string & out) {
        if (s[i] != '"') throw std::runtime_error("str");
        ++i;
        out.clear();
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                out.push_back(s[i + 1]);
                i += 2;
            } else {
                out.push_back(s[i++]);
            }
        }
        if (i < s.size()) ++i;
    }

    static int64_t parse_int(const std::string & s, size_t & i) {
        int sign = 1;
        if (s[i] == '-') { sign = -1; ++i; }
        int64_t v = 0;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
            v = v * 10 + (s[i] - '0');
            ++i;
        }
        return sign * v;
    }
};

// Byte-identity result: divergent fp32 lanes between two tensors of the
// same shape. Always reports max |a-b| over the whole vector, plus the
// first lane that differs in raw bits (uint32 cmp).
struct ByteIdentityResult {
    bool   identical = true;
    size_t total     = 0;
    size_t differ    = 0;
    int64_t first_idx = -1;
    float  first_a   = 0;
    float  first_b   = 0;
    uint32_t first_bits_a = 0;
    uint32_t first_bits_b = 0;
    float  max_abs_diff = 0;
};

inline ByteIdentityResult byte_identity(const std::vector<float> & a, const std::vector<float> & b) {
    ByteIdentityResult r;
    r.total = a.size();
    if (a.size() != b.size()) {
        r.identical = false;
        r.differ = (size_t) -1;
        return r;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t ua, ub;
        std::memcpy(&ua, &a[i], 4);
        std::memcpy(&ub, &b[i], 4);
        if (ua != ub) {
            ++r.differ;
            if (r.first_idx < 0) {
                r.first_idx     = (int64_t) i;
                r.first_a       = a[i];
                r.first_b       = b[i];
                r.first_bits_a  = ua;
                r.first_bits_b  = ub;
            }
            float d = std::fabs(a[i] - b[i]);
            if (d > r.max_abs_diff) r.max_abs_diff = d;
        }
    }
    r.identical = (r.differ == 0);
    return r;
}

inline void print_byte_identity(FILE * out, const std::string & label, const ByteIdentityResult & r) {
    if (r.identical) {
        std::fprintf(out, "[%s] PASS — %zu/%zu floats byte-identical\n", label.c_str(), r.total, r.total);
    } else {
        std::fprintf(out,
            "[%s] FAIL — %zu/%zu differ, first idx=%lld a=%g(%08x) b=%g(%08x), max|Δ|=%.3e\n",
            label.c_str(), r.differ, r.total, (long long) r.first_idx,
            r.first_a, r.first_bits_a, r.first_b, r.first_bits_b, r.max_abs_diff);
    }
}

} // namespace audit
