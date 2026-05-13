// npy-reader.h
//
// Minimal NPY v1.0/v2.0 reader for fp32/fp16/int32/int64 arrays.
// No external deps. Adequate for the DFlash test-side vLLM-dump
// consumers; not a general-purpose library.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace dflash_reference {

struct NpyArray {
    std::vector<int> shape;
    std::string      dtype;     // e.g., "<f4", "<f2", "<i4", "<i8"
    std::vector<unsigned char> data;
};

// Parse the NPY header line. Returns true on success and fills shape / dtype.
// Sets `data_offset` to the file position where raw data starts.
inline bool _parse_npy_header(std::FILE * f, std::string & dtype, std::vector<int> & shape, std::size_t & data_offset) {
    unsigned char magic[6];
    if (std::fread(magic, 1, 6, f) != 6) return false;
    if (magic[0] != 0x93 || magic[1] != 'N' || magic[2] != 'U' || magic[3] != 'M' ||
        magic[4] != 'P' || magic[5] != 'Y') {
        std::fprintf(stderr, "npy: bad magic\n");
        return false;
    }
    unsigned char major = 0, minor = 0;
    if (std::fread(&major, 1, 1, f) != 1) return false;
    if (std::fread(&minor, 1, 1, f) != 1) return false;
    std::size_t hlen = 0;
    if (major == 1) {
        uint16_t h = 0;
        if (std::fread(&h, 2, 1, f) != 1) return false;
        hlen = h;
    } else if (major == 2 || major == 3) {
        uint32_t h = 0;
        if (std::fread(&h, 4, 1, f) != 1) return false;
        hlen = h;
    } else {
        std::fprintf(stderr, "npy: unsupported major version %d\n", (int)major);
        return false;
    }
    std::vector<char> hdr(hlen + 1, '\0');
    if (std::fread(hdr.data(), 1, hlen, f) != hlen) return false;
    const char * h = hdr.data();
    auto find_value = [&](const char * key) -> std::string {
        const char * p = std::strstr(h, key);
        if (!p) return "";
        p += std::strlen(key);
        while (*p && (*p == ' ' || *p == ':')) ++p;
        if (*p == '\'') {
            const char * q = std::strchr(++p, '\'');
            if (!q) return "";
            return std::string(p, q - p);
        }
        if (*p == '(') {
            const char * q = std::strchr(++p, ')');
            if (!q) return "";
            return std::string(p, q - p);
        }
        return "";
    };
    dtype = find_value("'descr'");
    if (dtype.empty()) return false;
    std::string shape_s = find_value("'shape'");
    if (shape_s.empty()) return false;
    shape.clear();
    const char * sp = shape_s.c_str();
    while (*sp) {
        while (*sp && (*sp == ' ' || *sp == ',')) ++sp;
        if (*sp == 0) break;
        char * end = nullptr;
        long v = std::strtol(sp, &end, 10);
        if (end == sp) break;
        shape.push_back(static_cast<int>(v));
        sp = end;
    }
    data_offset = static_cast<std::size_t>(std::ftell(f));
    return true;
}

inline std::size_t _dtype_size(const std::string & dtype) {
    if (dtype.size() < 3) return 0;
    return static_cast<std::size_t>(std::atoi(dtype.c_str() + 2));
}

inline bool load_npy(const char * path, NpyArray & out) {
    std::FILE * f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "npy: cannot open %s\n", path);
        return false;
    }
    std::size_t data_off = 0;
    if (!_parse_npy_header(f, out.dtype, out.shape, data_off)) {
        std::fclose(f);
        return false;
    }
    std::size_t nelem = 1;
    for (int s : out.shape) nelem *= static_cast<std::size_t>(s);
    const std::size_t elem_sz = _dtype_size(out.dtype);
    const std::size_t nbytes = nelem * elem_sz;
    out.data.resize(nbytes);
    if (std::fread(out.data.data(), 1, nbytes, f) != nbytes) {
        std::fclose(f);
        std::fprintf(stderr, "npy: short read on data section\n");
        return false;
    }
    std::fclose(f);
    return true;
}

inline std::vector<float> load_npy_f32(const char * path) {
    NpyArray a;
    if (!load_npy(path, a) || a.dtype != "<f4") {
        return {};
    }
    std::vector<float> out(a.data.size() / sizeof(float));
    std::memcpy(out.data(), a.data.data(), a.data.size());
    return out;
}

inline std::vector<int64_t> load_npy_i64(const char * path) {
    NpyArray a;
    if (!load_npy(path, a) || a.dtype != "<i8") {
        return {};
    }
    std::vector<int64_t> out(a.data.size() / sizeof(int64_t));
    std::memcpy(out.data(), a.data.data(), a.data.size());
    return out;
}

} // namespace dflash_reference
