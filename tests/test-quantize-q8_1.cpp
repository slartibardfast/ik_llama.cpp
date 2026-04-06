// Test Q8_1 quantization format and layout analysis.
// Validates CPU reference Q8_1 quantization and documents the
// block_q8_1 vs block_q8_1_x4 layout incompatibility.
//
// Usage: ./test-quantize-q8_1
#include "ggml.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#pragma pack(push, 1)
struct block_q8_1_plain {
    uint16_t d;      // fp16 scale
    uint16_t s;      // fp16 sum*d
    int8_t qs[32];   // quantized values
};
// 36 bytes per block

struct block_q8_1_x4 {
    uint16_t ds[4][2]; // 4 blocks' [d, s] — 16 bytes
    int32_t qs[32];    // 4 blocks' quants packed as int32 — 128 bytes
};
// 144 bytes = 4 × 36 bytes (same total, different layout)
#pragma pack(pop)

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t expo = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (expo == 0 && mant == 0) { float r; uint32_t z = sign; memcpy(&r, &z, 4); return r; }
    if (expo == 0) { while (!(mant & 0x400)) { mant <<= 1; expo--; } expo++; mant &= 0x3FF; }
    else if (expo == 31) { uint32_t f = sign | 0x7F800000 | (mant << 13); float r; memcpy(&r, &f, 4); return r; }
    uint32_t f = sign | ((expo + 112) << 23) | (mant << 13);
    float r; memcpy(&r, &f, 4); return r;
}

// CPU Q8_1 quantization (matches quantize_q8_1.comp shader logic)
static void quantize_q8_1_cpu(const float* src, block_q8_1_plain* dst, int n_blocks) {
    for (int i = 0; i < n_blocks; i++) {
        float amax = 0;
        for (int j = 0; j < 32; j++) {
            float v = fabsf(src[i * 32 + j]);
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        float id = d != 0.0f ? 1.0f / d : 0.0f;
        float sum = 0;
        for (int j = 0; j < 32; j++) {
            float qf = roundf(src[i * 32 + j] * id);
            int8_t q = (int8_t)qf;
            dst[i].qs[j] = q;
            sum += qf;
        }
        // Store d and sum*d as fp16 (using GGML_FP32_TO_FP16 equivalent)
        float sd = sum * d;
        // Simplified fp32→fp16 (loses some edge cases but fine for testing)
        uint32_t db; memcpy(&db, &d, 4);
        uint32_t sb; memcpy(&sb, &sd, 4);
        auto to_fp16 = [](uint32_t bits) -> uint16_t {
            uint16_t sign = (bits >> 16) & 0x8000;
            int exp = ((bits >> 23) & 0xFF) - 127 + 15;
            uint16_t mant = (bits >> 13) & 0x3FF;
            if (exp <= 0) return sign;
            if (exp >= 31) return sign | 0x7C00;
            return sign | (exp << 10) | mant;
        };
        dst[i].d = to_fp16(db);
        dst[i].s = to_fp16(sb);
    }
}

// Repack plain blocks into x4 format (what quantize_q8_1_x4 shader should produce)
static void repack_to_x4(const block_q8_1_plain* plain, block_q8_1_x4* x4, int n_x4_blocks) {
    for (int g = 0; g < n_x4_blocks; g++) {
        // Pack scales: x4.ds[inner] = plain[g*4+inner].{d,s}
        for (int inner = 0; inner < 4; inner++) {
            x4[g].ds[inner][0] = plain[g * 4 + inner].d;
            x4[g].ds[inner][1] = plain[g * 4 + inner].s;
        }
        // Pack quants: x4.qs[inner*8 + iqs] = pack32(plain[g*4+inner].qs[iqs*4 .. iqs*4+3])
        for (int inner = 0; inner < 4; inner++) {
            for (int iqs = 0; iqs < 8; iqs++) {
                int8_t q0 = plain[g * 4 + inner].qs[iqs * 4 + 0];
                int8_t q1 = plain[g * 4 + inner].qs[iqs * 4 + 1];
                int8_t q2 = plain[g * 4 + inner].qs[iqs * 4 + 2];
                int8_t q3 = plain[g * 4 + inner].qs[iqs * 4 + 3];
                int32_t packed = ((uint8_t)q0) | ((uint8_t)q1 << 8) | ((uint8_t)q2 << 16) | ((uint8_t)q3 << 24);
                x4[g].qs[inner * 8 + iqs] = packed;
            }
        }
    }
}

int main() {
    struct ggml_init_params params = { 16 * 1024 * 1024, nullptr, true };
    ggml_init(params);

    fprintf(stderr, "=== Q8_1 Layout Analysis ===\n\n");
    fprintf(stderr, "sizeof(block_q8_1_plain) = %zu bytes\n", sizeof(block_q8_1_plain));
    fprintf(stderr, "sizeof(block_q8_1_x4)    = %zu bytes\n", sizeof(block_q8_1_x4));
    fprintf(stderr, "4 × plain                = %zu bytes\n", 4 * sizeof(block_q8_1_plain));
    fprintf(stderr, "\n");
    fprintf(stderr, "Plain memory layout (4 consecutive blocks):\n");
    fprintf(stderr, "  [d0 s0 qs0[32]] [d1 s1 qs1[32]] [d2 s2 qs2[32]] [d3 s3 qs3[32]]\n");
    fprintf(stderr, "  Byte offsets: 0, 36, 72, 108\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "x4 memory layout (1 packed block):\n");
    fprintf(stderr, "  [d0 s0 d1 s1 d2 s2 d3 s3] [qs0[32] qs1[32] qs2[32] qs3[32]]\n");
    fprintf(stderr, "  Scales at offset 0-15, quants at offset 16-143\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "These are NOT memory-compatible!\n");
    fprintf(stderr, "The quantize_q8_1_x4 shader must write in x4 format directly.\n\n");

    // Test quantization correctness
    fprintf(stderr, "=== Quantization Correctness Tests ===\n\n");

    std::vector<int> test_sizes = {32, 128, 256, 1024, 3072, 4096};
    int pass = 0, fail = 0;

    for (int ne : test_sizes) {
        int n_blocks = ne / 32;

        // Generate test data
        std::vector<float> src(ne);
        srand(42 + ne);
        for (int i = 0; i < ne; i++)
            src[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 4.0f;

        // CPU quantize
        std::vector<block_q8_1_plain> plain_blocks(n_blocks);
        quantize_q8_1_cpu(src.data(), plain_blocks.data(), n_blocks);

        // Verify: dequantize and check round-trip error
        float max_err = 0;
        for (int b = 0; b < n_blocks; b++) {
            float d = fp16_to_fp32(plain_blocks[b].d);
            for (int j = 0; j < 32; j++) {
                float deq = d * plain_blocks[b].qs[j];
                float err = fabsf(deq - src[b * 32 + j]);
                if (err > max_err) max_err = err;
            }
        }
        bool ok = max_err < 0.05f; // Q8_1 should be very precise
        fprintf(stderr, "  ne=%-5d: Q8_1 round-trip max_err=%.6f %s\n", ne, max_err, ok ? "PASS" : "FAIL");
        if (ok) pass++; else fail++;

        // Test x4 repacking
        if (n_blocks >= 4) {
            int n_x4 = n_blocks / 4;
            std::vector<block_q8_1_x4> x4_blocks(n_x4);
            repack_to_x4(plain_blocks.data(), x4_blocks.data(), n_x4);

            // Verify x4 layout: dequantize from x4 format
            float x4_max_err = 0;
            for (int g = 0; g < n_x4; g++) {
                for (int inner = 0; inner < 4; inner++) {
                    float d = fp16_to_fp32(x4_blocks[g].ds[inner][0]);
                    for (int iqs = 0; iqs < 8; iqs++) {
                        int32_t packed = x4_blocks[g].qs[inner * 8 + iqs];
                        for (int b = 0; b < 4; b++) {
                            int8_t q = (int8_t)((packed >> (b * 8)) & 0xFF);
                            int elem_idx = (g * 4 + inner) * 32 + iqs * 4 + b;
                            float deq = d * q;
                            float err = fabsf(deq - src[elem_idx]);
                            if (err > x4_max_err) x4_max_err = err;
                        }
                    }
                }
            }
            bool x4_ok = x4_max_err < 0.05f;
            fprintf(stderr, "  ne=%-5d: x4 repack round-trip max_err=%.6f %s\n", ne, x4_max_err, x4_ok ? "PASS" : "FAIL");
            if (x4_ok) pass++; else fail++;

            // Verify: plain and x4 produce the same dequantized values
            bool match = true;
            for (int g = 0; g < n_x4 && match; g++) {
                for (int inner = 0; inner < 4 && match; inner++) {
                    int block_idx = g * 4 + inner;
                    float d_plain = fp16_to_fp32(plain_blocks[block_idx].d);
                    float d_x4 = fp16_to_fp32(x4_blocks[g].ds[inner][0]);
                    if (d_plain != d_x4) {
                        fprintf(stderr, "    d mismatch at block %d: plain=%.6f x4=%.6f\n", block_idx, d_plain, d_x4);
                        match = false;
                    }
                }
            }
            fprintf(stderr, "  ne=%-5d: plain vs x4 scales match: %s\n", ne, match ? "PASS" : "FAIL");
            if (match) pass++; else fail++;
        }
    }

    fprintf(stderr, "\n=== Summary: %d pass, %d fail ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
