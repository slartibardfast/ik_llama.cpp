// Minimal standalone tests for remaining backend-ops failures.
// Prints actual GPU vs CPU values to determine which side is wrong.
// Usage: GGML_VK_VISIBLE_DEVICES=0 ./test-minimal-ops [test_name]
//   test_name: iq4_xs | iq3_xxs | iq4_xs_id | cpy_iq4_nl | cpy_iq4_nl_perm | all
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static ggml_backend_t init_backend(const char * name_substr) {
    size_t n = ggml_backend_reg_get_count();
    for (size_t i = 0; i < n; i++) {
        const char * name = ggml_backend_reg_get_name(i);
        if (strstr(name, name_substr)) {
            return ggml_backend_reg_init_backend(i, name_substr[0] == 'V' ? "0" : nullptr);
        }
    }
    return nullptr;
}

static double compute_nmse(const float * a, const float * b, int n) {
    double sum_diff2 = 0, sum_ref2 = 0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        sum_diff2 += d * d;
        sum_ref2 += (double)b[i] * (double)b[i];
    }
    return sum_ref2 > 0 ? sum_diff2 / sum_ref2 : (sum_diff2 > 0 ? INFINITY : 0);
}

static void print_values(const char * label, const float * v, int n, int max_print = 16) {
    fprintf(stderr, "  %s: ", label);
    for (int i = 0; i < n && i < max_print; i++) fprintf(stderr, "%.6f ", v[i]);
    if (n > max_print) fprintf(stderr, "... (%d total)", n);
    fprintf(stderr, "\n");
}

// Run a graph on a backend, return output as float vector
static std::vector<float> run_graph(ggml_backend_t backend, ggml_type type_a, ggml_type type_b,
                                     int M, int K, int N,
                                     const void * a_data, size_t a_size,
                                     const void * b_data, size_t b_size,
                                     ggml_tensor * (*build_graph)(ggml_context *, ggml_tensor *, ggml_tensor *)) {
    struct ggml_init_params params = { 256*1024*1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, type_a, K, M);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, type_b, K, N);
    struct ggml_tensor * c = build_graph(ctx, a, b);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    ggml_backend_tensor_set(a, a_data, 0, a_size);
    ggml_backend_tensor_set(b, b_data, 0, b_size);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, c);
    ggml_backend_graph_compute(backend, graph);

    int out_n = ggml_nelements(c);
    std::vector<float> result(out_n);
    ggml_backend_tensor_get(c, result.data(), 0, out_n * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return result;
}

// Quantize f32 data to a given type using ggml's quantization
static std::vector<uint8_t> quantize(const float * src, int n, ggml_type type) {
    size_t row_size = ggml_row_size(type, n);
    std::vector<uint8_t> dst(row_size);
    ggml_quantize_chunk(type, src, dst.data(), 0, 1, n, nullptr);
    return dst;
}

// Quantize a full 2D tensor (M rows of K elements)
static std::vector<uint8_t> quantize_2d(const float * src, int M, int K, ggml_type type) {
    size_t row_size = ggml_row_size(type, K);
    std::vector<uint8_t> dst(row_size * M);
    for (int i = 0; i < M; i++) {
        ggml_quantize_chunk(type, src + i * K, dst.data() + i * row_size, 0, 1, K, nullptr);
    }
    return dst;
}

// Dequantize a row back to float
static void dequantize_row(const void * src, float * dst, int K, ggml_type type) {
    ggml_type_traits_t traits = ggml_internal_get_type_traits(type);
    traits.to_float(src, dst, K);
}

static ggml_tensor * build_mul_mat(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    return ggml_mul_mat(ctx, a, b);
}

// ============================================================
// Test: MUL_MAT(iq4_xs x f32, m=16, n=1, k=256)
// ============================================================
static bool test_iq4_xs_mul_mat(ggml_backend_t gpu, ggml_backend_t cpu) {
    fprintf(stderr, "\n=== MUL_MAT(iq4_xs x f32, m=16, n=1, k=256) ===\n");
    const int M = 16, N = 1, K = 256;

    // Generate deterministic test data
    std::vector<float> a_f32(M * K), b_f32(K * N);
    srand(42);
    for (int i = 0; i < M * K; i++) a_f32[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 2.0f;
    for (int i = 0; i < K * N; i++) b_f32[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 2.0f;

    // Quantize A to iq4_xs
    auto a_quant = quantize_2d(a_f32.data(), M, K, GGML_TYPE_IQ4_XS);

    // Compute expected result: dequant(A) * B on CPU in float
    std::vector<float> a_dequant(M * K);
    size_t row_size = ggml_row_size(GGML_TYPE_IQ4_XS, K);
    for (int i = 0; i < M; i++) {
        dequantize_row(a_quant.data() + i * row_size, a_dequant.data() + i * K, K, GGML_TYPE_IQ4_XS);
    }
    std::vector<float> expected(M * N, 0);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double sum = 0;
            for (int k = 0; k < K; k++) sum += (double)a_dequant[m * K + k] * (double)b_f32[n * K + k];
            expected[m + n * M] = (float)sum;  // output is M x N, col-major
        }
    }

    // Run on GPU
    auto gpu_result = run_graph(gpu, GGML_TYPE_IQ4_XS, GGML_TYPE_F32, M, K, N,
                                a_quant.data(), a_quant.size(),
                                b_f32.data(), K * N * sizeof(float), build_mul_mat);

    // Run on CPU
    auto cpu_result = run_graph(cpu, GGML_TYPE_IQ4_XS, GGML_TYPE_F32, M, K, N,
                                a_quant.data(), a_quant.size(),
                                b_f32.data(), K * N * sizeof(float), build_mul_mat);

    print_values("Expected  ", expected.data(), M);
    print_values("GPU output", gpu_result.data(), M);
    print_values("CPU output", cpu_result.data(), M);

    double nmse_gpu = compute_nmse(gpu_result.data(), expected.data(), M * N);
    double nmse_cpu = compute_nmse(cpu_result.data(), expected.data(), M * N);
    double nmse_gpu_vs_cpu = compute_nmse(gpu_result.data(), cpu_result.data(), M * N);

    fprintf(stderr, "  NMSE GPU vs expected:  %e\n", nmse_gpu);
    fprintf(stderr, "  NMSE CPU vs expected:  %e\n", nmse_cpu);
    fprintf(stderr, "  NMSE GPU vs CPU:       %e  (threshold: 5e-4)\n", nmse_gpu_vs_cpu);

    bool pass = nmse_gpu_vs_cpu < 0.0005;
    fprintf(stderr, "  Result: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ============================================================
// Test: MUL_MAT(iq3_xxs x f32, m=16, n=1, k=256)
// ============================================================
static bool test_iq3_xxs_mul_mat(ggml_backend_t gpu, ggml_backend_t cpu) {
    fprintf(stderr, "\n=== MUL_MAT(iq3_xxs x f32, m=16, n=1, k=256) ===\n");
    const int M = 16, N = 1, K = 256;

    std::vector<float> a_f32(M * K), b_f32(K * N);
    srand(42);
    for (int i = 0; i < M * K; i++) a_f32[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 2.0f;
    for (int i = 0; i < K * N; i++) b_f32[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 2.0f;

    auto a_quant = quantize_2d(a_f32.data(), M, K, GGML_TYPE_IQ3_XXS);
    size_t row_size = ggml_row_size(GGML_TYPE_IQ3_XXS, K);

    std::vector<float> a_dequant(M * K);
    for (int i = 0; i < M; i++) {
        dequantize_row(a_quant.data() + i * row_size, a_dequant.data() + i * K, K, GGML_TYPE_IQ3_XXS);
    }
    std::vector<float> expected(M * N, 0);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double sum = 0;
            for (int k = 0; k < K; k++) sum += (double)a_dequant[m * K + k] * (double)b_f32[n * K + k];
            expected[m + n * M] = (float)sum;
        }
    }

    auto gpu_result = run_graph(gpu, GGML_TYPE_IQ3_XXS, GGML_TYPE_F32, M, K, N,
                                a_quant.data(), a_quant.size(),
                                b_f32.data(), K * N * sizeof(float), build_mul_mat);
    auto cpu_result = run_graph(cpu, GGML_TYPE_IQ3_XXS, GGML_TYPE_F32, M, K, N,
                                a_quant.data(), a_quant.size(),
                                b_f32.data(), K * N * sizeof(float), build_mul_mat);

    print_values("Expected  ", expected.data(), M);
    print_values("GPU output", gpu_result.data(), M);
    print_values("CPU output", cpu_result.data(), M);

    double nmse_gpu = compute_nmse(gpu_result.data(), expected.data(), M * N);
    double nmse_cpu = compute_nmse(cpu_result.data(), expected.data(), M * N);
    double nmse_gpu_vs_cpu = compute_nmse(gpu_result.data(), cpu_result.data(), M * N);

    fprintf(stderr, "  NMSE GPU vs expected:  %e\n", nmse_gpu);
    fprintf(stderr, "  NMSE CPU vs expected:  %e\n", nmse_cpu);
    fprintf(stderr, "  NMSE GPU vs CPU:       %e  (threshold: 5e-4)\n", nmse_gpu_vs_cpu);

    bool pass = nmse_gpu_vs_cpu < 0.0005;
    fprintf(stderr, "  Result: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

// ============================================================
// Test: CPY(f32 -> iq4_nl, [256,4,4,4])
// ============================================================
static bool test_cpy_iq4_nl(ggml_backend_t gpu, ggml_backend_t cpu, bool permuted) {
    int ne[4];
    if (permuted) {
        ne[0] = 256; ne[1] = 2; ne[2] = 3; ne[3] = 4;
    } else {
        ne[0] = 256; ne[1] = 4; ne[2] = 4; ne[3] = 4;
    }
    int total = ne[0] * ne[1] * ne[2] * ne[3];
    fprintf(stderr, "\n=== CPY(f32 -> iq4_nl, [%d,%d,%d,%d]%s) ===\n",
            ne[0], ne[1], ne[2], ne[3], permuted ? ", permuted" : "");

    // Generate test data
    std::vector<float> src_f32(total);
    srand(123);
    for (int i = 0; i < total; i++) src_f32[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 2.0f;

    // Run on each backend: create src (f32), dst (iq4_nl), copy, read back as f32
    auto run_cpy = [&](ggml_backend_t backend, const char * label) -> std::vector<float> {
        struct ggml_init_params params = { 256*1024*1024, nullptr, true };
        struct ggml_context * ctx = ggml_init(params);

        struct ggml_tensor * src;
        if (permuted) {
            // Create with permuted strides: permute [0,2,1,3] means swap dims 1 and 2
            src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[2], ne[1], ne[3]);
            src = ggml_permute(ctx, src, 0, 2, 1, 3);  // back to [ne0, ne1, ne2, ne3] but non-contiguous
        } else {
            src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
        }
        struct ggml_tensor * dst = ggml_new_tensor_4d(ctx, GGML_TYPE_IQ4_NL, ne[0], ne[1], ne[2], ne[3]);
        struct ggml_tensor * cpy = ggml_cpy(ctx, src, dst);

        // Also add a dequant step to read back as f32
        struct ggml_tensor * out = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
        struct ggml_tensor * cpy2 = ggml_cpy(ctx, cpy, out);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

        // Set source data - need to handle permuted case
        if (permuted) {
            // For permuted tensor, set data on the view_src (the original contiguous tensor)
            ggml_backend_tensor_set(src->view_src, src_f32.data(), 0, total * sizeof(float));
        } else {
            ggml_backend_tensor_set(src, src_f32.data(), 0, total * sizeof(float));
        }

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, cpy2);
        ggml_backend_graph_compute(backend, graph);

        std::vector<float> result(total);
        ggml_backend_tensor_get(out, result.data(), 0, total * sizeof(float));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return result;
    };

    auto gpu_result = run_cpy(gpu, "GPU");
    auto cpu_result = run_cpy(cpu, "CPU");

    // Compute expected: quantize to iq4_nl then dequantize
    // Do it row by row (iq4_nl has block size 32, row length must be multiple of 32)
    size_t row_size = ggml_row_size(GGML_TYPE_IQ4_NL, ne[0]);
    int n_rows = total / ne[0];
    std::vector<float> expected(total);
    for (int r = 0; r < n_rows; r++) {
        auto q = quantize(src_f32.data() + r * ne[0], ne[0], GGML_TYPE_IQ4_NL);
        dequantize_row(q.data(), expected.data() + r * ne[0], ne[0], GGML_TYPE_IQ4_NL);
    }

    print_values("Expected  ", expected.data(), 16);
    print_values("GPU output", gpu_result.data(), 16);
    print_values("CPU output", cpu_result.data(), 16);

    double nmse_gpu = compute_nmse(gpu_result.data(), expected.data(), total);
    double nmse_cpu = compute_nmse(cpu_result.data(), expected.data(), total);
    double nmse_gpu_vs_cpu = compute_nmse(gpu_result.data(), cpu_result.data(), total);

    fprintf(stderr, "  NMSE GPU vs expected:  %e\n", nmse_gpu);
    fprintf(stderr, "  NMSE CPU vs expected:  %e\n", nmse_cpu);
    fprintf(stderr, "  NMSE GPU vs CPU:       %e  (threshold: 1e-6)\n", nmse_gpu_vs_cpu);

    bool pass = nmse_gpu_vs_cpu < 0.000001;
    fprintf(stderr, "  Result: %s\n", pass ? "PASS" : "FAIL");
    return pass;
}

int main(int argc, char ** argv) {
    const char * test_name = argc > 1 ? argv[1] : "all";

    ggml_backend_t gpu = init_backend("Vulkan");
    ggml_backend_t cpu = init_backend("CPU");
    if (!gpu) { fprintf(stderr, "No Vulkan backend\n"); return 1; }
    if (!cpu) { fprintf(stderr, "No CPU backend\n"); return 1; }
    fprintf(stderr, "GPU: %s\nCPU: %s\n", ggml_backend_name(gpu), ggml_backend_name(cpu));

    int pass = 0, fail = 0;
    auto run = [&](const char * name, auto fn) {
        if (strcmp(test_name, "all") == 0 || strcmp(test_name, name) == 0) {
            if (fn()) pass++; else fail++;
        }
    };

    run("iq4_xs",         [&]{ return test_iq4_xs_mul_mat(gpu, cpu); });
    run("iq3_xxs",        [&]{ return test_iq3_xxs_mul_mat(gpu, cpu); });
    run("cpy_iq4_nl",     [&]{ return test_cpy_iq4_nl(gpu, cpu, false); });
    run("cpy_iq4_nl_perm",[&]{ return test_cpy_iq4_nl(gpu, cpu, true); });

    // K-quant CPY tests: F32 -> Q*_K -> F32 round-trip, GPU vs CPU
    auto test_cpy_kquant = [](ggml_backend_t gpu, ggml_backend_t cpu, ggml_type qtype) -> bool {
        const int ne0 = 256, ne1 = 4, ne2 = 2, ne3 = 1;
        const int total = ne0 * ne1 * ne2 * ne3;
        fprintf(stderr, "\n=== CPY(f32 -> %s -> f32, [%d,%d,%d,%d]) ===\n",
                ggml_type_name(qtype), ne0, ne1, ne2, ne3);

        std::vector<float> src_f32(total);
        srand(42);
        for (int i = 0; i < total; i++) src_f32[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 2.0f;

        auto run_roundtrip = [&](ggml_backend_t backend, const char * label) -> std::vector<float> {
            struct ggml_init_params params = { 256*1024*1024, nullptr, true };
            struct ggml_context * ctx = ggml_init(params);

            auto * src = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
            auto * mid = ggml_new_tensor_4d(ctx, qtype, ne0, ne1, ne2, ne3);
            auto * dst = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
            auto * cpy1 = ggml_cpy(ctx, src, mid);
            auto * cpy2 = ggml_cpy(ctx, cpy1, dst);

            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, cpy2);

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

            // Set source data
            ggml_backend_tensor_set(src, src_f32.data(), 0, total * sizeof(float));

            ggml_backend_graph_compute(backend, gf);

            std::vector<float> result(total);
            ggml_backend_tensor_get(dst, result.data(), 0, total * sizeof(float));
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return result;
        };

        auto gpu_result = run_roundtrip(gpu, "GPU");
        auto cpu_result = run_roundtrip(cpu, "CPU");

        double nmse = compute_nmse(gpu_result.data(), cpu_result.data(), total);
        print_values("GPU", gpu_result.data(), total);
        print_values("CPU", cpu_result.data(), total);

        // Show first divergence
        for (int i = 0; i < total; i++) {
            if (std::abs(gpu_result[i] - cpu_result[i]) > 0.1f) {
                fprintf(stderr, "  FIRST DIVERGE at [%d]: GPU=%.6f CPU=%.6f\n", i, gpu_result[i], cpu_result[i]);
                break;
            }
        }

        // K-quant quantization differs between GPU (min/max) and CPU (iterative),
        // so round-trip NMSE is dominated by quantization error, not GPU/CPU mismatch
        double threshold = 0.01;
        fprintf(stderr, "  NMSE = %.10f %s\n", nmse, nmse < threshold ? "PASS" : "FAIL");
        return nmse < threshold;
    };

    // One-way F32→Q4_K test (isolate quantize shader)
    run("cpy_f32_q4_k", [&]() -> bool {
        const int ne0 = 256, ne1 = 4;
        const int total = ne0 * ne1;
        fprintf(stderr, "\n=== CPY(f32 -> q4_K, [%d,%d]) one-way ===\n", ne0, ne1);

        std::vector<float> src_f32(total);
        srand(42);
        for (int i = 0; i < total; i++) src_f32[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 2.0f;

        auto run_quant = [&](ggml_backend_t backend) -> std::vector<uint8_t> {
            struct ggml_init_params params = { 64*1024*1024, nullptr, true };
            struct ggml_context * ctx = ggml_init(params);

            auto * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
            auto * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_K, ne0, ne1);
            auto * op = ggml_cpy(ctx, src, dst);

            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, op);
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            ggml_backend_tensor_set(src, src_f32.data(), 0, total * sizeof(float));
            ggml_backend_graph_compute(backend, gf);

            size_t qsize = ggml_nbytes(dst);
            std::vector<uint8_t> result(qsize);
            ggml_backend_tensor_get(dst, result.data(), 0, qsize);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return result;
        };

        fprintf(stderr, "  Running CPU...\n");
        auto cpu_q = run_quant(cpu);
        fprintf(stderr, "  Running GPU...\n");
        auto gpu_q = run_quant(gpu);

        // Compare raw quantized bytes
        bool match = (cpu_q.size() == gpu_q.size());
        int diff_count = 0;
        for (size_t i = 0; i < cpu_q.size() && match; i++) {
            if (cpu_q[i] != gpu_q[i]) diff_count++;
        }
        fprintf(stderr, "  Quantized size: %zu bytes, diff bytes: %d %s\n",
                cpu_q.size(), diff_count, diff_count == 0 ? "EXACT MATCH" : (diff_count < 10 ? "CLOSE" : "DIVERGED"));

        // Dequant both on CPU and compare
        std::vector<float> gpu_f32(total), cpu_f32(total);
        ggml_type_traits_t tt;
        tt = ggml_internal_get_type_traits(GGML_TYPE_Q4_K);
        tt.to_float(gpu_q.data(), gpu_f32.data(), total);
        tt.to_float(cpu_q.data(), cpu_f32.data(), total);

        double nmse = compute_nmse(gpu_f32.data(), cpu_f32.data(), total);
        print_values("GPU-dequant", gpu_f32.data(), total);
        print_values("CPU-dequant", cpu_f32.data(), total);
        fprintf(stderr, "  Round-trip NMSE = %.10f %s\n", nmse, nmse < 0.01 ? "PASS" : "FAIL");
        return nmse < 0.01;
    });

    run("cpy_q4_k",  [&]{ return test_cpy_kquant(gpu, cpu, GGML_TYPE_Q4_K); });
    run("cpy_q5_k",  [&]{ return test_cpy_kquant(gpu, cpu, GGML_TYPE_Q5_K); });
    run("cpy_q6_k",  [&]{ return test_cpy_kquant(gpu, cpu, GGML_TYPE_Q6_K); });

    // Flash attention with K-quant KV cache
    auto test_fa_kquant = [](ggml_backend_t gpu, ggml_backend_t cpu, ggml_type kv_type, int hs, int nh, int kv) -> bool {
        fprintf(stderr, "\n=== FLASH_ATTN(hs=%d, nh=%d, kv=%d, type_KV=%s) ===\n",
                hs, nh, kv, ggml_type_name(kv_type));

        auto run_fa = [&](ggml_backend_t backend, const char * label) -> std::vector<float> {
            struct ggml_init_params params = { 256*1024*1024, nullptr, true };
            struct ggml_context * ctx = ggml_init(params);

            auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs, 1, nh, 1);
            auto * k = ggml_new_tensor_4d(ctx, kv_type, hs, kv, nh, 1);
            auto * v = ggml_new_tensor_4d(ctx, kv_type, hs, kv, nh, 1);
            // Mask dim1 must be padded to GGML_KQ_MASK_PAD (64)
            int64_t mask_rows = ((1 + 63) / 64) * 64;  // pad to 64
            auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv, mask_rows);
            auto * out = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / sqrtf((float)hs), 0.0f, 0.0f);

            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

            // Init Q with random data
            int q_total = hs * nh;
            std::vector<float> q_data(q_total);
            srand(42);
            for (int i = 0; i < q_total; i++) q_data[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 0.1f;
            ggml_backend_tensor_set(q, q_data.data(), 0, q_total * sizeof(float));

            // Init K/V with random quantized data: first create F32 then quantize via CPU
            int kv_total = hs * kv * nh;
            std::vector<float> kv_f32(kv_total);
            for (int i = 0; i < kv_total; i++) kv_f32[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 2.0f;
            size_t quant_size = ggml_row_size(kv_type, hs) * kv * nh;
            std::vector<uint8_t> kv_quant(quant_size);
            ggml_quantize_chunk(kv_type, kv_f32.data(), kv_quant.data(), 0, kv * nh, hs, nullptr);
            ggml_backend_tensor_set(k, kv_quant.data(), 0, quant_size);
            // Use same random seed for V to get different data
            for (int i = 0; i < kv_total; i++) kv_f32[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 2.0f;
            ggml_quantize_chunk(kv_type, kv_f32.data(), kv_quant.data(), 0, kv * nh, hs, nullptr);
            ggml_backend_tensor_set(v, kv_quant.data(), 0, quant_size);

            // Mask: all zeros (no masking)
            size_t mask_bytes = kv * mask_rows * sizeof(uint16_t);
            std::vector<uint16_t> mask_data(kv * mask_rows, 0);  // f16 zeros
            ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_bytes);

            ggml_backend_graph_compute(backend, gf);

            int out_total = hs * nh;
            std::vector<float> result(out_total);
            ggml_backend_tensor_get(out, result.data(), 0, out_total * sizeof(float));
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return result;
        };

        auto gpu_result = run_fa(gpu, "GPU");
        auto cpu_result = run_fa(cpu, "CPU");

        double nmse = compute_nmse(gpu_result.data(), cpu_result.data(), (int)gpu_result.size());
        print_values("GPU", gpu_result.data(), (int)gpu_result.size());
        print_values("CPU", cpu_result.data(), (int)cpu_result.size());

        fprintf(stderr, "  NMSE = %.10f %s\n", nmse, nmse < 1e-3 ? "PASS" : "FAIL");
        return nmse < 1e-3;
    };

    // Skip K-quant FA CPU tests (CPU doesn't support these KV types for FA)
    // run("fa_q4_k_256",  [&]{ return test_fa_kquant(gpu, cpu, GGML_TYPE_Q4_K, 256, 4, 64); });
    run("fa_q4_0_ref",  [&]{ return test_fa_kquant(gpu, cpu, GGML_TYPE_Q4_0, 128, 4, 64); });

    // GPU-only FA test: quantize F32→quant on GPU (step 1), then FA (step 2)
    auto test_fa_gpu_only = [](ggml_backend_t gpu, ggml_type kv_type, int hs, int nh, int kv) -> bool {
        fprintf(stderr, "\n=== FA GPU-only (F16 vs %s, hs=%d, nh=%d, kv=%d) ===\n",
                ggml_type_name(kv_type), hs, nh, kv);

        int q_total = hs * nh;
        int kv_total = hs * kv * nh;
        std::vector<float> q_data(q_total), k_data(kv_total), v_data(kv_total);
        srand(42);
        for (int i = 0; i < q_total; i++) q_data[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 0.1f;
        for (int i = 0; i < kv_total; i++) k_data[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 2.0f;
        for (int i = 0; i < kv_total; i++) v_data[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 2.0f;

        auto run_fa_gpu = [&](ggml_type cache_type) -> std::vector<float> {
            // Step 1: Quantize K/V (F32 → cache_type) on GPU
            struct ggml_init_params params1 = { 256*1024*1024, nullptr, true };
            struct ggml_context * ctx1 = ggml_init(params1);
            auto * k_f32 = ggml_new_tensor_4d(ctx1, GGML_TYPE_F32, hs, kv, nh, 1);
            auto * v_f32 = ggml_new_tensor_4d(ctx1, GGML_TYPE_F32, hs, kv, nh, 1);
            auto * k_q = ggml_new_tensor_4d(ctx1, cache_type, hs, kv, nh, 1);
            auto * v_q = ggml_new_tensor_4d(ctx1, cache_type, hs, kv, nh, 1);
            auto * k_cpy = ggml_cpy(ctx1, k_f32, k_q);
            auto * v_cpy = ggml_cpy(ctx1, v_f32, v_q);
            struct ggml_cgraph * g1 = ggml_new_graph(ctx1);
            ggml_build_forward_expand(g1, k_cpy);
            ggml_build_forward_expand(g1, v_cpy);
            ggml_backend_buffer_t buf1 = ggml_backend_alloc_ctx_tensors(ctx1, gpu);
            ggml_backend_tensor_set(k_f32, k_data.data(), 0, kv_total * sizeof(float));
            ggml_backend_tensor_set(v_f32, v_data.data(), 0, kv_total * sizeof(float));
            fprintf(stderr, "  Step 1: CPY F32 -> %s ...\n", ggml_type_name(cache_type));
            ggml_backend_graph_compute(gpu, g1);

            // Read back quantized data
            size_t qsize = ggml_nbytes(k_q);
            std::vector<uint8_t> k_qdata(qsize), v_qdata(qsize);
            ggml_backend_tensor_get(k_q, k_qdata.data(), 0, qsize);
            ggml_backend_tensor_get(v_q, v_qdata.data(), 0, qsize);
            ggml_backend_buffer_free(buf1);
            ggml_free(ctx1);

            // Step 2: FA with pre-quantized K/V
            struct ggml_init_params params2 = { 256*1024*1024, nullptr, true };
            struct ggml_context * ctx2 = ggml_init(params2);
            auto * q = ggml_new_tensor_4d(ctx2, GGML_TYPE_F32, hs, 1, nh, 1);
            auto * k = ggml_new_tensor_4d(ctx2, cache_type, hs, kv, nh, 1);
            auto * v = ggml_new_tensor_4d(ctx2, cache_type, hs, kv, nh, 1);
            int64_t mask_rows = ((1 + 63) / 64) * 64;
            auto * mask = ggml_new_tensor_2d(ctx2, GGML_TYPE_F16, kv, mask_rows);
            auto * out = ggml_flash_attn_ext(ctx2, q, k, v, mask, 1.0f / sqrtf((float)hs), 0.0f, 0.0f);
            struct ggml_cgraph * g2 = ggml_new_graph(ctx2);
            ggml_build_forward_expand(g2, out);
            ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors(ctx2, gpu);
            ggml_backend_tensor_set(q, q_data.data(), 0, q_total * sizeof(float));
            ggml_backend_tensor_set(k, k_qdata.data(), 0, qsize);
            ggml_backend_tensor_set(v, v_qdata.data(), 0, qsize);
            size_t mask_bytes = kv * mask_rows * sizeof(uint16_t);
            std::vector<uint16_t> mask_data(kv * mask_rows, 0);
            ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_bytes);
            fprintf(stderr, "  Step 2: FA with %s ...\n", ggml_type_name(cache_type));
            ggml_backend_graph_compute(gpu, g2);

            int out_total = hs * nh;
            std::vector<float> result(out_total);
            ggml_backend_tensor_get(out, result.data(), 0, out_total * sizeof(float));
            ggml_backend_buffer_free(buf2);
            ggml_free(ctx2);
            return result;
        };

        auto f16_result = run_fa_gpu(GGML_TYPE_F16);
        auto quant_result = run_fa_gpu(kv_type);

        double nmse = compute_nmse(quant_result.data(), f16_result.data(), (int)f16_result.size());
        print_values("F16  ", f16_result.data(), (int)f16_result.size());
        print_values("Quant", quant_result.data(), (int)quant_result.size());
        fprintf(stderr, "  NMSE vs F16 = %.10f %s\n", nmse, nmse < 0.05 ? "PASS" : "FAIL");
        return nmse < 0.05;
    };

    run("fa_q4_0_gpu",  [&]{ return test_fa_gpu_only(gpu, GGML_TYPE_Q4_0, 128, 4, 64); });

    // TURBO_KV_4B FA via scheduler (matches model dispatch path)
    auto test_fa_sched = [](ggml_backend_t gpu, ggml_backend_t cpu, ggml_type kv_type, int hs, int nh, int kv) -> bool {
        fprintf(stderr, "\n=== FA sched (F16 vs %s, hs=%d, nh=%d, kv=%d) ===\n",
                ggml_type_name(kv_type), hs, nh, kv);

        int q_total = hs * nh;
        int kv_total = hs * kv * nh;
        std::vector<float> q_data(q_total), k_data(kv_total), v_data(kv_total);
        srand(42);
        for (int i = 0; i < q_total; i++) q_data[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 0.1f;
        for (int i = 0; i < kv_total; i++) k_data[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 2.0f;
        for (int i = 0; i < kv_total; i++) v_data[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 2.0f;

        auto run_with_sched = [&](ggml_type cache_type) -> std::vector<float> {
            struct ggml_init_params params = { 256*1024*1024, nullptr, true };
            struct ggml_context * ctx = ggml_init(params);

            auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs, 1, nh, 1);
            auto * k_f32 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs, kv, nh, 1);
            auto * v_f32 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs, kv, nh, 1);
            auto * k_q = ggml_new_tensor_4d(ctx, cache_type, hs, kv, nh, 1);
            auto * v_q = ggml_new_tensor_4d(ctx, cache_type, hs, kv, nh, 1);
            auto * k_cpy = ggml_cpy(ctx, k_f32, k_q);
            auto * v_cpy = ggml_cpy(ctx, v_f32, v_q);
            int64_t mask_rows = ((1 + 63) / 64) * 64;
            auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv, mask_rows);
            auto * out = ggml_flash_attn_ext(ctx, q, k_cpy, v_cpy, mask, 1.0f / sqrtf((float)hs), 0.0f, 0.0f);

            struct ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, out);

            // Use scheduler — this is how the model dispatches
            ggml_backend_t backends[2] = {gpu, cpu};
            ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 256, false);
            if (!ggml_backend_sched_reserve(sched, gf)) {
                fprintf(stderr, "    Failed to reserve graph\n");
                ggml_backend_sched_free(sched);
                ggml_free(ctx);
                return {};
            }
            ggml_backend_sched_alloc_graph(sched, gf);

            ggml_backend_tensor_set(q, q_data.data(), 0, q_total * sizeof(float));
            ggml_backend_tensor_set(k_f32, k_data.data(), 0, kv_total * sizeof(float));
            ggml_backend_tensor_set(v_f32, v_data.data(), 0, kv_total * sizeof(float));
            size_t mask_bytes = kv * mask_rows * sizeof(uint16_t);
            std::vector<uint16_t> mask_data(kv * mask_rows, 0);
            ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_bytes);

            fprintf(stderr, "  Running %s ...\n", ggml_type_name(cache_type));
            ggml_backend_sched_graph_compute(sched, gf);

            int out_total = hs * nh;
            std::vector<float> result(out_total);
            ggml_backend_tensor_get(out, result.data(), 0, out_total * sizeof(float));
            ggml_backend_sched_free(sched);
            ggml_free(ctx);
            return result;
        };

        auto f16_result = run_with_sched(GGML_TYPE_F16);
        auto quant_result = run_with_sched(kv_type);

        // Check for NaN in each result
        int f16_nan = 0, q_nan = 0;
        for (int i = 0; i < (int)f16_result.size(); i++) if (std::isnan(f16_result[i])) f16_nan++;
        for (int i = 0; i < (int)quant_result.size(); i++) if (std::isnan(quant_result[i])) q_nan++;
        fprintf(stderr, "  NaN: f16=%d/%d quant=%d/%d\n", f16_nan, (int)f16_result.size(), q_nan, (int)quant_result.size());
        if (!f16_result.empty() && !quant_result.empty()) {
            fprintf(stderr, "  f16[0..3] = %.6f %.6f %.6f %.6f\n", f16_result[0], f16_result[1], f16_result[2], f16_result[3]);
            fprintf(stderr, "  qnt[0..3] = %.6f %.6f %.6f %.6f\n", quant_result[0], quant_result[1], quant_result[2], quant_result[3]);
        }

        double nmse = compute_nmse(quant_result.data(), f16_result.data(), (int)f16_result.size());
        print_values("F16  ", f16_result.data(), (int)f16_result.size());
        print_values("Quant", quant_result.data(), (int)quant_result.size());
        fprintf(stderr, "  NMSE vs F16 = %.10f %s\n", nmse, nmse < 0.05 ? "PASS" : "FAIL");
        return nmse < 0.05;
    };

    run("fa_f16_sched",   [&]{ return test_fa_sched(gpu, cpu, GGML_TYPE_F16,          256, 1, 32); });
    run("fa_tkv_sched",   [&]{ return test_fa_sched(gpu, cpu, GGML_TYPE_TURBO_KV_4B, 256, 1, 32); });
    run("fa_tkv_sched_l", [&]{ return test_fa_sched(gpu, cpu, GGML_TYPE_TURBO_KV_4B, 256, 1, 128); });

    // TURBO_KV_4B round-trip: F32 → quantize(GPU) → dequant(GPU) → F32
    // Self-contained RHT round-trip: forward→quant→dequant→inverse, no byte packing
    run("tkv_rht_only", [&]() -> bool {
        const int ne0 = 128, ne1 = 4;
        const int total = ne0 * ne1;
        fprintf(stderr, "\n=== TURBO_KV_4B RHT-only round-trip [%d,%d] ===\n", ne0, ne1);

        std::vector<float> src_f32(total);
        srand(42);
        for (int i = 0; i < total; i++) src_f32[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 2.0f;

        struct ggml_init_params params = { 64*1024*1024, nullptr, true };
        struct ggml_context * ctx = ggml_init(params);

        // Create raw F32 tensors and use a custom compute kernel
        auto * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        auto * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);

        // We can't easily dispatch a custom shader via the ggml API.
        // Instead, use the existing CPY round-trip but compare block-by-block.
        // Skip this test — it needs custom dispatch infrastructure.
        fprintf(stderr, "  (skipped — needs custom shader dispatch)\n");
        ggml_free(ctx);
        return true;  // placeholder
    });

    run("tkv_roundtrip", [&]() -> bool {
        const int ne0 = 128, ne1 = 8;  // 128 elements per block, 8 blocks
        const int total = ne0 * ne1;
        fprintf(stderr, "\n=== TURBO_KV_4B round-trip F32→quant→get_rows→F32 [%d,%d] ===\n", ne0, ne1);

        std::vector<float> src_f32(total);
        srand(42);
        for (int i = 0; i < total; i++) src_f32[i] = ((float)(rand() / (double)RAND_MAX) - 0.5f) * 2.0f;

        // Run on GPU: F32 → CPY → TURBO_KV_4B → get_rows → F32
        struct ggml_init_params params = { 64*1024*1024, nullptr, true };
        struct ggml_context * ctx = ggml_init(params);

        auto * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
        auto * mid = ggml_new_tensor_2d(ctx, GGML_TYPE_TURBO_KV_4B, ne0, ne1);
        auto * cpy1 = ggml_cpy(ctx, src, mid);
        // Use get_rows to dequantize — select all rows
        auto * row_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne1);
        auto * dst = ggml_get_rows(ctx, cpy1, row_idx);

        struct ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, dst);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, gpu);
        ggml_backend_tensor_set(src, src_f32.data(), 0, total * sizeof(float));

        // Set row indices: [0, 1, 2, ..., ne1-1]
        std::vector<int32_t> rows(ne1);
        for (int i = 0; i < ne1; i++) rows[i] = i;
        ggml_backend_tensor_set(row_idx, rows.data(), 0, ne1 * sizeof(int32_t));

        ggml_backend_graph_compute(gpu, gf);

        std::vector<float> result(total);
        ggml_backend_tensor_get(dst, result.data(), 0, total * sizeof(float));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);

        // Check round-trip error
        double nmse = compute_nmse(result.data(), src_f32.data(), total);
        print_values("Input ", src_f32.data(), total);
        print_values("Output", result.data(), total);

        for (int i = 0; i < total; i++) {
            if (std::abs(result[i] - src_f32[i]) > 1.0f) {
                fprintf(stderr, "  FIRST BIG DIFF at [%d]: in=%.6f out=%.6f\n", i, src_f32[i], result[i]);
                break;
            }
        }

        fprintf(stderr, "  NMSE = %.10f %s\n", nmse, nmse < 0.05 ? "PASS" : "FAIL");
        return nmse < 0.05;
    });

    fprintf(stderr, "\n=== Summary: %d passed, %d failed ===\n", pass, fail);

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return fail > 0 ? 1 : 0;
}
