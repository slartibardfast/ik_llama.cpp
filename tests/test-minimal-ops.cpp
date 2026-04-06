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

    fprintf(stderr, "\n=== Summary: %d passed, %d failed ===\n", pass, fail);

    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return fail > 0 ? 1 : 0;
}
