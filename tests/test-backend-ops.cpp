#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <vector>


static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    // static RNG initialization (revisit if n_threads stops being constant)
    static const size_t n_threads = std::thread::hardware_concurrency();
    static std::vector<std::default_random_engine> generators = []() {
        std::random_device rd;
        std::vector<std::default_random_engine> vec;
        vec.reserve(n_threads);
        //for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(1234 + i); } // fixed seed
        for (size_t i = 0; i < n_threads; i++) { vec.emplace_back(rd()); }
        return vec;
    }();

    size_t size = ggml_nelements(tensor);
    std::vector<float> data(size);

    auto init_thread = [&](size_t ith, size_t start, size_t end) {
        std::uniform_real_distribution<float> distribution(min, max);
        for (size_t i = start; i < end; i++) {
            data[i] = distribution(generators[ith]);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (size_t i = 0; i < n_threads; i++) {
        size_t start =     i*size/n_threads;
        size_t end   = (i+1)*size/n_threads;
        threads.emplace_back(init_thread, i, start, end);
    }
    for (auto & t : threads) {
        t.join();
    }

#if 0
    const char * val_str = getenv("GGML_TEST_EPS");
    float val = 1e-9f;
    if (val_str != nullptr) {
        val = std::stof(val_str);
        printf("GGML_TEST_EPS=%e\n", val);
    }

    // test quantization with very small values that may result in nan scales due to division by zero
    if (ggml_is_quantized(tensor->type)) {
        for (int i = 0; i < 256; i++) {
            data[i] = val;
        }
    }
#endif

    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(tensor, data.data(), 0, size * sizeof(float));
    } else if (ggml_is_quantized(tensor->type) || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        GGML_ASSERT(size % ggml_blck_size(tensor->type) == 0);
        std::vector<uint8_t> dataq(ggml_row_size(tensor->type, size));
        std::vector<float> imatrix(tensor->ne[0], 1.0f); // dummy importance matrix
        const float * im = imatrix.data();
        if (!ggml_quantize_requires_imatrix(tensor->type)) {
            // when the imatrix is optional, we want to test both quantization with and without imatrix
            // use one of the random numbers to decide
            if (data[0] > 0.5f*(min + max)) {
                im = nullptr;
            }
        }

        ggml_quantize_chunk(tensor->type, data.data(), dataq.data(), 0, size/tensor->ne[0], tensor->ne[0], im);
        GGML_ASSERT(ggml_validate_row_data(tensor->type, dataq.data(), dataq.size()));
        // TODO: other cases
        //#pragma omp parallel for
        //for (int i = 0; i < tensor->ne[1]; i++) {
        //    ggml_quantize_chunk(tensor->type, data.data(), dataq.data(),
        //        i * tensor->ne[0], 1, tensor->ne[0], im);
        //}

        ggml_backend_tensor_set(tensor, dataq.data(), 0, dataq.size());
    } else if (tensor->type == GGML_TYPE_I8 || tensor->type == GGML_TYPE_I16 || tensor->type == GGML_TYPE_I32) {
        // This is going to create some weird integers though.
        ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static std::vector<float> tensor_to_float(const ggml_tensor * t) {
    std::vector<float> tv;
    tv.reserve(ggml_nelements(t));

    std::vector<uint8_t> buf(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));

    ggml_type_traits_t tt = ggml_internal_get_type_traits(t->type);
    size_t bs = ggml_blck_size(t->type);
    std::vector<float> vq(ggml_blck_size(t->type));
    bool quantized = ggml_is_quantized(t->type);

    // access elements by index to avoid gaps in views
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0 += bs) {
                    size_t i = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1] + i0/bs*t->nb[0];
                    if (t->type == GGML_TYPE_F16) {
                        tv.push_back(ggml_fp16_to_fp32(*(ggml_fp16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_BF16) {
                        tv.push_back(ggml_bf16_to_fp32(*(ggml_bf16_t*)&buf[i]));
                    } else if (t->type == GGML_TYPE_F32) {
                        tv.push_back(*(float *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I32) {
                        tv.push_back((float)*(int32_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I16) {
                        tv.push_back((float)*(int16_t *) &buf[i]);
                    } else if (t->type == GGML_TYPE_I8) {
                        tv.push_back((float)*(int8_t *) &buf[i]);
                    } else if (quantized) {
                        tt.to_float(&buf[i], vq.data(), bs);
                        tv.insert(tv.end(), vq.begin(), vq.end());
                    } else {
                        GGML_ABORT("fatal error");
                    }
                }
            }
        }
    }

    return tv;
}

/*
static double cosine_similarity(const float * v1, const float * v2, size_t n) {
    double dot = 0.0;
    double mag1 = 0.0;
    double mag2 = 0.0;

    for (size_t i = 0; i < n; i++) {
        if (std::isnan(v1[i]) || std::isnan(v2[i])) {
            return -1.0f;
        }
        if (std::isinf(v1[i]) && std::isinf(v2[i])) {
            continue;
        }
        dot  += v1[i]*v2[i];
        mag1 += v1[i]*v1[i];
        mag2 += v2[i]*v2[i];
    }

    return dot/sqrt(mag1*mag2);
}

static float distance(const float * v1, const float * v2, size_t n) {
    double d = 0.0;

    for (size_t i = 0; i < n; i++) {
        if (std::isnan(v1[i]) || std::isnan(v2[i])) {
            return INFINITY;
        }
        if (std::isinf(v1[i]) && std::isinf(v2[i])) {
            continue;
        }
        d += (v1[i] - v2[i])*(v1[i] - v2[i]);
    }

    return sqrt(d);
}

static float vec_len(const float * v, size_t n) {
    double d = 0.0;

    for (size_t i = 0; i < n; i++) {
        if (std::isnan(v[i])) {
            return INFINITY;
        }
        if (std::isinf(v[i])) {
            continue;
        }
        d += v[i]*v[i];
    }

    return sqrt(d);
}
*/

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

// utils for printing the variables of the test cases
#define VAR_TO_STR(x) (#x "=" + var_to_str(x))

template<typename T>
static std::string var_to_str(const T & x) {
    return std::to_string(x);
}

template<typename T, size_t N>
static std::string var_to_str(const T (&x)[N]) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

template<typename T, size_t N>
static std::string var_to_str(const std::array<T, N> & x) {
    std::string s = "[";
    for (size_t i = 0; i < N; i++) {
        if (i > 0) {
            s += ",";
        }
        s += var_to_str(x[i]);
    }
    s += "]";
    return s;
}

//static std::string var_to_str(ggml_unary_op unary_op) {
//    return ggml_unary_op_name(unary_op);
//}

static std::string var_to_str(ggml_type type) {
    return ggml_type_name(type);
}

static std::string var_to_str(ggml_op_pool pool) {
    switch (pool) {
        case GGML_OP_POOL_AVG:  return "avg";
        case GGML_OP_POOL_MAX:  return "max";
        default:                return std::to_string(pool);
    }
}

#define VARS_TO_STR1(a) VAR_TO_STR(a)
#define VARS_TO_STR2(a, b) VAR_TO_STR(a) + "," + VAR_TO_STR(b)
#define VARS_TO_STR3(a, b, c) VAR_TO_STR(a) + "," + VARS_TO_STR2(b, c)
#define VARS_TO_STR4(a, b, c, d) VAR_TO_STR(a) + "," + VARS_TO_STR3(b, c, d)
#define VARS_TO_STR5(a, b, c, d, e) VAR_TO_STR(a) + "," + VARS_TO_STR4(b, c, d, e)
#define VARS_TO_STR6(a, b, c, d, e, f) VAR_TO_STR(a) + "," + VARS_TO_STR5(b, c, d, e, f)
#define VARS_TO_STR7(a, b, c, d, e, f, g) VAR_TO_STR(a) + "," + VARS_TO_STR6(b, c, d, e, f, g)
#define VARS_TO_STR8(a, b, c, d, e, f, g, h) VAR_TO_STR(a) + "," + VARS_TO_STR7(b, c, d, e, f, g, h)
#define VARS_TO_STR9(a, b, c, d, e, f, g, h, i) VAR_TO_STR(a) + "," + VARS_TO_STR8(b, c, d, e, f, g, h, i)
#define VARS_TO_STR10(a, b, c, d, e, f, g, h, i, j) VAR_TO_STR(a) + "," + VARS_TO_STR9(b, c, d, e, f, g, h, i, j)
#define VARS_TO_STR11(a, b, c, d, e, f, g, h, i, j, k) VAR_TO_STR(a) + "," + VARS_TO_STR10(b, c, d, e, f, g, h, i, j, k)
#define VARS_TO_STR12(a, b, c, d, e, f, g, h, i, j, k, l) VAR_TO_STR(a) + "," + VARS_TO_STR11(b, c, d, e, f, g, h, i, j, k, l)

#ifdef GGML_USE_SYCL
static bool inline _isinf(float f) {
    return (*(uint32_t *)&f & 0x7fffffff) == 0x7f800000;
}
#else
static bool inline _isinf(float f) { return std::isinf(f); }
#endif

// accept FLT_MAX as infinity
static bool isinf_or_max(float f) {
    return _isinf(f) || f == FLT_MAX || f == -FLT_MAX;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

enum test_mode {
    MODE_TEST,
    MODE_PERF,
};

struct test_case {
    virtual ~test_case() {}

    virtual std::string op_desc(ggml_tensor * t) {
        return ggml_op_desc(t);
    }

    virtual std::string vars() {
        return "";
    }

    virtual ggml_tensor * build_graph(ggml_context * ctx) = 0;

    virtual double max_nmse_err() {
        return 1e-7;
    }

    virtual void initialize_tensors(ggml_context * ctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t);
        }
    }

    virtual size_t op_size(ggml_tensor * t) {
        size_t size = ggml_nbytes(t);
        // add source tensors
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (t->src[i] != NULL) {
                size += ggml_nbytes(t->src[i]);
            }
        }
        return size;
    }

    ggml_cgraph * gf = nullptr;

    static const int sentinel_size = 1024;

    test_mode mode;

    std::vector<ggml_tensor *> sentinels;

    void add_sentinel(ggml_context * ctx) {
        if (mode == MODE_PERF) {
            return;
        }
        ggml_tensor * sentinel = ::ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sentinel_size);
        ggml_format_name(sentinel, "sent_%zu", sentinels.size());
        sentinels.push_back(sentinel);
    }

    // hijack ggml_new_tensor to add sentinels after each tensor to check for overflows in the backend

    ggml_tensor * ggml_new_tensor(ggml_context * ctx, ggml_type type, int n_dims, const int64_t * ne) {
        ggml_tensor * t = ::ggml_new_tensor(ctx, type, n_dims, ne);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_1d(ggml_context * ctx, ggml_type type, int64_t ne0) {
        ggml_tensor * t = ::ggml_new_tensor_1d(ctx, type, ne0);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_2d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1) {
        ggml_tensor * t = ::ggml_new_tensor_2d(ctx, type, ne0, ne1);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_3d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
        ggml_tensor * t = ::ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
        add_sentinel(ctx);
        return t;
    }

    ggml_tensor * ggml_new_tensor_4d(ggml_context * ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        ggml_tensor * t = ::ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
        add_sentinel(ctx);
        return t;
    }

    bool eval(ggml_backend_t backend1, ggml_backend_t backend2, const char * op_name) {
        mode = MODE_TEST;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead(),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context * ctx = ggml_init(params);

        gf = ggml_new_graph(ctx);

        // pre-graph sentinel
        add_sentinel(ctx);

        ggml_tensor * out = build_graph(ctx);

        if (op_name != nullptr && op_desc(out) != op_name) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            ggml_free(ctx);
            return true;
        }

        printf("  %s(%s): ", op_desc(out).c_str(), vars().c_str());
        fflush(stdout);

        // check if the backends support the ops
        bool supported = true;
        for (ggml_backend_t backend : {backend1, backend2}) {
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
                if (!ggml_backend_supports_op(backend, t)) {
                    printf("not supported [%s] ", ggml_backend_name(backend));
                    supported = false;
                    break;
                }
            }
        }
        if (!supported) {
            printf("\n");
            ggml_free(ctx);
            return true;
        }

        // post-graph sentinel
        add_sentinel(ctx);

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend1);
        if (buf == NULL) {
            printf("failed to allocate tensors [%s] ", ggml_backend_name(backend1));
            ggml_free(ctx);
            return false;
        }

        // build graph
        ggml_build_forward_expand(gf, out);

        // add sentinels as graph nodes so that they are checked in the callback
        for (ggml_tensor * sentinel : sentinels) {
            gf->nodes[gf->n_nodes++] = sentinel;
        }

        // randomize tensors
        initialize_tensors(ctx);

        // compare
        struct callback_userdata {
            bool   ok;
            double max_err;
            ggml_backend_t backend1;
            ggml_backend_t backend2;
        };

        callback_userdata ud {
            true,
            max_nmse_err(),
            backend1,
            backend2
        };

        auto callback = [](int index, ggml_tensor * t1, ggml_tensor * t2, void * user_data) -> bool {
            callback_userdata * ud = (callback_userdata *) user_data;
            const char * bn1 = ggml_backend_name(ud->backend1);
            const char * bn2 = ggml_backend_name(ud->backend2);

            if (t1->op == GGML_OP_NONE) {
                // sentinels must be unchanged
                std::vector<uint8_t> t1_data(ggml_nbytes(t1));
                std::vector<uint8_t> t2_data(ggml_nbytes(t2));
                ggml_backend_tensor_get(t1, t1_data.data(), 0, ggml_nbytes(t1));
                ggml_backend_tensor_get(t2, t2_data.data(), 0, ggml_nbytes(t2));

                if (memcmp(t1_data.data(), t2_data.data(), ggml_nbytes(t1)) != 0) {
                    printf("sentinel mismatch: %s ", t1->name);
                    ud->ok = false;
                    return true;
                }
            }

            std::vector<float> f1 = tensor_to_float(t1);
            std::vector<float> f2 = tensor_to_float(t2);

            for (size_t i = 0; i < f1.size(); i++) {
                // check for nans
                if (std::isnan(f1[i]) || std::isnan(f2[i])) {
                    printf("[%s] NaN at index %zu (%s=%f %s=%f) ", ggml_op_desc(t1), i, bn1, f1[i], bn2, f2[i]);
                    ud->ok = false;
                    return true;
                }
                // check for infs: both must be inf of the same sign, or both must be finite
                if (isinf_or_max(f1[i]) || isinf_or_max(f2[i])) {
                    if (isinf_or_max(f1[i]) && isinf_or_max(f2[i])) {
                        if (std::signbit(f1[i]) != std::signbit(f2[i])) {
                            printf("[%s] inf sign mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                            ud->ok = false;
                            return true;
                        }
                    } else {
                        printf("[%s] inf mismatch: %s=%f %s=%f ", ggml_op_desc(t1), bn1, f1[i], bn2, f2[i]);
                        ud->ok = false;
                        return true;
                    }
                }
            }

            double err = nmse(f1.data(), f2.data(), f1.size());
            if (err > ud->max_err) {
                printf("[%s] NMSE = %.9f > %.9f ", ggml_op_desc(t1), err, ud->max_err);
                //for (int i = 0; i < (int) f1.size(); i++) {
                //    printf("%5d %9.6f %9.6f, diff = %9.6f\n", i, f1[i], f2[i], f1[i] - f2[i]);
                //}
                //printf("\n");
                //exit(1);
                ud->ok = false;
            }
            return true;

            GGML_UNUSED(index);
        };

        const bool cmp_ok = ggml_backend_compare_graph_backend(backend1, backend2, gf, callback, &ud);

        if (!cmp_ok) {
            printf("compare failed ");
        }

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);

        if (ud.ok && cmp_ok) {
            printf("\033[1;32mOK\033[0m\n");
            return true;
        }

        printf("\033[1;31mFAIL\033[0m\n");
        return false;
    }

    bool eval_perf(ggml_backend_t backend, const char * op_name) {
        mode = MODE_PERF;

        static const size_t graph_nodes = 8192;

        ggml_init_params params = {
            /* .mem_size = */ ggml_tensor_overhead()*128 + ggml_graph_overhead_custom(graph_nodes, false),
            /* .mem_base = */ NULL,
            /* .no_alloc = */ true,
        };
        ggml_context * ctx = ggml_init(params);

        ggml_tensor * out = build_graph(ctx);

        if (op_name != nullptr && op_desc(out) != op_name) {
            //printf("  %s: skipping\n", op_desc(out).c_str());
            ggml_free(ctx);
            return true;
        }

        int len = printf("  %s(%s): ", op_desc(out).c_str(), vars().c_str());
        fflush(stdout);

        // check if backends support op
        if (!ggml_backend_supports_op(backend, out)) {
            printf("not supported\n");
            ggml_free(ctx);
            return true;
        }

        // align while also leaving some margin for variations in parameters
        int align = 20;
        int last = (len + align - 1) / align * align;
        if (last - len < 5) {
            last += align;
        }
        last = std::max(last, 60);
        printf("%*s", last - len, "");

        // allocate
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buf == NULL) {
            printf("failed to allocate tensors\n");
            ggml_free(ctx);
            return false;
        }

        // randomize tensors
        initialize_tensors(ctx);

        // build graph
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_nodes, false);
        ggml_build_forward_expand(gf, out);

        // warmup run
        ggml_backend_graph_compute(backend, gf);

        // duplicate the op
        size_t target_size = ggml_backend_is_cpu(backend) ? 1ULL << 33 : 1ULL << 35; // 8 GB CPU, 32 GB GPU
        int n_runs = std::min((size_t)gf->size - gf->n_nodes, target_size / op_size(out)) + 1;
        for (int i = 1; i < n_runs; i++) {
            gf->nodes[gf->n_nodes++] = out;
        }

        // calculate memory
        size_t mem = n_runs * op_size(out);
        auto tensor_op_size = [](ggml_tensor * t) {
            size_t size = ggml_nbytes(t);
            // add source tensors
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (t->src[i] != NULL) {
                    size += ggml_nbytes(t->src[i]);
                }
            }
            return size;
        };
        for (int i = 0; i < gf->n_nodes; i++) {
            if (ggml_is_view_op(gf->nodes[i]->op) || gf->nodes[i] == out) {
                continue;
            }
            mem += tensor_op_size(gf->nodes[i]);
        }

        // run
        ggml_backend_synchronize(backend);

        int64_t start_time = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        int64_t end_time = ggml_time_us();
        double time_us = end_time - start_time;

        printf("    %5d runs - %8.2f us/run - %8zu kB/run - \033[1;34m%7.2f GB/s\033[0m\n",
            n_runs,
            time_us / n_runs,
            op_size(out) / 1024,
            mem / (time_us/1e6) / 1024.0 / 1024.0 / 1024.0);

        ggml_backend_buffer_free(buf);

        ggml_free(ctx);

        return true;
    }
};

// GGML_OP_UNARY
struct test_unary : public test_case {
    const ggml_unary_op op;
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int v; // view (1 : non-contiguous a)

    std::string vars() override {
        return VARS_TO_STR3(type, ne_a, v);
    }

    test_unary(ggml_unary_op op,
            ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {128, 10, 10, 10},
            int v = 0)
        : op(op), type(type), ne_a(ne_a), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        }
        ggml_tensor * out = ggml_unary(ctx, a, op);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            // test extended range of values to check for NaNs in GELU
            init_tensor_uniform(t, -150.f, 150.f);
        }
    }
};

// GGML_OP_GET_ROWS
struct test_get_rows : public test_case {
    const ggml_type type;
    const int n; // cols
    const int m; // rows
    const int r; // rows to get
    const int b; // batch size
    const bool v; // view (non-contiguous src1)

    std::string vars() override {
        return VARS_TO_STR6(type, n, m, r, b, v);
    }

    test_get_rows(ggml_type type = GGML_TYPE_F32, int n = 10, int m = 5, int r = 3, int b = 1, bool v = false)
        : type(type), n(n), m(m), r(r), b(b), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * in = ggml_new_tensor_3d(ctx, type, n, m, b);
        ggml_tensor * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, r, b);
        if (v) {
            rows = ggml_view_2d(ctx, rows, r/2, b, rows->nb[1], 0);
        }
        ggml_tensor * out = ggml_get_rows(ctx, in, rows);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                // rows
                std::vector<int> data(r*b);
                for (int i = 0; i < r*b; i++) {
                    data[i] = rand() % m;
                }
                ggml_backend_tensor_set(t, data.data(), 0, r * b * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_REPEAT
struct test_repeat : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4> nr;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, nr);
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) * 2;
    }

    test_repeat(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            std::array<int, 4> nr = {2, 2, 2, 2})
        : type(type), ne(ne), nr(nr) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * target = ggml_new_tensor_4d(ctx, type, ne[0]*nr[0], ne[1]*nr[1], ne[2]*nr[2], ne[3]*nr[3]);
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_repeat(ctx, src, target);
        return out;
    }
};

// GGML_OP_DUP
struct test_dup : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> permute;
    bool _use_permute;

    std::string vars() override {
        std::string v = VARS_TO_STR2(type, ne);
        if (_use_permute) v += "," + VAR_TO_STR(permute);
        return v;
    }

    test_dup(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 20, 1},
            std::array<int64_t, 4> permute = {0, 0, 0, 0})
        : type(type), ne(ne), permute(permute),
            _use_permute(permute[0] + permute[1] + permute[2] + permute[3] > 0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        if (_use_permute) {
            src = ggml_permute(ctx, src, permute[0], permute[1], permute[2], permute[3]);
        }
        ggml_tensor * out = ggml_dup(ctx, src);
        return out;
    }
};

// GGML_OP_CPY
struct test_cpy : public test_case {
    const ggml_type type_src;
    const ggml_type type_dst;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> permute;
    bool _src_use_permute;

    std::string vars() override {
        return VARS_TO_STR4(type_src, type_dst, ne, permute);
    }

    double max_nmse_err() override {
        if (type_src == type_dst) {
            return 0.0;
        }
        if (type_dst == GGML_TYPE_IQ4_NL) {
            // GPU quantizer uses single-pass fitting; CPU does iterative
            // gradient descent refinement. Different d values are expected.
            return 0.005;
        }
        if (type_dst == GGML_TYPE_Q4_0 || type_dst == GGML_TYPE_Q4_1 ||
            type_dst == GGML_TYPE_Q5_0 || type_dst == GGML_TYPE_Q5_1 || type_dst == GGML_TYPE_Q8_0) {
            double err_estimate = 1.0f/8.0f * 150.0f;
            if (type_dst == GGML_TYPE_Q5_0 || type_dst == GGML_TYPE_Q5_1) {
                err_estimate /= 2.0f;
            }
            if (type_dst == GGML_TYPE_Q8_0) {
                err_estimate /= 8.0f;
            }
            err_estimate *= err_estimate;
            err_estimate /= (150.0f*150.0f*0.25f)*float(ne[0] * ne[1] * ne[2] * ne[3]);
            return err_estimate;
        }
        return 1e-6;
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) + ggml_nbytes(t->src[0]);
    }

    test_cpy(ggml_type type_src = GGML_TYPE_F32, ggml_type type_dst = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 1},
            std::array<int64_t, 4> permute = {0, 0, 0, 0})
        : type_src(type_src), type_dst(type_dst), ne(ne), permute(permute),
          _src_use_permute(permute[0] + permute[1] + permute[2] + permute[3] > 0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type_src, 4, ne.data());
        if (_src_use_permute) {
            src = ggml_permute(ctx, src, permute[0], permute[1], permute[2], permute[3]);
        }
        ggml_tensor* dst = ggml_new_tensor(ctx, type_dst, 4, src->ne);
        ggml_tensor * out = ggml_cpy(ctx, src, dst);
        return out;
    }
};

// GGML_OP_CONT
struct test_cont : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_cont(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 1})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor(ctx, type, 4, ne.data());
        src = ggml_transpose(ctx, src);
        ggml_tensor * out = ggml_cont(ctx, src);

        return out;
    }
};

// GGML_OP_ADD
// GGML_OP_MUL
// GGML_OP_DIV
struct test_bin_bcast : public test_case {
    using op_t = ggml_tensor * (*) (ggml_context *, ggml_tensor *, ggml_tensor *);
    op_t op;
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int, 4> nr;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, nr);
    }

    size_t op_size(ggml_tensor * t) override {
        return ggml_nbytes(t) * 3;
    }

    test_bin_bcast(op_t op, ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 1, 1},
            std::array<int, 4> nr = {1, 2, 1, 1})
        : op(op), type(type), ne(ne), nr(nr) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type, ne[0]*nr[0], ne[1]*nr[1], ne[2]*nr[2], ne[3]*nr[3]);
        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = op(ctx, a, b);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (op == ggml_div) {
                // avoid division by zero
                init_tensor_uniform(t, 1.0f, 2.0f);
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_SCALE
struct test_scale : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float scale;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, scale);
    }

    test_scale(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            float scale = 2.0f)
        : type(type), ne(ne), scale(scale) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_scale(ctx, a, scale);
        return out;
    }
};

// GGML_OP_NORM
struct test_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float eps;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, eps);
    }

    test_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 10, 10, 10},
            float eps = 1e-6f)
        : type(type), ne(ne), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_norm(ctx, a, eps);
        return out;
    }
};

// GGML_OP_RMS_NORM
struct test_rms_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float eps;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, eps);
    }

    test_rms_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 10, 10, 10},
            float eps = 1e-6f)
        : type(type), ne(ne), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_rms_norm(ctx, a, eps);
        return out;
    }
};

// GGML_OP_SSM_CONV — ik 4-arg form: ggml_ssm_conv(s, x, c, sq)
//
// Single-sequence and multi-sequence cases. mode controls how the seq map
// is built:
//   0 = unique (each token a different seq id, round-robin)
//   1 = recurrent (all tokens to seq 0 — exercises slow-path self-read)
//   2 = fanout (token 0 writes state to all seqs via sq[1..])
struct test_ssm_conv : public test_case {
    const int64_t d_conv;
    const int64_t d_inner;
    const int64_t n_t;
    const int64_t n_kv;
    const int64_t sq_ne0;
    const int     mode;

    std::string vars() override {
        return VARS_TO_STR6(d_conv, d_inner, n_t, n_kv, sq_ne0, mode);
    }

    test_ssm_conv(int64_t d_conv = 4, int64_t d_inner = 4096,
                  int64_t n_t = 1, int64_t n_kv = 1,
                  int64_t sq_ne0 = 1, int mode = 0)
        : d_conv(d_conv), d_inner(d_inner), n_t(n_t),
          n_kv(n_kv), sq_ne0(sq_ne0), mode(mode) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * s  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_conv - 1, d_inner, n_kv);
        ggml_tensor * x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_inner, n_t);
        ggml_tensor * c  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, d_inner);
        ggml_tensor * sq = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, sq_ne0, n_t);
        ggml_set_name(s,  "s");
        ggml_set_name(x,  "x");
        ggml_set_name(c,  "c");
        ggml_set_name(sq, "sq");
        ggml_tensor * out = ggml_ssm_conv(ctx, s, x, c, sq);
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                std::vector<int32_t> data(ggml_nelements(t), -1);  // sentinel
                for (int64_t it = 0; it < n_t; ++it) {
                    int32_t * row = data.data() + it * sq_ne0;
                    if (mode == 0) {
                        // unique: each token a different seq, round-robin
                        row[0] = (int32_t)(it % n_kv);
                    } else if (mode == 1) {
                        // recurrent: all tokens write to seq 0
                        row[0] = 0;
                    } else if (mode == 2) {
                        // fanout: token 0 writes state to all seqs via sq[1..]
                        row[0] = 0;
                        if (it == 0) {
                            for (int64_t k = 1; k < std::min<int64_t>(n_kv, sq_ne0); ++k) {
                                row[k] = (int32_t)k;
                            }
                        }
                    }
                }
                ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(int32_t));
            } else {
                init_tensor_uniform(t, -1.0f, 1.0f);
            }
        }
    }
};

// GGML_OP_DELTA_NET — recurrent linear-attention used by Qwen3-Next /
// Qwen3.5-A3B. Mirrors the model's tensor construction in
// llama-delta-net.cpp::build_fused_delta_net so that the data layout
// (in particular g and beta, which the CPU formula reads in pre-permute
// order) matches what the production op expects.
struct test_delta_net : public test_case {
    const int64_t head_dim;     // S_k == S_v
    const int64_t n_tokens;
    const int64_t H_v;          // n_v_heads
    const int64_t gqa_ratio;    // H_v / H_k
    const int64_t n_seqs;
    const int     repeat_type;  // 0 = tiled, 1 = interleaved

    std::string vars() override {
        return VARS_TO_STR6(head_dim, n_tokens, H_v, gqa_ratio, n_seqs, repeat_type);
    }

    test_delta_net(int64_t head_dim = 128, int64_t n_tokens = 1, int64_t H_v = 32,
                   int64_t gqa_ratio = 4, int64_t n_seqs = 1, int repeat_type = 0)
        : head_dim(head_dim), n_tokens(n_tokens), H_v(H_v),
          gqa_ratio(gqa_ratio), n_seqs(n_seqs), repeat_type(repeat_type) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t H_k = H_v / gqa_ratio;
        GGML_ASSERT(H_k * gqa_ratio == H_v);

        // q, k: contiguous [head_dim, n_tokens, H_k, n_seqs]. The model
        // produces these via permute+l2_norm; here we construct them
        // directly since the access formula is the same for both layouts.
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_tokens, H_k, n_seqs);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, n_tokens, H_k, n_seqs);
        ggml_set_name(q, "q");
        ggml_set_name(k, "k");

        // v: model permutes from [head_dim, H_v, n_tokens, n_seqs] →
        // [head_dim, n_tokens, H_v, n_seqs]. Mirror that pattern so the
        // strides exercise the permuted layout in the dispatch.
        ggml_tensor * v_pre = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, H_v, n_tokens, n_seqs);
        ggml_set_name(v_pre, "v_pre");
        ggml_tensor * v = ggml_permute(ctx, v_pre, 0, 2, 1, 3);
        ggml_set_name(v, "v");

        // g: pre-permute shape [H_v, n_tokens, n_seqs] (contig), permute to
        // [n_tokens, 1, H_v, n_seqs]. The CPU access formula reads the
        // pre-permute memory order so the underlying data must live there.
        ggml_tensor * g_pre = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H_v, n_tokens, n_seqs);
        ggml_set_name(g_pre, "g_pre");
        ggml_tensor * g = ggml_permute(ctx, g_pre, 2, 0, 3, 1);
        ggml_set_name(g, "g");

        // beta: pre-permute shape [H_v, 1, n_tokens, n_seqs] contig,
        // permute(2, 0, 1, 3) → [1, n_tokens, H_v, n_seqs]. Same memory
        // layout principle as g.
        ggml_tensor * beta_pre = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, H_v, 1, n_tokens, n_seqs);
        ggml_set_name(beta_pre, "beta_pre");
        ggml_tensor * beta = ggml_permute(ctx, beta_pre, 2, 0, 1, 3);
        ggml_set_name(beta, "beta");

        // state: contiguous [head_dim, head_dim*H_v, 1, n_seqs] (matches the
        // model's reshape after the per-layer state buffer split).
        ggml_tensor * state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, head_dim * H_v, 1, n_seqs);
        ggml_set_name(state, "state");

        ggml_tensor * out = ggml_delta_net(ctx, q, k, v, g, beta, state);
        out->op_params[0] = repeat_type;
        ggml_set_name(out, "out");
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            // Use a small range for stable recurrent dynamics. The state
            // recurrence amplifies large inputs (decay = exp(g)) — keep g
            // strictly negative so decay stays bounded.
            if (std::string(t->name) == "g_pre") {
                init_tensor_uniform(t, -2.0f, -0.1f);
            } else if (std::string(t->name) == "beta_pre") {
                init_tensor_uniform(t, -1.0f, 1.0f);
            } else if (std::string(t->name) == "state") {
                init_tensor_uniform(t, -0.1f, 0.1f);
            } else {
                init_tensor_uniform(t, -0.5f, 0.5f);
            }
        }
    }

    double max_nmse_err() override {
        // The recurrent state update accumulates rounding error over n_tokens
        // iterations. Loosen the tolerance for longer sequences.
        return n_tokens > 16 ? 1e-3 : 5e-5;
    }
};

// GGML_OP_L2_NORM
struct test_l2_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const float eps;
    const bool v;  // when true, take a strided view of the input first (tests
                   // the non-contiguous path used by Qwen3.5 q_fused/k_fused)

    std::string vars() override {
        return VARS_TO_STR4(type, ne, eps, v);
    }

    test_l2_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 64, 320, 1},
            float eps = 1e-12f,
            bool v = false)
        : type(type), ne(ne), eps(eps), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_set_name(a, "a");
        if (v) {
            a = ggml_view_4d(ctx, a, a->ne[0]/2, a->ne[1]/2, a->ne[2]/2, a->ne[3]/2,
                             a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view of a");
        }
        ggml_tensor * out = ggml_l2_norm(ctx, a, eps);
        ggml_set_name(out, "out");
        return out;
    }
};

// GGML_OP_FUSED_RMS_NORM
struct test_fused_rms_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float eps;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, eps);
    }

    test_fused_rms_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 10, 10, 10},
            float eps = 1e-6f)
        : type(type), ne(ne), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[0]);
        ggml_tensor * out = ggml_fused_rms_norm(ctx, a, b, eps);
        return out;
    }
};

// GGML_OP_FUSED_MUL_UNARY
struct test_fused_mul_unary : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    ggml_unary_op uop;
    bool scalar_bcast;  // when true, src0 (the unary input) is a single scalar
                        // broadcast over src1 (used by MoE shared-expert gating)

    std::string vars() override {
        return VARS_TO_STR4(type, ne, uop, scalar_bcast);
    }

    test_fused_mul_unary(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {128, 10, 10, 1},
            ggml_unary_op uop = GGML_UNARY_OP_SILU,
            bool scalar_bcast = false)
        : type(type), ne(ne), uop(uop), scalar_bcast(scalar_bcast) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t one[4] = {1, 1, 1, 1};
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, scalar_bcast ? one : ne.data());
        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_fused_mul_unary(ctx, a, b, uop);
        return out;
    }
};

// GGML_OP_MUL_MAT
struct test_mul_mat : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const std::array<int64_t, 2> bs; // dims 3 and 4
    const std::array<int64_t, 2> nr; // repeat in dims 3 and 4

    std::string vars() override {
        return VARS_TO_STR7(type_a, type_b, m, n, k, bs, nr);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    size_t op_size(ggml_tensor * t) override {
        size_t a = ggml_nbytes(t->src[0]) * n * nr[0] * nr[1];
        size_t b = ggml_nbytes(t->src[1]) * m;
        size_t c  = ggml_nbytes(t);
        return a + b + c;

        GGML_UNUSED(t);
    }

    test_mul_mat(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int64_t m = 32, int64_t n = 32, int64_t k = 32,
            std::array<int64_t, 2> bs = {10, 10},
            std::array<int64_t, 2> nr = {2, 2})
        : type_a(type_a), type_b(type_b), m(m), n(n), k(k), bs(bs), nr(nr) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor * a = ggml_new_tensor_4d(ctx, type_a, k, m, bs[0]      , bs[1]);
        ggml_tensor * b = ggml_new_tensor_4d(ctx, type_b, k, n, bs[0]*nr[0], bs[1]*nr[1]);
        ggml_tensor * out = ggml_mul_mat(ctx, a, b);
        return out;
    }
};

// Stress variant of test_mul_mat: uses a wider B-vector (activation) range
// to catch f16 accumulator overflow on Vega's f16acc mul_mat_vec path.
// Real model activations after ~40 residual layers are 10-50× larger than
// the default [-1, 1] test data. Phase 20c's f16acc (FLOAT_TYPE=float16_t)
// overflows at these magnitudes; the f32 path does not.
struct test_mul_mat_stress : public test_mul_mat {
    float b_range;

    test_mul_mat_stress(ggml_type type_a, ggml_type type_b,
                        int64_t m, int64_t n, int64_t k, float b_range)
        : test_mul_mat(type_a, type_b, m, n, k, {1, 1}, {1, 1}),
          b_range(b_range) {}

    std::string vars() override {
        return VARS_TO_STR7(type_a, type_b, m, n, k, bs, nr) +
               ",b_range=" + std::to_string((int)b_range);
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16) {
                // B tensor (activations) — use wide range
                init_tensor_uniform(t, -b_range, b_range);
            } else {
                // A tensor (quantized weights) — standard range
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_MUL_MAT_ID
struct test_mul_mat_id : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int n_mats;
    const int n_used;
    const bool b; // brodcast b matrix
    const int64_t m;
    const int64_t n;
    const int64_t k;

    std::string vars() override {
        return VARS_TO_STR8(type_a, type_b, n_mats, n_used, b, m, n, k);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    size_t op_size(ggml_tensor * t) override {
        size_t a = ggml_nbytes(t->src[2]) * n;
        size_t b = ggml_nbytes(t->src[1]) * m;
        size_t c  = ggml_nbytes(t);
        return a + b + c;

        GGML_UNUSED(t);
    }

    test_mul_mat_id(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int n_mats = 8, int n_used = 2, bool b = false,
            int64_t m = 32, int64_t n = 32, int64_t k = 32)
        : type_a(type_a), type_b(type_b), n_mats(n_mats), n_used(n_used), b(b),
            m(m), n(n), k(k) {
            GGML_ASSERT(n_used <= n_mats);
        }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // C^T = A * B^T: (k, m) * (k, n) => (m, n)
        ggml_tensor * as = ggml_new_tensor_3d(ctx, type_a, k, m, n_mats);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_mats, n);
        if (n_used != n_mats) {
            ids = ggml_view_2d(ctx, ids, n_used, n, ids->nb[1], 0);
        }
        ggml_tensor * b = ggml_new_tensor_3d(ctx, type_b, k, this->b ? 1 : n_used, n);
        ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                if (ggml_is_view_op(t->op)) { continue; }
                // ids
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<int32_t> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i % n_mats;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
                }
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// GGML_OP_FUSED_UP_GATE
// CPU backend ABORTs on this op, so we can't use the standard compare-with-CPU approach.
// Instead: build decomposed graph (mul_mat + mul_mat + fused_mul_unary) on CPU as reference,
// build fused graph on Vulkan, compare outputs.
struct test_fused_up_gate {
    const ggml_type type_a;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const ggml_unary_op op;

    test_fused_up_gate(ggml_type type_a = GGML_TYPE_Q8_0,
            int64_t m = 32, int64_t n = 2, int64_t k = 32,
            ggml_unary_op op = GGML_UNARY_OP_SILU)
        : type_a(type_a), m(m), n(n), k(k), op(op) {}

    bool eval(ggml_backend_t backend_vk, ggml_backend_t backend_cpu) {
        printf("  FUSED_UP_GATE(type_a=%s,m=%lld,n=%lld,k=%lld,op=%d): ",
               ggml_type_name(type_a), (long long)m, (long long)n, (long long)k, (int)op);
        fflush(stdout);

        // --- Build reference graph (decomposed, runs on CPU) ---
        ggml_init_params params_ref = {
            ggml_tensor_overhead()*32 + ggml_graph_overhead(), NULL, true
        };
        ggml_context * ctx_ref = ggml_init(params_ref);
        ggml_cgraph * gf_ref = ggml_new_graph(ctx_ref);

        ggml_tensor * up_ref   = ggml_new_tensor_2d(ctx_ref, type_a, k, m);
        ggml_tensor * gate_ref = ggml_new_tensor_2d(ctx_ref, type_a, k, m);
        ggml_tensor * b_ref    = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_F32, k, n);
        ggml_set_name(up_ref, "up"); ggml_set_name(gate_ref, "gate"); ggml_set_name(b_ref, "b");

        // Decomposed: mul_mat(up, b) * activation(mul_mat(gate, b))
        ggml_tensor * up_out   = ggml_mul_mat(ctx_ref, up_ref, b_ref);
        ggml_tensor * gate_out = ggml_mul_mat(ctx_ref, gate_ref, b_ref);
        ggml_tensor * ref_out  = ggml_fused_mul_unary(ctx_ref, gate_out, up_out, op);
        ggml_build_forward_expand(gf_ref, ref_out);

        ggml_backend_buffer_t buf_ref = ggml_backend_alloc_ctx_tensors(ctx_ref, backend_cpu);
        if (!buf_ref) { printf("alloc fail (cpu)\n"); ggml_free(ctx_ref); return false; }

        // --- Build fused graph (runs on Vulkan) ---
        ggml_init_params params_vk = {
            ggml_tensor_overhead()*32 + ggml_graph_overhead(), NULL, true
        };
        ggml_context * ctx_vk = ggml_init(params_vk);
        ggml_cgraph * gf_vk = ggml_new_graph(ctx_vk);

        ggml_tensor * up_vk   = ggml_new_tensor_2d(ctx_vk, type_a, k, m);
        ggml_tensor * gate_vk = ggml_new_tensor_2d(ctx_vk, type_a, k, m);
        ggml_tensor * b_vk    = ggml_new_tensor_2d(ctx_vk, GGML_TYPE_F32, k, n);
        ggml_set_name(up_vk, "up"); ggml_set_name(gate_vk, "gate"); ggml_set_name(b_vk, "b");

        ggml_tensor * vk_out = ggml_fused_up_gate(ctx_vk, up_vk, gate_vk, b_vk, op);

        // Check that Vulkan supports this op
        if (!ggml_backend_supports_op(backend_vk, vk_out)) {
            printf("not supported [%s]\n", ggml_backend_name(backend_vk));
            ggml_free(ctx_vk); ggml_backend_buffer_free(buf_ref); ggml_free(ctx_ref);
            return true; // skip, not fail
        }

        ggml_build_forward_expand(gf_vk, vk_out);

        ggml_backend_buffer_t buf_vk = ggml_backend_alloc_ctx_tensors(ctx_vk, backend_vk);
        if (!buf_vk) { printf("alloc fail (vk)\n"); ggml_free(ctx_vk); ggml_backend_buffer_free(buf_ref); ggml_free(ctx_ref); return false; }

        // --- Initialize both with identical random data ---
        size_t up_bytes   = ggml_nbytes(up_ref);
        size_t gate_bytes = ggml_nbytes(gate_ref);
        size_t b_bytes    = ggml_nbytes(b_ref);

        std::vector<uint8_t> up_data(up_bytes), gate_data(gate_bytes);
        std::vector<float> b_data(n * k);

        // Generate random quantized data
        {
            std::vector<float> tmp(k * m);
            std::default_random_engine rng(42);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto & v : tmp) v = dist(rng);
            ggml_quantize_chunk(type_a, tmp.data(), up_data.data(), 0, m, k, nullptr);
            for (auto & v : tmp) v = dist(rng);
            ggml_quantize_chunk(type_a, tmp.data(), gate_data.data(), 0, m, k, nullptr);
            for (auto & v : b_data) v = dist(rng);
        }

        // Set tensors on both backends
        ggml_backend_tensor_set(up_ref,   up_data.data(),   0, up_bytes);
        ggml_backend_tensor_set(gate_ref, gate_data.data(), 0, gate_bytes);
        ggml_backend_tensor_set(b_ref,    b_data.data(),    0, b_bytes);

        ggml_backend_tensor_set(up_vk,   up_data.data(),   0, up_bytes);
        ggml_backend_tensor_set(gate_vk, gate_data.data(), 0, gate_bytes);
        ggml_backend_tensor_set(b_vk,    b_data.data(),    0, b_bytes);

        // --- Compute both ---
        ggml_backend_graph_compute(backend_cpu, gf_ref);
        ggml_backend_graph_compute(backend_vk,  gf_vk);
        ggml_backend_synchronize(backend_vk);

        // --- Compare outputs ---
        size_t nelements = ggml_nelements(ref_out);
        std::vector<float> f_ref(nelements), f_vk(nelements);
        ggml_backend_tensor_get(ref_out, f_ref.data(), 0, nelements * sizeof(float));
        ggml_backend_tensor_get(vk_out,  f_vk.data(),  0, nelements * sizeof(float));

        // NMSE comparison
        double sum_diff2 = 0, sum_ref2 = 0;
        for (size_t i = 0; i < nelements; i++) {
            if (std::isnan(f_ref[i]) || std::isnan(f_vk[i])) {
                printf("NaN at %zu (ref=%f vk=%f) ", i, f_ref[i], f_vk[i]);
                printf("\033[1;31mFAIL\033[0m\n");
                ggml_backend_buffer_free(buf_vk); ggml_free(ctx_vk);
                ggml_backend_buffer_free(buf_ref); ggml_free(ctx_ref);
                return false;
            }
            double d = (double)f_ref[i] - (double)f_vk[i];
            sum_diff2 += d * d;
            sum_ref2  += (double)f_ref[i] * (double)f_ref[i];
        }
        double nmse = (sum_ref2 > 0) ? sum_diff2 / sum_ref2 : sum_diff2;
        double threshold = 5e-4;

        ggml_backend_buffer_free(buf_vk); ggml_free(ctx_vk);
        ggml_backend_buffer_free(buf_ref); ggml_free(ctx_ref);

        if (nmse > threshold) {
            printf("NMSE = %.9f > %.9f ", nmse, threshold);
            printf("\033[1;31mFAIL\033[0m\n");
            return false;
        }

        printf("\033[1;32mOK\033[0m (NMSE=%.2e)\n", nmse);
        return true;
    }
};

// GGML_OP_MOE_FUSED_UP_GATE
// MoE variant of FUSED_UP_GATE: per-token expert routing via i32 ids tensor.
// The CPU backend supports this op directly, so we can use it as the reference
// (compare-with-CPU). The Vulkan backend dispatches the new MUL_MAT_ID
// fused_up_gate shader.
struct test_moe_fused_up_gate {
    const ggml_type type_a;
    const int64_t   k;            // hidden dim
    const int64_t   m;            // n_ff per expert
    const int64_t   n_experts;    // total experts
    const int64_t   n_expert_used; // active per token
    const int64_t   n_tokens;
    const ggml_unary_op op;

    test_moe_fused_up_gate(ggml_type type_a = GGML_TYPE_Q8_0,
            int64_t k = 32, int64_t m = 32,
            int64_t n_experts = 4, int64_t n_expert_used = 2,
            int64_t n_tokens = 4,
            ggml_unary_op op = GGML_UNARY_OP_SILU)
        : type_a(type_a), k(k), m(m), n_experts(n_experts),
          n_expert_used(n_expert_used), n_tokens(n_tokens), op(op) {}

    bool eval(ggml_backend_t backend_vk, ggml_backend_t backend_cpu) {
        printf("  MOE_FUSED_UP_GATE(type_a=%s,k=%lld,m=%lld,n_exp=%lld,n_used=%lld,n_tok=%lld,op=%d): ",
               ggml_type_name(type_a), (long long)k, (long long)m,
               (long long)n_experts, (long long)n_expert_used, (long long)n_tokens, (int)op);
        fflush(stdout);

        // --- Build CPU reference: ggml_moe_up_gate (CPU implements this op directly) ---
        ggml_init_params params_ref = {
            ggml_tensor_overhead()*32 + ggml_graph_overhead(), NULL, true
        };
        ggml_context * ctx_ref = ggml_init(params_ref);
        ggml_cgraph * gf_ref = ggml_new_graph(ctx_ref);

        // b is [k, 1, n_tokens] (the standard MoE convention — n_tokens lives in
        // ne[2] so that ids[n_expert_used, n_tokens] can route per-token).
        ggml_tensor * up_ref   = ggml_new_tensor_3d(ctx_ref, type_a, k, m, n_experts);
        ggml_tensor * gate_ref = ggml_new_tensor_3d(ctx_ref, type_a, k, m, n_experts);
        ggml_tensor * b_ref    = ggml_new_tensor_3d(ctx_ref, GGML_TYPE_F32, k, 1, n_tokens);
        ggml_tensor * ids_ref  = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_I32, n_expert_used, n_tokens);
        ggml_set_name(up_ref, "up_ref");
        ggml_set_name(gate_ref, "gate_ref");
        ggml_set_name(b_ref, "b_ref");
        ggml_set_name(ids_ref, "ids_ref");

        ggml_tensor * ref_out = ggml_moe_up_gate(ctx_ref, up_ref, gate_ref, b_ref, ids_ref, op);
        ggml_build_forward_expand(gf_ref, ref_out);

        ggml_backend_buffer_t buf_ref = ggml_backend_alloc_ctx_tensors(ctx_ref, backend_cpu);
        if (!buf_ref) { printf("alloc fail (cpu)\n"); ggml_free(ctx_ref); return false; }

        // --- Build Vulkan graph (same op) ---
        ggml_init_params params_vk = {
            ggml_tensor_overhead()*32 + ggml_graph_overhead(), NULL, true
        };
        ggml_context * ctx_vk = ggml_init(params_vk);
        ggml_cgraph * gf_vk = ggml_new_graph(ctx_vk);

        ggml_tensor * up_vk   = ggml_new_tensor_3d(ctx_vk, type_a, k, m, n_experts);
        ggml_tensor * gate_vk = ggml_new_tensor_3d(ctx_vk, type_a, k, m, n_experts);
        ggml_tensor * b_vk    = ggml_new_tensor_3d(ctx_vk, GGML_TYPE_F32, k, 1, n_tokens);
        ggml_tensor * ids_vk  = ggml_new_tensor_2d(ctx_vk, GGML_TYPE_I32, n_expert_used, n_tokens);

        ggml_tensor * vk_out = ggml_moe_up_gate(ctx_vk, up_vk, gate_vk, b_vk, ids_vk, op);

        if (!ggml_backend_supports_op(backend_vk, vk_out)) {
            printf("not supported [%s]\n", ggml_backend_name(backend_vk));
            ggml_free(ctx_vk); ggml_backend_buffer_free(buf_ref); ggml_free(ctx_ref);
            return true; // skip, not fail
        }

        ggml_build_forward_expand(gf_vk, vk_out);

        ggml_backend_buffer_t buf_vk = ggml_backend_alloc_ctx_tensors(ctx_vk, backend_vk);
        if (!buf_vk) { printf("alloc fail (vk)\n"); ggml_free(ctx_vk); ggml_backend_buffer_free(buf_ref); ggml_free(ctx_ref); return false; }

        // --- Initialize both with identical random data ---
        const size_t up_bytes   = ggml_nbytes(up_ref);
        const size_t gate_bytes = ggml_nbytes(gate_ref);
        const size_t b_bytes    = ggml_nbytes(b_ref);
        const size_t ids_bytes  = ggml_nbytes(ids_ref);

        std::vector<uint8_t> up_data(up_bytes), gate_data(gate_bytes);
        std::vector<float>   b_data(k * n_tokens);
        std::vector<int32_t> ids_data(n_expert_used * n_tokens);

        {
            std::vector<float> tmp(k * m * n_experts);
            std::default_random_engine rng(1337);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

            for (auto & v : tmp) v = dist(rng);
            ggml_quantize_chunk(type_a, tmp.data(), up_data.data(), 0, m * n_experts, k, nullptr);

            for (auto & v : tmp) v = dist(rng);
            ggml_quantize_chunk(type_a, tmp.data(), gate_data.data(), 0, m * n_experts, k, nullptr);

            for (auto & v : b_data) v = dist(rng);

            // Random expert IDs in [0, n_experts), no duplicates per token (matches
            // real top-k routing where each token picks distinct experts).
            std::vector<int32_t> all_experts(n_experts);
            std::iota(all_experts.begin(), all_experts.end(), 0);
            for (int t = 0; t < n_tokens; t++) {
                std::shuffle(all_experts.begin(), all_experts.end(), rng);
                for (int e = 0; e < n_expert_used; e++) {
                    ids_data[t * n_expert_used + e] = all_experts[e];
                }
            }
        }

        ggml_backend_tensor_set(up_ref,   up_data.data(),   0, up_bytes);
        ggml_backend_tensor_set(gate_ref, gate_data.data(), 0, gate_bytes);
        ggml_backend_tensor_set(b_ref,    b_data.data(),    0, b_bytes);
        ggml_backend_tensor_set(ids_ref,  ids_data.data(),  0, ids_bytes);

        ggml_backend_tensor_set(up_vk,   up_data.data(),   0, up_bytes);
        ggml_backend_tensor_set(gate_vk, gate_data.data(), 0, gate_bytes);
        ggml_backend_tensor_set(b_vk,    b_data.data(),    0, b_bytes);
        ggml_backend_tensor_set(ids_vk,  ids_data.data(),  0, ids_bytes);

        ggml_backend_graph_compute(backend_cpu, gf_ref);
        ggml_backend_graph_compute(backend_vk,  gf_vk);
        ggml_backend_synchronize(backend_vk);

        // --- Compare outputs (NMSE) ---
        const size_t nelements = ggml_nelements(ref_out);
        std::vector<float> f_ref(nelements), f_vk(nelements);
        ggml_backend_tensor_get(ref_out, f_ref.data(), 0, nelements * sizeof(float));
        ggml_backend_tensor_get(vk_out,  f_vk.data(),  0, nelements * sizeof(float));

        double sum_diff2 = 0, sum_ref2 = 0;
        for (size_t i = 0; i < nelements; i++) {
            if (std::isnan(f_ref[i]) || std::isnan(f_vk[i])) {
                printf("NaN at %zu (ref=%f vk=%f) ", i, f_ref[i], f_vk[i]);
                printf("\033[1;31mFAIL\033[0m\n");
                ggml_backend_buffer_free(buf_vk); ggml_free(ctx_vk);
                ggml_backend_buffer_free(buf_ref); ggml_free(ctx_ref);
                return false;
            }
            const double d = (double)f_ref[i] - (double)f_vk[i];
            sum_diff2 += d * d;
            sum_ref2  += (double)f_ref[i] * (double)f_ref[i];
        }
        const double nmse = (sum_ref2 > 0) ? sum_diff2 / sum_ref2 : sum_diff2;
        const double threshold = 5e-4;

        ggml_backend_buffer_free(buf_vk); ggml_free(ctx_vk);
        ggml_backend_buffer_free(buf_ref); ggml_free(ctx_ref);

        if (nmse > threshold) {
            printf("NMSE = %.9f > %.9f ", nmse, threshold);
            printf("\033[1;31mFAIL\033[0m\n");
            return false;
        }

        printf("\033[1;32mOK\033[0m (NMSE=%.2e)\n", nmse);
        return true;
    }
};

// GGML_OP_GROUPED_TOPK — uses custom eval (the CPU backend implements the
// op directly via iqk_grouped_top_k, so we compare CPU to Vulkan).
struct test_grouped_topk {
    const int64_t n_experts;
    const int64_t n_groups;
    const int64_t n_top_groups;
    const int64_t nk;            // top-k WITHIN each group used for scoring
    const int64_t topk_experts;  // number of experts to select overall
    const int64_t n_tokens;

    test_grouped_topk(int64_t n_experts = 256, int64_t n_groups = 8,
            int64_t n_top_groups = 4, int64_t nk = 2, int64_t topk_experts = 8,
            int64_t n_tokens = 8)
        : n_experts(n_experts), n_groups(n_groups),
          n_top_groups(n_top_groups), nk(nk),
          topk_experts(topk_experts), n_tokens(n_tokens) {}

    bool eval(ggml_backend_t backend_vk, ggml_backend_t backend_cpu) {
        printf("  GROUPED_TOPK(n_exp=%lld,n_grp=%lld,n_top_grp=%lld,nk=%lld,topk=%lld,n_tok=%lld): ",
               (long long)n_experts, (long long)n_groups,
               (long long)n_top_groups, (long long)nk,
               (long long)topk_experts, (long long)n_tokens);
        fflush(stdout);

        // Build CPU reference graph.
        ggml_init_params params_ref = {
            ggml_tensor_overhead()*16 + ggml_graph_overhead(), NULL, true
        };
        ggml_context * ctx_ref = ggml_init(params_ref);
        ggml_cgraph * gf_ref = ggml_new_graph(ctx_ref);
        ggml_tensor * src_ref = ggml_new_tensor_2d(ctx_ref, GGML_TYPE_F32, n_experts, n_tokens);
        ggml_tensor * out_ref = ggml_grouped_topk(ctx_ref, src_ref,
                (int)n_groups, (int)n_top_groups, (int)nk, (int)topk_experts);
        ggml_build_forward_expand(gf_ref, out_ref);

        ggml_backend_buffer_t buf_ref = ggml_backend_alloc_ctx_tensors(ctx_ref, backend_cpu);
        if (!buf_ref) { printf("alloc fail (cpu)\n"); ggml_free(ctx_ref); return false; }

        // Build Vulkan graph.
        ggml_init_params params_vk = {
            ggml_tensor_overhead()*16 + ggml_graph_overhead(), NULL, true
        };
        ggml_context * ctx_vk = ggml_init(params_vk);
        ggml_cgraph * gf_vk = ggml_new_graph(ctx_vk);
        ggml_tensor * src_vk = ggml_new_tensor_2d(ctx_vk, GGML_TYPE_F32, n_experts, n_tokens);
        ggml_tensor * out_vk = ggml_grouped_topk(ctx_vk, src_vk,
                (int)n_groups, (int)n_top_groups, (int)nk, (int)topk_experts);

        if (!ggml_backend_supports_op(backend_vk, out_vk)) {
            printf("not supported [%s]\n", ggml_backend_name(backend_vk));
            ggml_free(ctx_vk); ggml_backend_buffer_free(buf_ref); ggml_free(ctx_ref);
            return true; // skip, not fail
        }

        ggml_build_forward_expand(gf_vk, out_vk);
        ggml_backend_buffer_t buf_vk = ggml_backend_alloc_ctx_tensors(ctx_vk, backend_vk);
        if (!buf_vk) { printf("alloc fail (vk)\n"); ggml_free(ctx_vk); ggml_backend_buffer_free(buf_ref); ggml_free(ctx_ref); return false; }

        // Initialize identical input on both backends. We use distinct random
        // values per element to keep tie-breaking deterministic across the
        // bitonic sort vs std::partial_sort comparison.
        std::vector<float> src_data(n_experts * n_tokens);
        std::default_random_engine rng(7777);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto & v : src_data) v = dist(rng);

        ggml_backend_tensor_set(src_ref, src_data.data(), 0, src_data.size() * sizeof(float));
        ggml_backend_tensor_set(src_vk,  src_data.data(), 0, src_data.size() * sizeof(float));

        ggml_backend_graph_compute(backend_cpu, gf_ref);
        ggml_backend_graph_compute(backend_vk,  gf_vk);
        ggml_backend_synchronize(backend_vk);

        // Compare outputs. Both are I32 indices. They must be set-equal per
        // row (the order WITHIN a row may differ slightly between CPU
        // partial_sort and bitonic sort because of f32 score equality, but
        // the SET of selected experts must match exactly).
        const size_t out_n = ggml_nelements(out_ref);
        std::vector<int32_t> ref(out_n), vk(out_n);
        ggml_backend_tensor_get(out_ref, ref.data(), 0, out_n * sizeof(int32_t));
        ggml_backend_tensor_get(out_vk,  vk.data(),  0, out_n * sizeof(int32_t));

        bool ok = true;
        for (int t = 0; t < n_tokens && ok; t++) {
            std::vector<int32_t> a(ref.begin() + t * topk_experts, ref.begin() + (t+1) * topk_experts);
            std::vector<int32_t> b(vk.begin()  + t * topk_experts, vk.begin()  + (t+1) * topk_experts);
            std::sort(a.begin(), a.end());
            std::sort(b.begin(), b.end());
            if (a != b) {
                printf("\n    [debug] token %d mismatch: ref={", t);
                for (auto x : a) printf("%d ", x);
                printf("} vk={");
                for (auto x : b) printf("%d ", x);
                printf("}\n");
                ok = false;
            }
        }

        ggml_backend_buffer_free(buf_vk); ggml_free(ctx_vk);
        ggml_backend_buffer_free(buf_ref); ggml_free(ctx_ref);

        if (!ok) {
            printf("\033[1;31mFAIL\033[0m\n");
            return false;
        }
        printf("\033[1;32mOK\033[0m\n");
        return true;
    }
};

// GGML_OP_MUL_MULTI_ADD
struct test_mul_multi_add : public test_case {
    const int64_t ne0;       // n_embd
    const int64_t n_expert;  // ne[1] of a (expert dim)
    const int64_t n_tokens;  // ne[2] of a (token dim)

    std::string vars() override {
        return VARS_TO_STR3(ne0, n_expert, n_tokens);
    }

    test_mul_multi_add(int64_t ne0 = 128, int64_t n_expert = 6, int64_t n_tokens = 4)
        : ne0(ne0), n_expert(n_expert), n_tokens(n_tokens) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        // a: [n_embd, n_expert_used, n_tokens]
        // b: [1,      n_expert_used, n_tokens]
        // dst: [n_embd, n_tokens]
        ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, n_expert, n_tokens);
        ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1,   n_expert, n_tokens);
        ggml_tensor * out = ggml_mul_multi_add(ctx, a, b);
        return out;
    }
};

// BATCH_INVARIANCE probes — run the SAME op on the SAME backend twice, once
// with n_tokens=1 and once with n_tokens=N (seeded so that position 0 sees
// identical input in both runs), then compare the output at position 0
// byte-identical. Any divergence means the op has a batch-shape-dependent
// code path that breaks sequential equivalence, which propagates drift in
// speculative decoding (see test-35b-pos-i-sequential-equivalence).

// Seeded uniform fill to keep test deterministic across runs and libstdc++.
static inline void bi_fill(float * data, size_t n, float lo, float hi, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        const float u = (float)s / (float)0xFFFFFFFFu;
        data[i] = lo + (hi - lo) * u;
    }
}

// test_batch_invariance_mul_mat: plain MUL_MAT (W @ x) at n=1 vs n=N.
//   Weight: [K, M] type_a.  Input: [K, N] f32.  Output: [M, N] f32.
//   Fills position 0 of input with a fixed seed; positions 1..N-1 with a
//   different seed. Weight is shared between both runs.
//   Compares output[:, 0] byte-identical.
struct test_batch_invariance_mul_mat {
    const ggml_type type_a;
    const int64_t K;
    const int64_t M;
    const int64_t N;   // n_tokens for the "batch=N" run

    test_batch_invariance_mul_mat(ggml_type type_a = GGML_TYPE_Q8_0,
            int64_t K = 256, int64_t M = 256, int64_t N = 4)
        : type_a(type_a), K(K), M(M), N(N) {}

    bool eval(ggml_backend_t backend) {
        printf("  BI_MUL_MAT(type_a=%s,K=%lld,M=%lld,N=%lld): ",
               ggml_type_name(type_a), (long long)K, (long long)M, (long long)N);
        fflush(stdout);

        // Weight is shared across both runs — quantize once.
        std::vector<uint8_t> w_bytes(ggml_row_size(type_a, K) * M);
        {
            std::vector<float> tmp(K * M);
            bi_fill(tmp.data(), tmp.size(), -0.5f, 0.5f, 0xA);
            ggml_quantize_chunk(type_a, tmp.data(), w_bytes.data(), 0, M, K, nullptr);
        }

        // Input at position 0 (shared between both runs).
        std::vector<float> x0(K);
        bi_fill(x0.data(), K, -0.5f, 0.5f, 0xB);

        auto run_one = [&](int64_t n_tokens, std::vector<float> & out_pos0) -> bool {
            ggml_init_params p = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), NULL, true };
            ggml_context * ctx = ggml_init(p);
            ggml_cgraph * gf = ggml_new_graph(ctx);

            ggml_tensor * w = ggml_new_tensor_2d(ctx, type_a, K, M);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n_tokens);
            ggml_tensor * y = ggml_mul_mat(ctx, w, x);
            ggml_build_forward_expand(gf, y);

            if (!ggml_backend_supports_op(backend, y)) {
                printf("not supported\n"); ggml_free(ctx); return false;
            }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buf) { printf("alloc fail\n"); ggml_free(ctx); return false; }

            ggml_backend_tensor_set(w, w_bytes.data(), 0, w_bytes.size());

            std::vector<float> x_host((size_t)K * n_tokens);
            // Position 0: shared seed.
            memcpy(x_host.data(), x0.data(), K * sizeof(float));
            // Positions 1..N-1: different seed to exercise the batch.
            if (n_tokens > 1) {
                bi_fill(x_host.data() + K, K * (n_tokens - 1), -0.5f, 0.5f, 0xC);
            }
            ggml_backend_tensor_set(x, x_host.data(), 0, x_host.size() * sizeof(float));

            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);

            out_pos0.resize(M);
            ggml_backend_tensor_get(y, out_pos0.data(), 0, M * sizeof(float));

            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return true;
        };

        std::vector<float> y1a, y1b, yN;
        // Run A: n=1 baseline.
        if (!run_one(1, y1a)) return true;
        // Run B: n=1 again, same inputs. If y1a != y1b, the shader is
        // non-deterministic at the GPU level — same pipeline, same buffers,
        // same dispatch grid produces different output. That's a deeper
        // problem than batch-variance.
        if (!run_one(1, y1b)) return true;
        // Run C: n=N, pos-0 should equal n=1 output.
        if (!run_one(N, yN)) return true;

        auto cmp = [](const std::vector<float> & a, const std::vector<float> & b, int64_t M) {
            size_t diff_count = 0; double max_abs = 0.0;
            for (size_t i = 0; i < (size_t)M; i++) {
                if (a[i] != b[i]) {
                    diff_count++;
                    double d = std::fabs((double)a[i] - (double)b[i]);
                    if (d > max_abs) max_abs = d;
                }
            }
            return std::make_pair(diff_count, max_abs);
        };

        auto [det_diff, det_max] = cmp(y1a, y1b, M);
        auto [bi_diff, bi_max]   = cmp(y1a, yN, M);

        if (det_diff != 0) {
            printf("\033[1;33mNONDETERMINISTIC\033[0m A!=B diff=%zu max|Δ|=%.3g  (BI also: diff=%zu max|Δ|=%.3g)\n",
                   det_diff, det_max, bi_diff, bi_max);
            return false;
        }
        if (bi_diff == 0) {
            printf("\033[1;32mOK\033[0m (byte-identical, det=OK)\n");
            return true;
        }
        printf("\033[1;31mFAIL\033[0m BI diff=%zu/%lld max|Δ|=%.3g (det=OK: same-input → same-output)\n",
               bi_diff, (long long)M, bi_max);
        return false;
    }
};

// test_batch_invariance_soft_max: SOFT_MAX along axis-0 of [K, N].
struct test_batch_invariance_soft_max {
    const int64_t K;   // row length (softmax axis)
    const int64_t N;   // n_tokens

    test_batch_invariance_soft_max(int64_t K = 512, int64_t N = 4) : K(K), N(N) {}

    bool eval(ggml_backend_t backend) {
        printf("  BI_SOFT_MAX(K=%lld,N=%lld): ", (long long)K, (long long)N);
        fflush(stdout);

        std::vector<float> x0(K);
        bi_fill(x0.data(), K, -5.0f, 5.0f, 0xB);

        auto run_one = [&](int64_t n_tokens, std::vector<float> & out_pos0) -> bool {
            ggml_init_params p = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), NULL, true };
            ggml_context * ctx = ggml_init(p);
            ggml_cgraph * gf = ggml_new_graph(ctx);

            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n_tokens);
            ggml_tensor * y = ggml_soft_max(ctx, x);
            ggml_build_forward_expand(gf, y);

            if (!ggml_backend_supports_op(backend, y)) {
                printf("not supported\n"); ggml_free(ctx); return false;
            }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buf) { printf("alloc fail\n"); ggml_free(ctx); return false; }

            std::vector<float> x_host((size_t)K * n_tokens);
            memcpy(x_host.data(), x0.data(), K * sizeof(float));
            if (n_tokens > 1) {
                bi_fill(x_host.data() + K, K * (n_tokens - 1), -5.0f, 5.0f, 0xC);
            }
            ggml_backend_tensor_set(x, x_host.data(), 0, x_host.size() * sizeof(float));

            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);

            out_pos0.resize(K);
            ggml_backend_tensor_get(y, out_pos0.data(), 0, K * sizeof(float));

            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return true;
        };

        std::vector<float> y1, yN;
        if (!run_one(1, y1)) return true;
        if (!run_one(N, yN)) return true;

        size_t diff_count = 0; double max_abs = 0.0;
        for (size_t i = 0; i < (size_t)K; i++) {
            if (y1[i] != yN[i]) { diff_count++; double d = std::fabs((double)y1[i] - (double)yN[i]); if (d > max_abs) max_abs = d; }
        }
        if (diff_count == 0) { printf("\033[1;32mOK\033[0m (byte-identical)\n"); return true; }
        printf("\033[1;31mFAIL\033[0m diff=%zu/%lld max|Δ|=%.3g\n", diff_count, (long long)K, max_abs);
        return false;
    }
};

// test_batch_invariance_rms_norm: RMS_NORM along axis-0 of [K, N].
struct test_batch_invariance_rms_norm {
    const int64_t K;
    const int64_t N;

    test_batch_invariance_rms_norm(int64_t K = 512, int64_t N = 4) : K(K), N(N) {}

    bool eval(ggml_backend_t backend) {
        printf("  BI_RMS_NORM(K=%lld,N=%lld): ", (long long)K, (long long)N);
        fflush(stdout);

        std::vector<float> x0(K);
        bi_fill(x0.data(), K, -1.0f, 1.0f, 0xB);

        auto run_one = [&](int64_t n_tokens, std::vector<float> & out_pos0) -> bool {
            ggml_init_params p = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), NULL, true };
            ggml_context * ctx = ggml_init(p);
            ggml_cgraph * gf = ggml_new_graph(ctx);

            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n_tokens);
            ggml_tensor * y = ggml_rms_norm(ctx, x, 1e-5f);
            ggml_build_forward_expand(gf, y);

            if (!ggml_backend_supports_op(backend, y)) {
                printf("not supported\n"); ggml_free(ctx); return false;
            }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buf) { printf("alloc fail\n"); ggml_free(ctx); return false; }

            std::vector<float> x_host((size_t)K * n_tokens);
            memcpy(x_host.data(), x0.data(), K * sizeof(float));
            if (n_tokens > 1) bi_fill(x_host.data() + K, K * (n_tokens - 1), -1.0f, 1.0f, 0xC);
            ggml_backend_tensor_set(x, x_host.data(), 0, x_host.size() * sizeof(float));

            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);

            out_pos0.resize(K);
            ggml_backend_tensor_get(y, out_pos0.data(), 0, K * sizeof(float));

            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return true;
        };

        std::vector<float> y1, yN;
        if (!run_one(1, y1)) return true;
        if (!run_one(N, yN)) return true;

        size_t diff_count = 0; double max_abs = 0.0;
        for (size_t i = 0; i < (size_t)K; i++) {
            if (y1[i] != yN[i]) { diff_count++; double d = std::fabs((double)y1[i] - (double)yN[i]); if (d > max_abs) max_abs = d; }
        }
        if (diff_count == 0) { printf("\033[1;32mOK\033[0m (byte-identical)\n"); return true; }
        printf("\033[1;31mFAIL\033[0m diff=%zu/%lld max|Δ|=%.3g\n", diff_count, (long long)K, max_abs);
        return false;
    }
};

// test_batch_invariance_unary: SILU / GELU / RELU at n=1 vs n=N.
struct test_batch_invariance_unary {
    const ggml_unary_op op;
    const int64_t K;
    const int64_t N;

    test_batch_invariance_unary(ggml_unary_op op = GGML_UNARY_OP_SILU,
            int64_t K = 512, int64_t N = 4) : op(op), K(K), N(N) {}

    bool eval(ggml_backend_t backend) {
        printf("  BI_UNARY(op=%d,K=%lld,N=%lld): ", (int)op, (long long)K, (long long)N);
        fflush(stdout);

        std::vector<float> x0(K);
        bi_fill(x0.data(), K, -3.0f, 3.0f, 0xB);

        auto run_one = [&](int64_t n_tokens, std::vector<float> & out_pos0) -> bool {
            ggml_init_params p = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), NULL, true };
            ggml_context * ctx = ggml_init(p);
            ggml_cgraph * gf = ggml_new_graph(ctx);

            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n_tokens);
            ggml_tensor * y = ggml_unary(ctx, x, op);
            ggml_build_forward_expand(gf, y);

            if (!ggml_backend_supports_op(backend, y)) {
                printf("not supported\n"); ggml_free(ctx); return false;
            }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buf) { printf("alloc fail\n"); ggml_free(ctx); return false; }

            std::vector<float> x_host((size_t)K * n_tokens);
            memcpy(x_host.data(), x0.data(), K * sizeof(float));
            if (n_tokens > 1) bi_fill(x_host.data() + K, K * (n_tokens - 1), -3.0f, 3.0f, 0xC);
            ggml_backend_tensor_set(x, x_host.data(), 0, x_host.size() * sizeof(float));

            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);

            out_pos0.resize(K);
            ggml_backend_tensor_get(y, out_pos0.data(), 0, K * sizeof(float));

            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return true;
        };

        std::vector<float> y1, yN;
        if (!run_one(1, y1)) return true;
        if (!run_one(N, yN)) return true;

        size_t diff_count = 0; double max_abs = 0.0;
        for (size_t i = 0; i < (size_t)K; i++) {
            if (y1[i] != yN[i]) { diff_count++; double d = std::fabs((double)y1[i] - (double)yN[i]); if (d > max_abs) max_abs = d; }
        }
        if (diff_count == 0) { printf("\033[1;32mOK\033[0m (byte-identical)\n"); return true; }
        printf("\033[1;31mFAIL\033[0m diff=%zu/%lld max|Δ|=%.3g\n", diff_count, (long long)K, max_abs);
        return false;
    }
};

// test_batch_invariance_fused_up_gate: dense FUSED_UP_GATE at n=1 vs n=N.
//   up/gate: [K, M] quantized. b: [K, N] f32. op: activation.
//   Compares the pos-0 row of the output byte-identical.
struct test_batch_invariance_fused_up_gate {
    const ggml_type type_a;
    const int64_t K;
    const int64_t M;
    const int64_t N;
    const ggml_unary_op op;

    test_batch_invariance_fused_up_gate(ggml_type type_a = GGML_TYPE_Q8_0,
            int64_t K = 2048, int64_t M = 512, int64_t N = 4,
            ggml_unary_op op = GGML_UNARY_OP_SILU)
        : type_a(type_a), K(K), M(M), N(N), op(op) {}

    bool eval(ggml_backend_t backend) {
        printf("  BI_FUSED_UP_GATE(type_a=%s,K=%lld,M=%lld,N=%lld,op=%d): ",
               ggml_type_name(type_a), (long long)K, (long long)M, (long long)N, (int)op);
        fflush(stdout);

        std::vector<uint8_t> up_bytes(ggml_row_size(type_a, K) * M);
        std::vector<uint8_t> gate_bytes(ggml_row_size(type_a, K) * M);
        {
            std::vector<float> tmp(K * M);
            bi_fill(tmp.data(), tmp.size(), -0.5f, 0.5f, 0xA);
            ggml_quantize_chunk(type_a, tmp.data(), up_bytes.data(), 0, M, K, nullptr);
            bi_fill(tmp.data(), tmp.size(), -0.5f, 0.5f, 0xD);
            ggml_quantize_chunk(type_a, tmp.data(), gate_bytes.data(), 0, M, K, nullptr);
        }
        std::vector<float> b0(K);
        bi_fill(b0.data(), K, -0.5f, 0.5f, 0xB);

        auto run_one = [&](int64_t n_tokens, std::vector<float> & out_pos0) -> bool {
            ggml_init_params p = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), NULL, true };
            ggml_context * ctx = ggml_init(p);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_tensor * up   = ggml_new_tensor_2d(ctx, type_a, K, M);
            ggml_tensor * gate = ggml_new_tensor_2d(ctx, type_a, K, M);
            ggml_tensor * b    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n_tokens);
            ggml_tensor * y    = ggml_fused_up_gate(ctx, up, gate, b, op);
            ggml_build_forward_expand(gf, y);
            if (!ggml_backend_supports_op(backend, y)) { printf("not supported\n"); ggml_free(ctx); return false; }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buf) { printf("alloc fail\n"); ggml_free(ctx); return false; }
            ggml_backend_tensor_set(up,   up_bytes.data(),   0, up_bytes.size());
            ggml_backend_tensor_set(gate, gate_bytes.data(), 0, gate_bytes.size());
            std::vector<float> b_host((size_t)K * n_tokens);
            memcpy(b_host.data(), b0.data(), K * sizeof(float));
            if (n_tokens > 1) bi_fill(b_host.data() + K, K * (n_tokens - 1), -0.5f, 0.5f, 0xC);
            ggml_backend_tensor_set(b, b_host.data(), 0, b_host.size() * sizeof(float));
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
            out_pos0.resize(M);
            ggml_backend_tensor_get(y, out_pos0.data(), 0, M * sizeof(float));
            ggml_backend_buffer_free(buf); ggml_free(ctx);
            return true;
        };

        std::vector<float> y1, yN;
        if (!run_one(1, y1)) return true;
        if (!run_one(N, yN)) return true;

        size_t diff_count = 0; double max_abs = 0.0;
        for (size_t i = 0; i < (size_t)M; i++) {
            if (y1[i] != yN[i]) { diff_count++; double d = std::fabs((double)y1[i] - (double)yN[i]); if (d > max_abs) max_abs = d; }
        }
        if (diff_count == 0) { printf("\033[1;32mOK\033[0m (byte-identical)\n"); return true; }
        printf("\033[1;31mFAIL\033[0m diff=%zu/%lld max|Δ|=%.3g\n", diff_count, (long long)M, max_abs);
        return false;
    }
};

// test_pos1_invariance_moe_fused_up_gate: probes an op-level BI bug that
// pos-0-only testing misses. The fused_moe shader handles pos=0 correctly
// (matches sequential) but positions 1+ in a batched dispatch diverge from
// what an isolated n_tokens=1 call would produce with the same pos-1 input.
// Full-model evidence: decode(N=4).logits[pos=1] differs from
// sequential-through-pos-1.logits[0] by max|Δ|=5.1 on qwen35-A3B.
struct test_pos1_invariance_moe_fused_up_gate {
    const ggml_type type_a;
    const int64_t K;
    const int64_t M;
    const int64_t n_experts;
    const int64_t n_expert_used;
    const ggml_unary_op op;

    test_pos1_invariance_moe_fused_up_gate(ggml_type type_a = GGML_TYPE_Q4_K,
            int64_t K = 256, int64_t M = 256,
            int64_t n_experts = 16, int64_t n_expert_used = 2,
            ggml_unary_op op = GGML_UNARY_OP_SILU)
        : type_a(type_a), K(K), M(M), n_experts(n_experts),
          n_expert_used(n_expert_used), op(op) {}

    bool eval(ggml_backend_t backend) {
        printf("  BI_MOE_FUSED_UP_GATE_POS1(type_a=%s,K=%lld,M=%lld,n_exp=%lld,n_used=%lld): ",
               ggml_type_name(type_a), (long long)K, (long long)M,
               (long long)n_experts, (long long)n_expert_used);
        fflush(stdout);

        const size_t leaf_bytes = (size_t)ggml_row_size(type_a, K) * (size_t)M * (size_t)n_experts;
        if (leaf_bytes > (size_t)1 << 30) {
            printf("\033[1;33mSKIP\033[0m (tensor %.2f GiB > 1 GiB)\n", (double)leaf_bytes / (double)(1ULL<<30));
            return true;
        }

        std::vector<uint8_t> up_bytes(ggml_row_size(type_a, K) * M * n_experts);
        std::vector<uint8_t> gate_bytes(ggml_row_size(type_a, K) * M * n_experts);
        {
            std::vector<float> tmp(K * M * n_experts);
            bi_fill(tmp.data(), tmp.size(), -0.5f, 0.5f, 0xA);
            ggml_quantize_chunk(type_a, tmp.data(), up_bytes.data(), 0, M * n_experts, K, nullptr);
            bi_fill(tmp.data(), tmp.size(), -0.5f, 0.5f, 0xD);
            ggml_quantize_chunk(type_a, tmp.data(), gate_bytes.data(), 0, M * n_experts, K, nullptr);
        }
        // Two independent input vectors, fixed across runs.
        std::vector<float> b_pos0(K), b_pos1(K);
        bi_fill(b_pos0.data(), K, -0.5f, 0.5f, 0xB);
        bi_fill(b_pos1.data(), K, -0.5f, 0.5f, 0xC);
        // Fixed expert ids per position (token-0 uses ids_pos0, token-1 uses ids_pos1).
        std::vector<int32_t> ids_pos0(n_expert_used), ids_pos1(n_expert_used);
        for (int e = 0; e < n_expert_used; e++) {
            ids_pos0[e] = (int32_t)((e * 7 + 3) % n_experts);
            ids_pos1[e] = (int32_t)((e * 11 + 5) % n_experts);
        }

        // run_one: ggml_moe_up_gate with n_tokens=nt; input[pos]=b_pos, ids[pos]=ids_pos.
        auto run_one = [&](int nt, const std::vector<std::vector<float>> & bs,
                           const std::vector<std::vector<int32_t>> & idss,
                           std::vector<std::vector<float>> & out_per_pos) -> bool {
            ggml_init_params p = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), NULL, true };
            ggml_context * ctx = ggml_init(p);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_tensor * up   = ggml_new_tensor_3d(ctx, type_a, K, M, n_experts);
            ggml_tensor * gate = ggml_new_tensor_3d(ctx, type_a, K, M, n_experts);
            ggml_tensor * b    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, nt);
            ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, nt);
            ggml_tensor * y    = ggml_moe_up_gate(ctx, up, gate, b, ids, op);
            ggml_build_forward_expand(gf, y);
            if (!ggml_backend_supports_op(backend, y)) { printf("not supported\n"); ggml_free(ctx); return false; }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buf) { printf("alloc fail\n"); ggml_free(ctx); return false; }
            ggml_backend_tensor_set(up,   up_bytes.data(),   0, up_bytes.size());
            ggml_backend_tensor_set(gate, gate_bytes.data(), 0, gate_bytes.size());
            std::vector<float> b_host((size_t)K * nt);
            std::vector<int32_t> ids_host((size_t)n_expert_used * nt);
            for (int t = 0; t < nt; t++) {
                memcpy(b_host.data() + t * K, bs[t].data(), K * sizeof(float));
                memcpy(ids_host.data() + t * n_expert_used, idss[t].data(), n_expert_used * sizeof(int32_t));
            }
            ggml_backend_tensor_set(b, b_host.data(), 0, b_host.size() * sizeof(float));
            ggml_backend_tensor_set(ids, ids_host.data(), 0, ids_host.size() * sizeof(int32_t));
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
            const size_t per_pos = (size_t)M * n_expert_used;
            out_per_pos.assign(nt, std::vector<float>(per_pos));
            for (int t = 0; t < nt; t++) {
                ggml_backend_tensor_get(y, out_per_pos[t].data(), t * per_pos * sizeof(float), per_pos * sizeof(float));
            }
            ggml_backend_buffer_free(buf); ggml_free(ctx);
            return true;
        };

        // Batched: N=2 with [b_pos0, b_pos1] / [ids_pos0, ids_pos1].
        std::vector<std::vector<float>>   bs_batched   = { b_pos0, b_pos1 };
        std::vector<std::vector<int32_t>> idss_batched = { ids_pos0, ids_pos1 };
        std::vector<std::vector<float>>   out_batched;
        if (!run_one(2, bs_batched, idss_batched, out_batched)) return true;

        // Solo-at-pos-1: N=1 with [b_pos1] / [ids_pos1].
        std::vector<std::vector<float>>   bs_solo   = { b_pos1 };
        std::vector<std::vector<int32_t>> idss_solo = { ids_pos1 };
        std::vector<std::vector<float>>   out_solo;
        if (!run_one(1, bs_solo, idss_solo, out_solo)) return true;

        // Compare batched[pos=1] vs solo[pos=0].
        size_t diff_count = 0; double max_abs = 0.0;
        for (size_t i = 0; i < out_batched[1].size(); i++) {
            if (out_batched[1][i] != out_solo[0][i]) {
                diff_count++;
                double d = std::fabs((double)out_batched[1][i] - (double)out_solo[0][i]);
                if (d > max_abs) max_abs = d;
            }
        }
        if (diff_count == 0) {
            printf("\033[1;32mOK\033[0m (pos-1 byte-identical to solo run)\n");
            return true;
        }
        printf("\033[1;31mFAIL\033[0m diff=%zu/%zu max|Δ|=%.3g  (op produces non-BI pos-1 output in batched mode)\n",
               diff_count, out_batched[1].size(), max_abs);
        return false;
    }
};

// test_batch_invariance_moe_fused_up_gate: MoE FUSED_UP_GATE at n=1 vs n=N.
//   up/gate: [K, M, n_experts] quantized. b: [K, 1, n_tokens] f32. ids: [n_used, n_tokens] i32.
//   Expert routing is deterministic per fixed seed; pos-0 (token 0) sees same experts in both runs.
struct test_batch_invariance_moe_fused_up_gate {
    const ggml_type type_a;
    const int64_t K;
    const int64_t M;
    const int64_t n_experts;
    const int64_t n_expert_used;
    const int64_t N;
    const ggml_unary_op op;

    test_batch_invariance_moe_fused_up_gate(ggml_type type_a = GGML_TYPE_Q8_0,
            int64_t K = 2048, int64_t M = 512,
            int64_t n_experts = 16, int64_t n_expert_used = 2,
            int64_t N = 4, ggml_unary_op op = GGML_UNARY_OP_SILU)
        : type_a(type_a), K(K), M(M), n_experts(n_experts),
          n_expert_used(n_expert_used), N(N), op(op) {}

    bool eval(ggml_backend_t backend) {
        printf("  BI_MOE_FUSED_UP_GATE(type_a=%s,K=%lld,M=%lld,n_exp=%lld,n_used=%lld,N=%lld): ",
               ggml_type_name(type_a), (long long)K, (long long)M,
               (long long)n_experts, (long long)n_expert_used, (long long)N);
        fflush(stdout);

        // Skip shapes that exceed the conservative 1 GiB single-buffer limit
        // most Vulkan drivers report. We print an explicit SKIP so the large-
        // shape coverage gap is visible in the test output rather than a
        // silent pass after a ggml-level "tensor too large" error.
        const size_t leaf_bytes = (size_t)ggml_row_size(type_a, K) * (size_t)M * (size_t)n_experts;
        const size_t kMaxBufferBytes = (size_t)1 << 30;  // 1 GiB
        if (leaf_bytes > kMaxBufferBytes) {
            printf("\033[1;33mSKIP\033[0m (expert-weight tensor %.2f GiB > 1 GiB buffer limit)\n",
                   (double)leaf_bytes / (double)(1ULL << 30));
            return true;
        }

        std::vector<uint8_t> up_bytes(ggml_row_size(type_a, K) * M * n_experts);
        std::vector<uint8_t> gate_bytes(ggml_row_size(type_a, K) * M * n_experts);
        {
            std::vector<float> tmp(K * M * n_experts);
            bi_fill(tmp.data(), tmp.size(), -0.5f, 0.5f, 0xA);
            ggml_quantize_chunk(type_a, tmp.data(), up_bytes.data(), 0, M * n_experts, K, nullptr);
            bi_fill(tmp.data(), tmp.size(), -0.5f, 0.5f, 0xD);
            ggml_quantize_chunk(type_a, tmp.data(), gate_bytes.data(), 0, M * n_experts, K, nullptr);
        }
        std::vector<float> b0(K);
        bi_fill(b0.data(), K, -0.5f, 0.5f, 0xB);
        // Token 0's expert ids — randomized to exercise arbitrary routing
        // across the full expert grid (real models rarely pick [0,1,...,k-1]).
        // Still identical between N=1 and N=N runs so token-0 comparison is
        // meaningful. Deterministic LCG seeded from the test shape.
        std::vector<int32_t> ids0(n_expert_used);
        {
            uint32_t s = 0x600D ^ (uint32_t)K ^ ((uint32_t)M << 8) ^ ((uint32_t)n_experts << 16);
            std::vector<bool> used(n_experts, false);
            for (int e = 0; e < n_expert_used; e++) {
                int32_t pick;
                do {
                    s = s * 1664525u + 1013904223u;
                    pick = (int32_t)(s % n_experts);
                } while (used[pick]);
                used[pick] = true;
                ids0[e] = pick;
            }
        }

        auto run_one = [&](int64_t n_tokens, std::vector<float> & out_pos0) -> bool {
            ggml_init_params p = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), NULL, true };
            ggml_context * ctx = ggml_init(p);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_tensor * up   = ggml_new_tensor_3d(ctx, type_a, K, M, n_experts);
            ggml_tensor * gate = ggml_new_tensor_3d(ctx, type_a, K, M, n_experts);
            ggml_tensor * b    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, n_tokens);
            ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, n_tokens);
            ggml_tensor * y    = ggml_moe_up_gate(ctx, up, gate, b, ids, op);
            ggml_build_forward_expand(gf, y);
            if (!ggml_backend_supports_op(backend, y)) { printf("not supported\n"); ggml_free(ctx); return false; }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buf) { printf("alloc fail\n"); ggml_free(ctx); return false; }
            ggml_backend_tensor_set(up,   up_bytes.data(),   0, up_bytes.size());
            ggml_backend_tensor_set(gate, gate_bytes.data(), 0, gate_bytes.size());
            std::vector<float> b_host((size_t)K * n_tokens);
            memcpy(b_host.data(), b0.data(), K * sizeof(float));
            if (n_tokens > 1) bi_fill(b_host.data() + K, K * (n_tokens - 1), -0.5f, 0.5f, 0xC);
            ggml_backend_tensor_set(b, b_host.data(), 0, b_host.size() * sizeof(float));
            // Token 0's ids fixed; other tokens get random routing.
            std::vector<int32_t> ids_host((size_t)n_expert_used * n_tokens);
            memcpy(ids_host.data(), ids0.data(), n_expert_used * sizeof(int32_t));
            for (int t = 1; t < n_tokens; t++) {
                uint32_t s = 0xE + t;
                for (int e = 0; e < n_expert_used; e++) {
                    s = s * 1664525u + 1013904223u;
                    ids_host[t * n_expert_used + e] = (int32_t)(s % n_experts);
                }
            }
            ggml_backend_tensor_set(ids, ids_host.data(), 0, ids_host.size() * sizeof(int32_t));
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
            // Output is [M, n_expert_used, n_tokens]. Read token-0 slice = first M*n_expert_used floats.
            const size_t token0_n = (size_t)M * n_expert_used;
            out_pos0.resize(token0_n);
            ggml_backend_tensor_get(y, out_pos0.data(), 0, token0_n * sizeof(float));
            ggml_backend_buffer_free(buf); ggml_free(ctx);
            return true;
        };

        std::vector<float> y1, yN;
        if (!run_one(1, y1)) return true;
        if (!run_one(N, yN)) return true;

        size_t diff_count = 0; double max_abs = 0.0;
        for (size_t i = 0; i < y1.size(); i++) {
            if (y1[i] != yN[i]) { diff_count++; double d = std::fabs((double)y1[i] - (double)yN[i]); if (d > max_abs) max_abs = d; }
        }
        if (diff_count == 0) { printf("\033[1;32mOK\033[0m (byte-identical)\n"); return true; }
        printf("\033[1;31mFAIL\033[0m diff=%zu/%zu max|Δ|=%.3g\n", diff_count, y1.size(), max_abs);
        return false;
    }
};

// test_batch_invariance_mul_mat_id: non-fused MoE mat-mat routing at n=1 vs n=N.
//   as:  [K, M, n_experts] quantized.
//   b:   [K, 1, n_tokens]  f32 (broadcast over n_expert_used).
//   ids: [n_expert_used, n_tokens] i32. Token-0's ids fixed across runs.
//   Output [M, n_expert_used, n_tokens]. Compares token-0 slice byte-identical.
struct test_batch_invariance_mul_mat_id {
    const ggml_type type_a;
    const int64_t K;
    const int64_t M;
    const int64_t n_experts;
    const int64_t n_expert_used;
    const int64_t N;

    test_batch_invariance_mul_mat_id(ggml_type type_a = GGML_TYPE_Q8_0,
            int64_t K = 2048, int64_t M = 512,
            int64_t n_experts = 16, int64_t n_expert_used = 2,
            int64_t N = 4)
        : type_a(type_a), K(K), M(M), n_experts(n_experts),
          n_expert_used(n_expert_used), N(N) {}

    bool eval(ggml_backend_t backend) {
        printf("  BI_MUL_MAT_ID(type_a=%s,K=%lld,M=%lld,n_exp=%lld,n_used=%lld,N=%lld): ",
               ggml_type_name(type_a), (long long)K, (long long)M,
               (long long)n_experts, (long long)n_expert_used, (long long)N);
        fflush(stdout);

        std::vector<uint8_t> as_bytes(ggml_row_size(type_a, K) * M * n_experts);
        {
            std::vector<float> tmp(K * M * n_experts);
            bi_fill(tmp.data(), tmp.size(), -0.5f, 0.5f, 0xA);
            ggml_quantize_chunk(type_a, tmp.data(), as_bytes.data(), 0, M * n_experts, K, nullptr);
        }
        std::vector<float> b0(K);
        bi_fill(b0.data(), K, -0.5f, 0.5f, 0xB);
        // Token 0's expert ids — fixed, identical across runs.
        std::vector<int32_t> ids0(n_expert_used);
        for (int e = 0; e < n_expert_used; e++) ids0[e] = e;

        auto run_one = [&](int64_t n_tokens, std::vector<float> & out_pos0) -> bool {
            ggml_init_params p = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), NULL, true };
            ggml_context * ctx = ggml_init(p);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_tensor * as  = ggml_new_tensor_3d(ctx, type_a, K, M, n_experts);
            ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, n_tokens);
            ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_expert_used, n_tokens);
            ggml_tensor * y   = ggml_mul_mat_id(ctx, as, b, ids);
            ggml_build_forward_expand(gf, y);
            if (!ggml_backend_supports_op(backend, y)) { printf("not supported\n"); ggml_free(ctx); return false; }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buf) { printf("alloc fail\n"); ggml_free(ctx); return false; }
            ggml_backend_tensor_set(as, as_bytes.data(), 0, as_bytes.size());
            std::vector<float> b_host((size_t)K * n_tokens);
            memcpy(b_host.data(), b0.data(), K * sizeof(float));
            if (n_tokens > 1) bi_fill(b_host.data() + K, K * (n_tokens - 1), -0.5f, 0.5f, 0xC);
            ggml_backend_tensor_set(b, b_host.data(), 0, b_host.size() * sizeof(float));
            std::vector<int32_t> ids_host((size_t)n_expert_used * n_tokens);
            memcpy(ids_host.data(), ids0.data(), n_expert_used * sizeof(int32_t));
            for (int t = 1; t < n_tokens; t++) {
                uint32_t s = 0xE + t;
                for (int e = 0; e < n_expert_used; e++) {
                    s = s * 1664525u + 1013904223u;
                    ids_host[t * n_expert_used + e] = (int32_t)(s % n_experts);
                }
            }
            ggml_backend_tensor_set(ids, ids_host.data(), 0, ids_host.size() * sizeof(int32_t));
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
            // Output [M, n_expert_used, n_tokens] row-major in ne order. Token-0 slice = first M*n_expert_used floats.
            const size_t token0_n = (size_t)M * n_expert_used;
            out_pos0.resize(token0_n);
            ggml_backend_tensor_get(y, out_pos0.data(), 0, token0_n * sizeof(float));
            ggml_backend_buffer_free(buf); ggml_free(ctx);
            return true;
        };

        std::vector<float> y1, yN;
        if (!run_one(1, y1)) return true;
        if (!run_one(N, yN)) return true;

        size_t diff_count = 0; double max_abs = 0.0;
        for (size_t i = 0; i < y1.size(); i++) {
            if (y1[i] != yN[i]) { diff_count++; double d = std::fabs((double)y1[i] - (double)yN[i]); if (d > max_abs) max_abs = d; }
        }
        if (diff_count == 0) { printf("\033[1;32mOK\033[0m (byte-identical)\n"); return true; }
        printf("\033[1;31mFAIL\033[0m diff=%zu/%zu max|Δ|=%.3g\n", diff_count, y1.size(), max_abs);
        return false;
    }
};

// test_batch_invariance_flash_attn: FLASH_ATTN_EXT at nb=1 vs nb=N.
//   q:    [hs, nb,  nh_q, 1] f32
//   k/v:  [hs, kv,  nh_kv, 1] f16
//   mask: [kv, GGML_PAD(nb, GGML_KQ_MASK_PAD), 1, 1] f16 (optional)
//   Compares pos-0 output slice (first HSV * nh_q floats) byte-identical.
//   Covers Vulkan pipeline variants: {cm2/cm1/scalar} × {f16acc/f32acc} ×
//   {small_rows/large_rows} × {aligned/unaligned}. The path actually selected
//   depends on backend capability + N + KV; setting prec and varying nb
//   drives a representative subset.
struct test_batch_invariance_flash_attn {
    const int64_t hs;        // head size (Q/K dim 0)
    const int64_t nh_q;      // num query heads
    const int64_t nh_kv;     // num kv heads (for GQA)
    const int64_t kv;        // kv ctx len
    const int64_t N;         // nb: query batch size for the "batch=N" run
    const bool    f32acc;    // GGML_PREC_F32 accumulator vs default F16
    const bool    use_mask;  // whether to provide a mask

    test_batch_invariance_flash_attn(int64_t hs = 128, int64_t nh_q = 16,
            int64_t nh_kv = 8, int64_t kv = 512, int64_t N = 4,
            bool f32acc = false, bool use_mask = true)
        : hs(hs), nh_q(nh_q), nh_kv(nh_kv), kv(kv), N(N),
          f32acc(f32acc), use_mask(use_mask) {}

    bool eval(ggml_backend_t backend) {
        printf("  BI_FLASH_ATTN(hs=%lld,nh_q=%lld,nh_kv=%lld,kv=%lld,N=%lld,f32acc=%d,mask=%d): ",
               (long long)hs, (long long)nh_q, (long long)nh_kv,
               (long long)kv, (long long)N, (int)f32acc, (int)use_mask);
        fflush(stdout);

        GGML_ASSERT(nh_q % nh_kv == 0);

        const int64_t hs_padded = hs; // f16 KV → blck_size 1
        const float scale = 1.0f / sqrtf((float)hs);

        // K, V shared between both runs.
        std::vector<ggml_fp16_t> k_host((size_t)hs_padded * kv * nh_kv);
        std::vector<ggml_fp16_t> v_host((size_t)hs_padded * kv * nh_kv);
        {
            std::vector<float> tmp(hs_padded * kv * nh_kv);
            bi_fill(tmp.data(), tmp.size(), -0.5f, 0.5f, 0xA);
            ggml_fp32_to_fp16_row(tmp.data(), k_host.data(), tmp.size());
            bi_fill(tmp.data(), tmp.size(), -0.5f, 0.5f, 0xD);
            ggml_fp32_to_fp16_row(tmp.data(), v_host.data(), tmp.size());
        }

        // Q at position 0 (shared across runs).
        std::vector<float> q0((size_t)hs_padded * nh_q);
        bi_fill(q0.data(), q0.size(), -0.5f, 0.5f, 0xB);

        auto run_one = [&](int64_t nb, std::vector<float> & out_pos0) -> bool {
            const int64_t nb_pad = GGML_PAD(nb, GGML_KQ_MASK_PAD);

            ggml_init_params p = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), NULL, true };
            ggml_context * ctx = ggml_init(p);
            ggml_cgraph * gf = ggml_new_graph(ctx);

            ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs_padded, nb,     nh_q,  1);
            ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hs_padded, kv,     nh_kv, 1);
            ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, hs_padded, kv,     nh_kv, 1);
            ggml_tensor * m = use_mask
                ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, nb_pad, 1, 1)
                : nullptr;
            ggml_tensor * y = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);
            if (f32acc) ggml_flash_attn_ext_set_prec(y, GGML_PREC_F32);
            ggml_build_forward_expand(gf, y);

            if (!ggml_backend_supports_op(backend, y)) {
                printf("not supported\n"); ggml_free(ctx); return false;
            }
            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buf) { printf("alloc fail\n"); ggml_free(ctx); return false; }

            ggml_backend_tensor_set(k, k_host.data(), 0, k_host.size() * sizeof(ggml_fp16_t));
            ggml_backend_tensor_set(v, v_host.data(), 0, v_host.size() * sizeof(ggml_fp16_t));

            // Q: position 0 shared, positions 1..nb-1 differ across the batch.
            // Layout: [hs, nb, nh_q, 1] — for each head, nb rows of hs floats.
            // Token 0 = row 0 of each head.
            std::vector<float> q_host((size_t)hs_padded * nb * nh_q);
            for (int64_t h = 0; h < nh_q; h++) {
                // Pos-0 row for head h.
                memcpy(q_host.data() + h * hs_padded * nb,
                       q0.data() + h * hs_padded,
                       hs_padded * sizeof(float));
                // Remaining rows (pos 1..nb-1) — different seed per head.
                if (nb > 1) {
                    bi_fill(q_host.data() + h * hs_padded * nb + hs_padded,
                            hs_padded * (nb - 1), -0.5f, 0.5f, 0xC + (uint32_t)h);
                }
            }
            ggml_backend_tensor_set(q, q_host.data(), 0, q_host.size() * sizeof(float));

            if (m) {
                // Uniform zero mask (all keys attended). Pos-0 row is identical
                // across runs, which is what the BI check requires.
                std::vector<ggml_fp16_t> m_host((size_t)kv * nb_pad, ggml_fp32_to_fp16(0.0f));
                ggml_backend_tensor_set(m, m_host.data(), 0, m_host.size() * sizeof(ggml_fp16_t));
            }

            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);

            // Output y has shape {hs, nh_q, nb, 1} per ggml_flash_attn_ext.
            //   ne[0]=hs, ne[1]=nh_q, ne[2]=nb. Linear index: t*(nh_q*hs) + h*hs + d.
            //   Token-0 slice = the first hs*nh_q floats.
            const size_t token0_n = (size_t)hs * nh_q;
            out_pos0.resize(token0_n);
            ggml_backend_tensor_get(y, out_pos0.data(), 0, token0_n * sizeof(float));

            ggml_backend_buffer_free(buf); ggml_free(ctx);
            return true;
        };

        std::vector<float> y1, yN;
        if (!run_one(1, y1)) return true;
        if (!run_one(N, yN)) return true;

        size_t diff_count = 0; double max_abs = 0.0;
        for (size_t i = 0; i < y1.size(); i++) {
            if (y1[i] != yN[i]) { diff_count++; double d = std::fabs((double)y1[i] - (double)yN[i]); if (d > max_abs) max_abs = d; }
        }
        if (diff_count == 0) { printf("\033[1;32mOK\033[0m (byte-identical)\n"); return true; }
        printf("\033[1;31mFAIL\033[0m diff=%zu/%zu max|Δ|=%.3g\n", diff_count, y1.size(), max_abs);
        return false;
    }
};

// GGML_OP_MULTI_ADD — uses custom eval (like test_fused_up_gate) because
// the op expects a strided view_2d input that the standard test framework
// doesn't initialize correctly.
struct test_multi_add {
    const int64_t ne0;       // n_embd
    const int64_t ne1;       // n_tokens
    const int n_experts;

    test_multi_add(int64_t ne0 = 128, int64_t ne1 = 4, int n_experts = 6)
        : ne0(ne0), ne1(ne1), n_experts(n_experts) {}

    bool eval(ggml_backend_t backend_vk, ggml_backend_t backend_cpu) {
        printf("  MULTI_ADD(ne0=%lld,ne1=%lld,n_experts=%d): ",
               (long long)ne0, (long long)ne1, n_experts);
        fflush(stdout);

        // Both backends get the same graph: experts[ne0, n_experts, ne1] → view_2d → multi_add
        auto make_graph = [&](ggml_context * ctx, ggml_tensor ** p_experts, ggml_tensor ** p_out) {
            *p_experts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, n_experts, ne1);
            ggml_set_name(*p_experts, "experts");
            ggml_tensor * a = ggml_view_2d(ctx, *p_experts, ne0, ne1, (*p_experts)->nb[2], 0);
            *p_out = ggml_multi_add(ctx, a, n_experts);
            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, *p_out);
            return gf;
        };

        // CPU graph
        ggml_init_params params_cpu = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), NULL, true };
        ggml_context * ctx_cpu = ggml_init(params_cpu);
        ggml_tensor * experts_cpu = nullptr, * out_cpu = nullptr;
        ggml_cgraph * gf_cpu = make_graph(ctx_cpu, &experts_cpu, &out_cpu);

        if (!ggml_backend_supports_op(backend_cpu, out_cpu)) {
            printf("not supported [%s]\n", ggml_backend_name(backend_cpu));
            ggml_free(ctx_cpu);
            return true;
        }

        ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
        if (!buf_cpu) { printf("alloc fail (cpu)\n"); ggml_free(ctx_cpu); return false; }

        // VK graph
        ggml_init_params params_vk = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), NULL, true };
        ggml_context * ctx_vk = ggml_init(params_vk);
        ggml_tensor * experts_vk = nullptr, * out_vk = nullptr;
        ggml_cgraph * gf_vk = make_graph(ctx_vk, &experts_vk, &out_vk);

        if (!ggml_backend_supports_op(backend_vk, out_vk)) {
            printf("not supported [%s]\n", ggml_backend_name(backend_vk));
            ggml_free(ctx_vk); ggml_backend_buffer_free(buf_cpu); ggml_free(ctx_cpu);
            return true;
        }

        ggml_backend_buffer_t buf_vk = ggml_backend_alloc_ctx_tensors(ctx_vk, backend_vk);
        if (!buf_vk) { printf("alloc fail (vk)\n"); ggml_free(ctx_vk); ggml_backend_buffer_free(buf_cpu); ggml_free(ctx_cpu); return false; }

        // Initialize experts with identical random data on both backends
        size_t nbytes = ggml_nbytes(experts_cpu);
        std::vector<float> data(ne0 * n_experts * ne1);
        std::default_random_engine rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto & v : data) v = dist(rng);

        ggml_backend_tensor_set(experts_cpu, data.data(), 0, nbytes);
        ggml_backend_tensor_set(experts_vk,  data.data(), 0, nbytes);

        // Compute
        ggml_backend_graph_compute(backend_cpu, gf_cpu);
        ggml_backend_graph_compute(backend_vk,  gf_vk);
        ggml_backend_synchronize(backend_vk);

        // Compare outputs
        size_t nelements = ggml_nelements(out_cpu);
        std::vector<float> f_cpu(nelements), f_vk(nelements);
        ggml_backend_tensor_get(out_cpu, f_cpu.data(), 0, nelements * sizeof(float));
        ggml_backend_tensor_get(out_vk,  f_vk.data(),  0, nelements * sizeof(float));

        double sum_diff2 = 0, sum_ref2 = 0;
        for (size_t i = 0; i < nelements; i++) {
            if (std::isnan(f_cpu[i]) || std::isnan(f_vk[i])) {
                printf("NaN at %zu\n\033[1;31mFAIL\033[0m\n", i);
                ggml_backend_buffer_free(buf_vk); ggml_free(ctx_vk);
                ggml_backend_buffer_free(buf_cpu); ggml_free(ctx_cpu);
                return false;
            }
            double d = (double)f_cpu[i] - (double)f_vk[i];
            sum_diff2 += d * d;
            sum_ref2  += (double)f_cpu[i] * (double)f_cpu[i];
        }
        double nmse = (sum_ref2 > 0) ? sum_diff2 / sum_ref2 : sum_diff2;

        ggml_backend_buffer_free(buf_vk); ggml_free(ctx_vk);
        ggml_backend_buffer_free(buf_cpu); ggml_free(ctx_cpu);

        if (nmse > 1e-6) {
            printf("NMSE = %.9f > 1e-6 \033[1;31mFAIL\033[0m\n", nmse);
            return false;
        }
        printf("\033[1;32mOK\033[0m (NMSE=%.2e)\n", nmse);
        return true;
    }
};

// GGML_OP_SQR
struct test_sqr : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_sqr(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_sqr(ctx, a);
        return out;
    }
};

// GGML_OP_SQRT
struct test_sqrt : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_sqrt(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_sqrt(ctx, a);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        // fill with positive values
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, 0.0f, 100.0f);
        }
    }
};

// GGML_OP_CLAMP
struct test_clamp : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    float min;
    float max;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, min, max);
    }

    test_clamp(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            float min = -0.5f, float max = 0.5f)
        : type(type), ne(ne), min(min), max(max) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_clamp(ctx, a, min, max);
        return out;
    }
};

// GGML_OP_DIAG_MASK_INF
struct test_diag_mask_inf : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int n_past;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, n_past);
    }

    test_diag_mask_inf(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            int n_past = 5)
        : type(type), ne(ne), n_past(n_past) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_diag_mask_inf(ctx, a, n_past);
        return out;
    }
};

// GGML_OP_SOFT_MAX
struct test_soft_max : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const bool mask;
    const ggml_type mask_type;
    const float scale;
    const float max_bias;

    std::string vars() override {
        return VARS_TO_STR6(type, ne, mask, mask_type, scale, max_bias);
    }

    // the 1024 test with bias occasionally fails:
    // SOFT_MAX(type=f32,ne=[1024,16,1,1],mask=1,scale=1.000000,max_bias=8.000000): [SOFT_MAX] NMSE = 0.000000103 > 0.000000100 FAIL
    virtual double max_nmse_err() override {
        return 1e-6;
    }

    test_soft_max(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10},
            bool mask = false,
            float scale = 1.0f,
            float max_bias = 0.0f,
            ggml_type mask_type = GGML_TYPE_F32)
        : type(type), ne(ne), mask(mask), mask_type(mask_type), scale(scale), max_bias(max_bias) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * mask = nullptr;
        if (this->mask) {
            mask = ggml_new_tensor_2d(ctx, mask_type, ne[0], ne[1]);
        }
        ggml_tensor * out = ggml_soft_max_ext(ctx, a, mask, scale, max_bias);
        return out;
    }
};


// GGML_OP_ROPE
struct test_rope : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    int n_dims;
    int mode;
    int n_ctx; // used to generate positions
    float fs; // freq_scale
    float ef; // ext_factor
    float af; // attn_factor
    bool ff;
    int v; // view (1 : non-contiguous a)

    std::string vars() override {
        return VARS_TO_STR10(type, ne_a, n_dims, mode, n_ctx, fs, ef, af, ff, v);
    }

    test_rope(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {10, 10, 10, 1},
            int n_dims = 10, int mode = 0, int n_ctx = 512, float fs = 1.0f, float ef = 0.0f, float af = 0.0f, bool ff = false, int v = 0)
        : type(type), ne_a(ne_a), n_dims(n_dims), mode(mode), n_ctx(n_ctx), fs(fs), ef(ef), af(af), ff(ff), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 2; ne[1] *= 4; ne[2] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        }
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, ne_a[2]);
        ggml_tensor * freq = ff ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_dims/2) : nullptr;
        ggml_tensor * out = ggml_rope_ext(ctx, a, pos, freq, n_dims, mode, 0, 10000.0f, fs, ef, af, 1.0f, 1.0f);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // pos
                std::vector<int> data(ne_a[2]);
                for (int i = 0; i < ne_a[2]; i++) {
                    data[i] = rand() % n_ctx;
                }
                ggml_backend_tensor_set(t, data.data(), 0, ne_a[2] * sizeof(int));
            } else {
                if (t->ne[0] == n_dims/2) {
                    // frequency factors in the range [0.9f, 1.1f]
                    init_tensor_uniform(t, 0.9f, 1.1f);
                } else {
                    init_tensor_uniform(t);
                }
            }
        }
    }
};

// GGML_OP_POOL2D
struct test_pool2d : public test_case {
    enum ggml_op_pool pool_type;
    const ggml_type type_input;
    const std::array<int64_t, 4> ne_input;
    // kernel size
    const int k0;
    const int k1;
    // stride
    const int s0;
    const int s1;
    // padding
    const int p0;
    const int p1;

    std::string vars() override {
        return VARS_TO_STR9(pool_type, type_input, ne_input, k0, k1, s0, s1, p0, p1);
    }

    test_pool2d(ggml_op_pool pool_type = GGML_OP_POOL_AVG,
            ggml_type type_input = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_input = {10, 10, 3, 1}, // [input_width, input_height, input_channels, 1]
            int k0 = 3, int k1 = 3,
            int s0 = 1, int s1 = 1,
            int p0 = 1, int p1 = 1)
        : pool_type(pool_type), type_input(type_input), ne_input(ne_input), k0(k0), k1(k1), s0(s0), s1(s1), p0(p0), p1(p1) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, type_input, 4, ne_input.data());
        ggml_tensor * out = ggml_pool_2d(ctx, input, pool_type, k0, k1, s0, s1, p0, p1);
        return out;
    }
};

// GGML_OP_CONV_TRANSPOSE_1D
struct test_conv_transpose_1d : public test_case {
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;

    const int s0; // stride
    const int p0; // padding
    const int d0; // dilation

    std::string vars() override {
        return VARS_TO_STR5(ne_input, ne_kernel, s0, p0, d0);
    }

    test_conv_transpose_1d(std::array<int64_t, 4> ne_input = {197, 32, 1, 1}, // [input_width, input_height, input_channels, 1]
                           std::array<int64_t, 4> ne_kernel = {16, 32, 32, 1}, // [kernel_width, kernel_height, input_channels, 1]
                           int s0 = 1, int p0 = 0, int d0 = 1)
        : ne_input(ne_input), ne_kernel(ne_kernel), s0(s0), p0(p0), d0(d0) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_input.data());
        ggml_tensor * kernel = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_kernel.data());
        ggml_tensor * out = ggml_conv_transpose_1d(ctx, kernel, input, s0, p0, d0);
        return out;
    }
};

// GGML_OP_IM2COL
struct test_im2col : public test_case {
    const ggml_type type_input;
    const ggml_type type_kernel;
    const ggml_type dst_type;
    const std::array<int64_t, 4> ne_input;
    const std::array<int64_t, 4> ne_kernel;
    // stride
    const int s0;
    const int s1;
    // padding
    const int p0;
    const int p1;
    // dilation
    const int d0;
    const int d1;
    // mode
    const bool is_2D;

    std::string vars() override {
        return VARS_TO_STR12(type_input, type_kernel, dst_type, ne_input, ne_kernel, s0, s1, p0, p1, d0, d1, is_2D);
    }

    test_im2col(ggml_type type_input = GGML_TYPE_F32, ggml_type type_kernel = GGML_TYPE_F16, ggml_type dst_type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_input = {10, 10, 3, 1}, // [input_width, input_height, input_channels, 1]
            std::array<int64_t, 4> ne_kernel = {3, 3, 3, 1}, // [kernel_width, kernel_height, input_channels, 1]
            int s0 = 1, int s1 = 1,
            int p0 = 1, int p1 = 1,
            int d0 = 1, int d1 = 1,
            bool is_2D = true)
        : type_input(type_input), type_kernel(type_kernel), dst_type(dst_type), ne_input(ne_input), ne_kernel(ne_kernel), s0(s0), s1(s1), p0(p0), p1(p1), d0(d0), d1(d1), is_2D(is_2D) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * input = ggml_new_tensor(ctx, type_input, 4, ne_input.data());
        ggml_tensor * kernel = ggml_new_tensor(ctx, type_kernel, 4, ne_kernel.data());
        ggml_tensor * out = ggml_im2col(ctx, kernel, input, s0, s1, p0, p1, d0, d1, is_2D, dst_type);
        return out;
    }
};

// GGML_OP_CONCAT
struct test_concat : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int64_t ne_b_d;
    const int dim;
    const int v; // view (1 << 0: non-cont a, 1 << 1: non-cont b)

    std::string vars() override {
        return VARS_TO_STR5(type, ne_a, ne_b_d, dim, v);
    }

    test_concat(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {10, 10, 10, 10},
            int64_t ne_b_d = 10,
            int dim = 2, int v = 0)
        : type(type), ne_a(ne_a), ne_b_d(ne_b_d), dim(dim), v(v) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        auto ne_b = ne_a;
        ne_b[dim] = ne_b_d;
        ggml_tensor * a;
        if (v & 1) {
            auto ne = ne_a; ne[0] *= 2; ne[1] *= 4; ne[2] *= 3;
            a = ggml_new_tensor(ctx, type, 4, ne.data());
            a = ggml_view_4d(ctx, a, ne_a[0], ne_a[1], ne_a[2], ne_a[3], a->nb[1], a->nb[2], a->nb[3], 0);
        } else {
            a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        }
        ggml_tensor * b;
        if (v & 2) {
            auto ne = ne_b; ne[0] *= 3; ne[1] *= 2; ne[2] *= 4;
            b = ggml_new_tensor(ctx, type, 4, ne.data());
            b = ggml_view_4d(ctx, b, ne_b[0], ne_b[1], ne_b[2], ne_b[3], b->nb[1], b->nb[2], b->nb[3], 0);
        } else {
            b = ggml_new_tensor(ctx, type, 4, ne_b.data());
        }
        ggml_tensor * out = ggml_concat(ctx, a, b, dim);
        return out;
    }
};

// GGML_OP_CONCAT chain — multiple sequential concats along the same
// dim, mirroring the MTP chained-rollout output stacking pattern.
// This was isolated as a potential trigger for the Vulkan RADV heap
// corruption observed at LLAMA_MTP_ROLLOUT>=2.
struct test_concat_chain : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_single;
    const int dim;
    const int chain_len;

    std::string vars() override {
        return VARS_TO_STR4(type, ne_single, dim, chain_len);
    }

    test_concat_chain(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_single = {32768, 8, 1, 1},
            int dim = 1, int chain_len = 3)
        : type(type), ne_single(ne_single), dim(dim), chain_len(chain_len) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        GGML_ASSERT(chain_len >= 2);
        ggml_tensor * acc = ggml_new_tensor(ctx, type, 4, ne_single.data());
        ggml_set_name(acc, "concat_chain_0");
        for (int i = 1; i < chain_len; ++i) {
            ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne_single.data());
            ggml_set_name(b, (std::string("concat_chain_in_") + std::to_string(i)).c_str());
            acc = ggml_concat(ctx, acc, b, dim);
            ggml_set_name(acc, (std::string("concat_chain_") + std::to_string(i)).c_str());
        }
        return acc;
    }
};

// Argmax → get_rows chain. Exercises the MTP pattern of
// `greedy = ggml_argmax(logits); emb = ggml_get_rows(tok_embd, greedy);`
// across iterations. Validates scheduler preservation of intermediate
// tensors used in subsequent iteration steps.
struct test_argmax_getrows : public test_case {
    const ggml_type logits_type;
    const ggml_type embd_type;
    const int64_t vocab;
    const int64_t n_tokens;
    const int64_t embd_dim;

    std::string vars() override {
        return VARS_TO_STR5(logits_type, embd_type, vocab, n_tokens, embd_dim);
    }

    test_argmax_getrows(ggml_type logits_type = GGML_TYPE_F32,
            ggml_type embd_type = GGML_TYPE_F32,
            int64_t vocab = 32768, int64_t n_tokens = 8, int64_t embd_dim = 1024)
        : logits_type(logits_type), embd_type(embd_type),
          vocab(vocab), n_tokens(n_tokens), embd_dim(embd_dim) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        int64_t logits_ne[4] = { vocab, n_tokens, 1, 1 };
        ggml_tensor * logits = ggml_new_tensor(ctx, logits_type, 4, logits_ne);
        ggml_set_name(logits, "logits");

        ggml_tensor * clamped = ggml_clamp(ctx, logits, -1e4f, 1e4f);
        ggml_set_name(clamped, "clamped");

        ggml_tensor * greedy = ggml_argmax(ctx, clamped);
        ggml_set_name(greedy, "greedy");
        ggml_set_input(greedy);
        ggml_set_output(greedy);

        int64_t embd_ne[4] = { embd_dim, vocab, 1, 1 };
        ggml_tensor * tok_embd = ggml_new_tensor(ctx, embd_type, 4, embd_ne);
        ggml_set_name(tok_embd, "tok_embd");

        ggml_tensor * out = ggml_get_rows(ctx, tok_embd, greedy);
        ggml_set_name(out, "emb_out");
        return out;
    }
};

// GGML_OP_ARGSORT
struct test_argsort : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    ggml_sort_order order;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, order);
    }

    test_argsort(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {16, 10, 10, 10},
            ggml_sort_order order = GGML_SORT_ORDER_ASC)
        : type(type), ne(ne), order(order) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_argsort(ctx, a, order);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // indices
                std::vector<int> data(ggml_nelements(t));
                for (int i = 0; i < ggml_nelements(t); i++) {
                    data[i] = rand();
                }
                std::shuffle(data.begin(), data.end(), rng);
                ggml_backend_tensor_set(t, data.data(), 0, ne[0]*ne[1]*ne[2]*ne[3] * sizeof(int));
            } else if (t->type == GGML_TYPE_F32) {
                // initialize with unique values to avoid ties
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::vector<float> data(t->ne[0]);
                    for (int i = 0; i < t->ne[0]; i++) {
                        data[i] = i;
                    }
                    std::shuffle(data.begin(), data.end(), rng);
                    ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(float));
                }
            } else {
                GGML_ABORT("fatal error");
            }
        }
    }
};

// GGML_OP_SUM_ROWS
struct test_sum_rows : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;

    std::string vars() override {
        return VARS_TO_STR2(type, ne);
    }

    test_sum_rows(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {10, 10, 10, 10})
        : type(type), ne(ne) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_sum_rows(ctx, a);
        return out;
    }
};

// GGML_OP_UPSCALE
struct test_upscale : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int32_t scale_factor;
    const bool transpose;

    std::string vars() override {
        return VARS_TO_STR4(type, ne, scale_factor, transpose);
    }

    test_upscale(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {512, 512, 3, 1},
            int32_t scale_factor = 2, bool transpose = false)
        : type(type), ne(ne), scale_factor(scale_factor), transpose(transpose) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        if (transpose) a = ggml_transpose(ctx, a);
        ggml_tensor * out = ggml_upscale(ctx, a, scale_factor, GGML_SCALE_MODE_NEAREST);
        return out;
    }
};

// GGML_OP_UPSCALE (ext)
struct test_upscale_ext : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const std::array<int64_t, 4> ne_tgt;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, ne_tgt);
    }

    test_upscale_ext(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne     = {2, 5,  7, 11},
            std::array<int64_t, 4> ne_tgt = {5, 7, 11, 13})
        : type(type), ne(ne), ne_tgt(ne_tgt) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_upscale_ext(ctx, a, ne_tgt[0], ne_tgt[1], ne_tgt[2], ne_tgt[3], GGML_SCALE_MODE_NEAREST);
        return out;
    }
};

// GGML_OP_GROUP_NORM
struct test_group_norm : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne;
    const int32_t num_groups;
    const float eps;

    std::string vars() override {
        return VARS_TO_STR3(type, ne, num_groups);
    }

    test_group_norm(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne = {64, 64, 320, 1},
            int32_t num_groups = 32,
            float eps = 1e-6f)
        : type(type), ne(ne), num_groups(num_groups), eps(eps) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
        ggml_tensor * out = ggml_group_norm(ctx, a, num_groups, eps);
        return out;
    }
};

// GGML_OP_ACC
struct test_acc : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const std::array<int64_t, 4> ne_b;

    std::string vars() override {
        return VARS_TO_STR3(type, ne_a, ne_b);
    }

    test_acc(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {1024, 577, 1, 1},
            std::array<int64_t, 4> ne_b = {1024, 576, 1, 1})
        : type(type), ne_a(ne_a), ne_b(ne_b) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_tensor * b = ggml_new_tensor(ctx, type, 4, ne_b.data());
        ggml_tensor * out = ggml_acc(ctx, a, b, a->nb[1], a->nb[2], a->nb[3], b->nb[1]);
        return out;
    }
};

// GGML_OP_PAD
struct test_pad : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int pad_0;
    const int pad_1;

    std::string vars() override {
        return VARS_TO_STR4(type, ne_a, pad_0, pad_1);
    }

    test_pad(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {512, 512, 1, 1},
            int pad_0 = 1, int pad_1 = 1)
        : type(type), ne_a(ne_a), pad_0(pad_0), pad_1(pad_1)  {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_tensor * out = ggml_pad(ctx, a, pad_0, pad_1, 0, 0);
        return out;
    }
};

// GGML_OP_ARANGE
struct test_arange : public test_case {
    const ggml_type type;
    const float start;
    const float stop;
    const float step;

    std::string vars() override {
        return VARS_TO_STR4(type, start, stop, step);
    }

    test_arange(ggml_type type = GGML_TYPE_F32,
            float start = 0.f, float stop = 10.f, float step = 1.f)
        : type(type), start(start), stop(stop), step(step)  {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * out = ggml_arange(ctx, start, stop, step);
        return out;
    }
};

// GGML_OP_TIMESTEP_EMBEDDING
struct test_timestep_embedding : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const int dim;
    const int max_period;

    std::string vars() override {
        return VARS_TO_STR4(type, ne_a, dim, max_period);
    }

    test_timestep_embedding(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {2, 1, 1, 1},
            int dim = 320, int max_period=10000)
        : type(type), ne_a(ne_a), dim(dim), max_period(max_period)  {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_tensor * out = ggml_timestep_embedding(ctx, a, dim, max_period);
        return out;
    }
};

// GGML_OP_LEAKY_RELU
struct test_leaky_relu : public test_case {
    const ggml_type type;
    const std::array<int64_t, 4> ne_a;
    const float negative_slope;

    std::string vars() override {
        return VARS_TO_STR3(type, ne_a, negative_slope);
    }

    test_leaky_relu(ggml_type type = GGML_TYPE_F32,
            std::array<int64_t, 4> ne_a = {10, 10, 10, 10},
            float negative_slope = 0.1f)
        : type(type), ne_a(ne_a), negative_slope(negative_slope)  {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne_a.data());
        ggml_tensor * out = ggml_leaky_relu(ctx, a, negative_slope, true);
        return out;
    }
};

// GGML_OP_FLASH_ATTN_EXT
struct test_flash_attn_ext : public test_case {
    const int64_t hs; // head size
    const int64_t nh; // num heads
    const int64_t kv; // kv size
    const int64_t nb; // batch size

    const bool mask; // use mask

    const float max_bias; // ALiBi
    const float softcap;  // Gemma-2

    const ggml_type type_KV;

    std::string vars() override {
        return VARS_TO_STR8(hs, nh, kv, nb, mask, max_bias, softcap, type_KV);
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    test_flash_attn_ext(int64_t hs = 128, int64_t nh = 32, int64_t kv = 96, int64_t nb = 8, bool mask = true, float max_bias = 0.0f, float softcap = 0.0f, ggml_type type_KV = GGML_TYPE_F16)
        : hs(hs), nh(nh), kv(kv), nb(nb), mask(mask), max_bias(max_bias), softcap(softcap), type_KV(type_KV) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t hs_padded = GGML_PAD(hs, ggml_blck_size(type_KV));

        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hs_padded, nb, nh, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, type_KV,       hs_padded, kv, nh, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, type_KV,       hs_padded, kv, nh, 1);
        ggml_tensor * m = mask ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, GGML_PAD(nb, GGML_KQ_MASK_PAD), 1, 1) : nullptr;
        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf(hs), max_bias, softcap);
        return out;
    }
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
};

struct llama_hparams {
    uint32_t n_vocab;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_head_kv;
    static constexpr uint32_t n_layer = 1;
    uint32_t n_rot;
    uint32_t n_embd_head; // dimension of values (d_v)
    uint32_t n_ff;

    float f_norm_eps;
    float f_norm_rms_eps;

    // cparams
    static constexpr uint32_t n_ctx = 512; // user-specified context size
    static constexpr uint32_t n_ctx_orig = n_ctx;

    // batch
    int32_t n_tokens;

    // llm_build_context
    static constexpr int32_t n_kv    = 32; // size of KV cache to consider (n_kv <= n_ctx
    static constexpr int32_t kv_head = 1;  // index of where we store new KV data in the cache

    uint32_t n_embd_gqa() const { // dimension of key embeddings across all k-v heads
        return n_embd_head * n_head_kv;
    }
};

// LLM base class
struct test_llm : public test_case {
    llama_hparams hp;

protected:
    test_llm(llama_hparams hp)
        : hp(std::move(hp)) {
    }

public:
    struct ggml_tensor * llm_build_norm(
            struct ggml_context * ctx,
             struct ggml_tensor * cur,
             struct ggml_tensor * mw,
             struct ggml_tensor * mb,
                  llm_norm_type   type) {
        switch (type) {
            case LLM_NORM:     cur = ggml_norm    (ctx, cur, hp.f_norm_eps); break;
            case LLM_NORM_RMS: cur = ggml_rms_norm(ctx, cur, hp.f_norm_rms_eps); break;
        }
        cur = ggml_mul(ctx, cur, mw);
        if (mb) {
            cur = ggml_add(ctx, cur, mb);
        }
        return cur;
    }

    void llm_build_kv_store(
            struct ggml_context * ctx,
             struct ggml_tensor * k_l,
             struct ggml_tensor * v_l,
             struct ggml_tensor * k_cur,
             struct ggml_tensor * v_cur) {
        // compute the transposed [n_tokens, n_embd] V matrix
        struct ggml_tensor * v_cur_t = ggml_transpose(ctx, ggml_reshape_2d(ctx, v_cur, hp.n_embd_gqa(), hp.n_tokens));

        struct ggml_tensor * k_cache_view = ggml_view_1d(ctx, k_l, hp.n_tokens*hp.n_embd_gqa(),
                (ggml_row_size(k_l->type, hp.n_embd_gqa()))*hp.kv_head);

        struct ggml_tensor * v_cache_view = ggml_view_2d(ctx, v_l, hp.n_tokens, hp.n_embd_gqa(),
                (  hp.n_ctx)*ggml_element_size(v_l),
                (hp.kv_head)*ggml_element_size(v_l));

        // important: storing RoPE-ed version of K in the KV cache!
        ggml_cpy(ctx, k_cur,   k_cache_view);
        ggml_cpy(ctx, v_cur_t, v_cache_view);
    }

    struct ggml_tensor * llm_build_kqv(
            struct ggml_context * ctx,
             struct ggml_tensor * k_l,
             struct ggml_tensor * v_l,
             struct ggml_tensor * q_cur,
             struct ggml_tensor * kq_mask,
                        float     kq_scale) {
        struct ggml_tensor * q = ggml_permute(ctx, q_cur, 0, 2, 1, 3);

        struct ggml_tensor * k =
            ggml_view_3d(ctx, k_l,
                    hp.n_embd_head, hp.n_kv, hp.n_head_kv,
                    ggml_row_size(k_l->type, hp.n_embd_gqa()),
                    ggml_row_size(k_l->type, hp.n_embd_head),
                    0);

        struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q);

        kq = ggml_soft_max_ext(ctx, kq, kq_mask, kq_scale, 0.0f);

        // split cached v into n_head heads
        struct ggml_tensor * v =
            ggml_view_3d(ctx, v_l,
                    hp.n_kv, hp.n_embd_head, hp.n_head_kv,
                    ggml_element_size(v_l)*hp.n_ctx,
                    ggml_element_size(v_l)*hp.n_ctx*hp.n_embd_head,
                    0);

        struct ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);

        struct ggml_tensor * kqv_merged = ggml_permute(ctx, kqv, 0, 2, 1, 3);

        struct ggml_tensor * cur = ggml_cont_2d(ctx, kqv_merged, hp.n_embd_head*hp.n_head, hp.n_tokens);

        struct ggml_tensor * wo = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd);
        cur = ggml_mul_mat(ctx, wo, cur);

        return cur;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // pos
                std::vector<int> data(hp.n_tokens);
                for (int i = 0; i < hp.n_tokens; i++) {
                    data[i] = rand() % hp.n_ctx;
                }
                ggml_backend_tensor_set(t, data.data(), 0, hp.n_tokens * sizeof(int));
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// Llama
struct test_llama : public test_llm {
    static constexpr float freq_base = 10000.0f;
    static constexpr float freq_scale = 1.0f;
    static constexpr float ext_factor = 0.0f;
    static constexpr float attn_factor = 1.0f;
    static constexpr float beta_fast = 32.0f;
    static constexpr float beta_slow = 1.0f;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "LLAMA";
    }

    std::string vars() override {
        auto n_tokens = hp.n_tokens;
        return VARS_TO_STR1(n_tokens);
    }

    double max_nmse_err() override {
        return 2e-3;
    }

    test_llama(int n_tokens = 1)
        : test_llm({
            /*n_vocab        =*/ 32000,
            /*n_embd         =*/ 3200,
            /*n_head         =*/ 32,
            /*n_head_kv      =*/ 32,
            /*n_rot          =*/ 100,
            /*n_embd_head    =*/ 100,
            /*n_ff           =*/ 8640,
            /*f_norm_eps     =*/ 0.f,
            /*f_norm_rms_eps =*/ 1e-5f,
            /*n_tokens       =*/ n_tokens,
        }) {
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, hp.n_tokens);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, hp.n_tokens);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hp.n_kv, hp.n_tokens, 1);

        ggml_tensor * k_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);
        ggml_tensor * v_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);

        for (uint32_t il = 0; il < hp.n_layer; ++il) {
            struct ggml_tensor * inpSA = inpL;

            // norm
            ggml_tensor * attn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            cur = llm_build_norm(ctx, inpL, attn_norm, nullptr, LLM_NORM_RMS);

            // self-attention
            {
                ggml_tensor * wq = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd);
                ggml_tensor * wk = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd_gqa());
                ggml_tensor * wv = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd_gqa());

                // compute Q and K and RoPE them
                struct ggml_tensor * Qcur = ggml_mul_mat(ctx, wq, cur);
                struct ggml_tensor * Kcur = ggml_mul_mat(ctx, wk, cur);
                struct ggml_tensor * Vcur = ggml_mul_mat(ctx, wv, cur);

                Qcur = ggml_rope_ext(
                    ctx, ggml_reshape_3d(ctx, Qcur, hp.n_embd_head, hp.n_head,    hp.n_tokens), inp_pos, nullptr,
                    hp.n_rot, 0, hp.n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );

                Kcur = ggml_rope_ext(
                    ctx, ggml_reshape_3d(ctx, Kcur, hp.n_embd_head, hp.n_head_kv, hp.n_tokens), inp_pos, nullptr,
                    hp.n_rot, 0, hp.n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                );

                llm_build_kv_store(ctx, k_l, v_l, Kcur, Vcur);

                cur = llm_build_kqv(ctx, k_l, v_l, Qcur, KQ_mask, 1.0f/sqrtf(float(hp.n_embd_head)));
            }

            struct ggml_tensor * ffn_inp = ggml_add(ctx, cur, inpSA);

            // feed-forward network
            ggml_tensor * ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            cur = llm_build_norm(ctx, ffn_inp, ffn_norm, nullptr, LLM_NORM_RMS);

            ggml_tensor * ffn_gate = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
            ggml_tensor * ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_ff,   hp.n_embd);
            ggml_tensor * ffn_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
            struct ggml_tensor * tmp = ggml_mul_mat(ctx, ffn_up, cur);
            cur = ggml_mul_mat(ctx, ffn_gate, cur);
            cur = ggml_silu(ctx, cur);
            cur = ggml_mul(ctx, cur, tmp);
            cur = ggml_mul_mat(ctx, ffn_down, cur);

            cur = ggml_add(ctx, cur, ffn_inp);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        ggml_tensor * output_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
        cur = llm_build_norm(ctx, cur, output_norm, nullptr, LLM_NORM_RMS);

        // lm_head
        ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_vocab);
        cur = ggml_mul_mat(ctx, output, cur);

        return cur;
    }
};

// Falcon
struct test_falcon : public test_llm {
    static constexpr float freq_base = 10000.0f;
    static constexpr float freq_scale = 1.0f;
    static constexpr float ext_factor = 0.0f;
    static constexpr float attn_factor = 1.0f;
    static constexpr float beta_fast = 32.0f;
    static constexpr float beta_slow = 1.0f;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "FALCON";
    }

    std::string vars() override {
        auto n_tokens = hp.n_tokens;
        return VARS_TO_STR1(n_tokens);
    }

    double max_nmse_err() override {
        return 2e-3;
    }

    test_falcon(int n_tokens = 1)
        : test_llm({
            /*n_vocab        =*/ 32000,
            /*n_embd         =*/ 3200,
            /*n_head         =*/ 50,
            /*n_head_kv      =*/ 1,
            /*n_rot          =*/ 64,
            /*n_embd_head    =*/ 64,
            /*n_ff           =*/ 8640,
            /*f_norm_eps     =*/ 1e-5f,
            /*f_norm_rms_eps =*/ 0.f,
            /*n_tokens       =*/ n_tokens,
        }) {
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        struct ggml_tensor * cur;
        struct ggml_tensor * inpL;

        inpL = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.n_embd, hp.n_tokens);

        // inp_pos - contains the positions
        struct ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, hp.n_tokens);

        // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
        struct ggml_tensor * KQ_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, hp.n_kv, hp.n_tokens, 1);

        ggml_tensor * k_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);
        ggml_tensor * v_l = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 1638400);

        for (uint32_t il = 0; il < hp.n_layer; ++il) {
            // norm
            ggml_tensor * attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            ggml_tensor * attn_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
            ggml_tensor * attn_norm = llm_build_norm(ctx, inpL, attn_norm_w, attn_norm_b, LLM_NORM);

            // self-attention
            {
                cur = attn_norm;

                ggml_tensor * wqkv = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_embd + 2*hp.n_embd_gqa());

                cur = ggml_mul_mat(ctx, wqkv, cur);

                struct ggml_tensor * Qcur = ggml_cont(ctx, ggml_view_2d(ctx, cur, hp.n_embd,     hp.n_tokens, cur->nb[1], 0*sizeof(float)*(hp.n_embd)));
                struct ggml_tensor * Kcur = ggml_cont(ctx, ggml_view_2d(ctx, cur, hp.n_embd_gqa(), hp.n_tokens, cur->nb[1], 1*sizeof(float)*(hp.n_embd)));
                struct ggml_tensor * Vcur = ggml_cont(ctx, ggml_view_2d(ctx, cur, hp.n_embd_gqa(), hp.n_tokens, cur->nb[1], 1*sizeof(float)*(hp.n_embd + hp.n_embd_gqa())));

                Qcur = ggml_reshape_3d(ctx, Qcur, hp.n_embd_head, hp.n_head,    hp.n_tokens);
                Kcur = ggml_reshape_3d(ctx, Kcur, hp.n_embd_head, hp.n_head_kv, hp.n_tokens);

                // using mode = 2 for neox mode
                Qcur = ggml_rope_ext(
                    ctx, Qcur, inp_pos, nullptr, hp.n_rot, 2, hp.n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );

                Kcur = ggml_rope_ext(
                    ctx, Kcur, inp_pos, nullptr, hp.n_rot, 2, hp.n_ctx_orig,
                    freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow
                );

                llm_build_kv_store(ctx, k_l, v_l, Kcur, Vcur);

                cur = llm_build_kqv(ctx, k_l, v_l, Qcur, KQ_mask, 1.0f/sqrtf(float(hp.n_embd_head)));
            }

            struct ggml_tensor * ffn_inp = cur;

            // feed forward
            {
                ggml_tensor * ffn_up   = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_embd, hp.n_ff);
                ggml_tensor * ffn_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, hp.n_ff, hp.n_embd);
                cur = attn_norm;
                cur = ggml_mul_mat(ctx, ffn_up, cur);
                cur = ggml_gelu(ctx, cur);
                cur = ggml_mul_mat(ctx, ffn_down, cur);
            }

            cur = ggml_add(ctx, cur, ffn_inp);

            cur = ggml_add(ctx, cur, inpL);

            // input for next layer
            inpL = cur;
        }

        cur = inpL;

        ggml_tensor * output_norm   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
        ggml_tensor * output_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hp.n_embd);
        cur = llm_build_norm(ctx, cur, output_norm, output_norm_b, LLM_NORM);

        // lm_head
        ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, hp.n_embd, hp.n_vocab);
        cur = ggml_mul_mat(ctx, output, cur);

        return cur;
    }
};

static bool test_backend(ggml_backend_t backend, test_mode mode, const char * op_name) {
    std::vector<std::unique_ptr<test_case>> test_cases;
    std::default_random_engine rng(0);

    const ggml_type all_types[] = {
        GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
        GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
        GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS,
        GGML_TYPE_TURBO_KV_4B,
    };

    const ggml_type base_types[] = {
        GGML_TYPE_F32, GGML_TYPE_F16,
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_K,
        GGML_TYPE_IQ2_XXS
    };

    const ggml_type other_types[] = {
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K, GGML_TYPE_Q3_K,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
        GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M,
        GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS,
        GGML_TYPE_BF16,
    };

    // unary ops
    for (int v : {0, 1}) {
        for (int op = 0; op < GGML_UNARY_OP_COUNT; op++) {
            test_cases.emplace_back(new test_unary((ggml_unary_op) op, GGML_TYPE_F32, { 128, 10, 10, 10 }, v));
            test_cases.emplace_back(new test_unary((ggml_unary_op) op, GGML_TYPE_F32, { 7, 13, 19, 23 }, v));
        }
    }

    test_cases.emplace_back(new test_get_rows(GGML_TYPE_F32, 1, 8, 2, 1, false));
    for (ggml_type type : all_types) {
        for (int b : {1, 7}) {
            for (bool v : {false, true}) {
                test_cases.emplace_back(new test_get_rows(type, 256, 5, 4, b, v));
            }
        }
    }
    for (int b : {1, 7}) {
        for (bool v : {false, true}) {
            test_cases.emplace_back(new test_get_rows(GGML_TYPE_I32, 256, 5, 4, b, v));
        }
    }

    for (ggml_type type_input : {GGML_TYPE_F32}) {
        for (ggml_op_pool pool_type : {GGML_OP_POOL_AVG, GGML_OP_POOL_MAX}) {
            for (int k0 : {1, 3}) {
                for (int k1 : {1, 3}) {
                    for (int s0 : {1, 2}) {
                        for (int s1 : {1, 2}) {
                            for (int p0 : {0, 1}) {
                                for (int p1 : {0, 1}) {
                                    test_cases.emplace_back(new test_pool2d(pool_type, type_input, {10, 10, 3, 1}, k0, k1, s0, s1, p0, p1));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16));
    // test cases for 1D im2col
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F16, {3000, 128, 1, 1}, {3, 128, 1280, 1}, 1, 0, 1, 0, 1, 0, false));
    test_cases.emplace_back(new test_im2col(GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32, {3000, 128, 1, 1}, {3, 128, 1280, 1}, 1, 0, 1, 0, 1, 0, false));

    test_cases.emplace_back(new test_conv_transpose_1d());
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {2,3,2,1}, 3, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {2,3,2,1}, 2, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {2,3,2,1}, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {3,2,2,1}, 2, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {3,2,2,1}, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {3,1,2,1}, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({2,1,1,1}, {3,1,1,1}, 1, 0, 1));


    test_cases.emplace_back(new test_conv_transpose_1d());
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {2,3,2,1}, 3, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {2,3,2,1}, 2, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {2,3,2,1}, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {3,2,2,1}, 2, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {3,2,2,1}, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({3,2,1,1}, {3,1,2,1}, 1, 0, 1));
    test_cases.emplace_back(new test_conv_transpose_1d({2,1,1,1}, {3,1,1,1}, 1, 0, 1));


    test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 1, 1, 1}));
    test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 10, 10, 10}, {2, 1, 1, 1}));
    test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 2, 1, 1}));
    test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 1, 2, 1}));
    test_cases.emplace_back(new test_repeat(GGML_TYPE_F32, {10, 10, 10, 10}, {1, 1, 1, 2}));
    test_cases.emplace_back(new test_repeat(GGML_TYPE_I32, {10, 10, 10, 10}, {2, 1, 1, 1}));
    test_cases.emplace_back(new test_repeat(GGML_TYPE_I16, {10, 10, 10, 10}, {1, 1, 1, 2}));

    test_cases.emplace_back(new test_dup(GGML_TYPE_F32));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I32));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F32, {10, 10, 5, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16, {10, 10, 5, 1}, {0, 2, 1, 3})); // dup by rows
    test_cases.emplace_back(new test_dup(GGML_TYPE_F32, {10, 10, 5, 1}, {1, 0, 2, 3}));
    test_cases.emplace_back(new test_dup(GGML_TYPE_F16, {10, 10, 5, 1}, {1, 0, 2, 3})); // dup dst not-contiguous
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16, {10, 8, 3, 1}, {0, 2, 1, 3}));
    test_cases.emplace_back(new test_dup(GGML_TYPE_I16, {10, 8, 3, 1}, {1, 2, 0, 3}));

    for (ggml_type type_src : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (ggml_type type_dst : all_types) {
           test_cases.emplace_back(new test_cpy(type_src, type_dst, {256, 4, 4, 4}));
           test_cases.emplace_back(new test_cpy(type_src, type_dst, {256, 2, 3, 4}, {0, 2, 1, 3})); // cpy by rows
        }
    }
    for (ggml_type type_src : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (ggml_type type_dst : {GGML_TYPE_F16, GGML_TYPE_F32}) {
            test_cases.emplace_back(new test_cpy(type_src, type_dst, {256, 2, 3, 4}, {1, 0, 2, 3})); // cpy not-contiguous
        }
    }

    test_cases.emplace_back(new test_cont());
    // CONT: F16 and Nemotron-relevant shapes
    test_cases.emplace_back(new test_cont(GGML_TYPE_F16, {10, 10, 10, 1}));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F32, {128, 32, 1, 1}));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F16, {128, 32, 1, 1}));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F32, {3072, 4, 1, 1}));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F16, {3072, 4, 1, 1}));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F32, {128, 1, 1, 1}));
    test_cases.emplace_back(new test_cont(GGML_TYPE_F32, {5120, 32, 1, 1}));

    auto add_test_bin_bcast = [&](ggml_type type, std::array<int64_t, 4> ne, std::array<int, 4> nr) {
        for (auto op : {ggml_add, ggml_mul, ggml_div}) {
            test_cases.emplace_back(new test_bin_bcast(op, type, ne, nr));
        }
    };

    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 8, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 1, 1}, {32, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 320, 320}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 1, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {2, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 2, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 1, 2, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 1, 1, 2});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 1, 2, 2});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {1, 2, 2, 2});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 10, 10, 10}, {2, 2, 2, 2});

    // stable diffusion
    add_test_bin_bcast(GGML_TYPE_F32, {1280, 1, 1, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1280, 1, 1, 1}, {1, 16, 16, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1280, 16, 16, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1280, 1, 1, 1}, {1, 256, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 1280, 1}, {16, 16, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {16, 16, 1280, 1}, {1, 1, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 1920, 1}, {16, 16, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 2560, 1}, {16, 16, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 1280, 1}, {32, 32, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 1920, 1}, {32, 32, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {1, 1, 640, 1}, {32, 32, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {5120, 1, 1, 1}, {1, 256, 1, 1});
    add_test_bin_bcast(GGML_TYPE_F32, {640, 1, 1, 1}, {1, 1, 1, 1});
    //add_test_bin_bcast(GGML_TYPE_F32, {3, 3, 2560, 1280}, {1, 1, 1, 1});
    //add_test_bin_bcast(GGML_TYPE_F32, {3, 3, 2560, 1280}, {2, 1, 1, 1});

    test_cases.emplace_back(new test_scale());

    for (float eps : {1e-6f, 1e-5f, 1e-3f, 1e-1f}) {
        test_cases.emplace_back(new test_norm(GGML_TYPE_F32, {64, 10, 10, 10}, eps));
        test_cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, {64, 10, 10, 10}, eps));
        test_cases.emplace_back(new test_fused_rms_norm(GGML_TYPE_F32, {64, 10, 10, 10}, eps));
    }
    // L2_NORM: contiguous and strided-view (q_fused/k_fused on Qwen3.5)
    for (float eps : {1e-12f, 1e-7f, 1e-3f}) {
        test_cases.emplace_back(new test_l2_norm(GGML_TYPE_F32, {64, 5, 4, 3}, eps, false));
        test_cases.emplace_back(new test_l2_norm(GGML_TYPE_F32, {64, 5, 4, 3}, eps, true));
    }
    // SSM_CONV single-sequence (Mamba/SSM hot path). Covers Qwen3.5-A3B
    // (d_conv=4, d_inner=4096) decode/prefill plus smaller / general nc
    // shapes for fast iteration and code coverage.
    test_cases.emplace_back(new test_ssm_conv(4, 4096, 1));    // Qwen3.5 decode
    test_cases.emplace_back(new test_ssm_conv(4, 4096, 8));    // small batch
    test_cases.emplace_back(new test_ssm_conv(4, 4096, 512));  // typical prefill
    test_cases.emplace_back(new test_ssm_conv(4, 128,  4));    // fast iteration
    test_cases.emplace_back(new test_ssm_conv(8, 64,   4));    // general nc path
    test_cases.emplace_back(new test_ssm_conv(4, 64,   1));    // degenerate
    test_cases.emplace_back(new test_ssm_conv(4, 4,    1));    // single thread
    // SSM_CONV multi-sequence — slow path (init + serial-over-tokens
    // per-row). Covers unique-seq, recurrent (same seq, multiple tokens),
    // and multi-target fanout (one token writes state to multiple seqs).
    test_cases.emplace_back(new test_ssm_conv(4, 256, 4,  4, 4, 0)); // multi-seq unique nc4
    test_cases.emplace_back(new test_ssm_conv(4, 256, 16, 4, 4, 0)); // multi-seq unique nc4 larger batch
    test_cases.emplace_back(new test_ssm_conv(4, 256, 8,  4, 4, 1)); // multi-seq recurrent nc4
    test_cases.emplace_back(new test_ssm_conv(4, 256, 4,  4, 4, 2)); // multi-seq fanout nc4
    test_cases.emplace_back(new test_ssm_conv(8, 64,  4,  4, 4, 0)); // multi-seq unique general nc
    test_cases.emplace_back(new test_ssm_conv(8, 64,  4,  4, 4, 1)); // multi-seq recurrent general nc

    // DELTA_NET — Qwen3-Next / Qwen3.5-A3B recurrent linear-attention.
    // Coverage matrix: head_dim ∈ {64, 128} × n_tokens × GQA × repeat_type.
    // The Qwen3-Next-shaped cases (head_dim=128, H_v=32, gqa=4) are the
    // primary correctness target.
    test_cases.emplace_back(new test_delta_net( 64,   1,  2, 1, 1, 0)); // smallest tg
    test_cases.emplace_back(new test_delta_net( 64,   1,  8, 2, 1, 0)); // GQA tiled tg
    test_cases.emplace_back(new test_delta_net( 64,   1,  8, 2, 1, 1)); // GQA interleaved tg
    test_cases.emplace_back(new test_delta_net( 64,   4,  8, 1, 1, 0)); // small pp
    test_cases.emplace_back(new test_delta_net( 64,  64,  8, 1, 1, 0)); // medium pp
    test_cases.emplace_back(new test_delta_net( 64,  64,  8, 4, 1, 0)); // GQA stress tiled
    test_cases.emplace_back(new test_delta_net( 64,  64,  8, 4, 1, 1)); // GQA stress interleaved
    test_cases.emplace_back(new test_delta_net( 64,  64,  8, 1, 2, 0)); // multi-seq
    test_cases.emplace_back(new test_delta_net(128,   1,  2, 1, 1, 0)); // h128 baseline
    test_cases.emplace_back(new test_delta_net(128,   1, 32, 4, 1, 0)); // Qwen3-Next tg tiled
    test_cases.emplace_back(new test_delta_net(128,   1, 32, 4, 1, 1)); // Qwen3-Next tg interleaved
    test_cases.emplace_back(new test_delta_net(128,   8, 32, 4, 1, 0)); // small Qwen3-Next pp
    test_cases.emplace_back(new test_delta_net(128,  64, 32, 4, 1, 0)); // medium Qwen3-Next pp
    test_cases.emplace_back(new test_delta_net(128, 256, 32, 4, 1, 0)); // full Qwen3-Next pp tiled
    test_cases.emplace_back(new test_delta_net(128, 256, 32, 4, 1, 1)); // full Qwen3-Next pp interleaved
    test_cases.emplace_back(new test_delta_net(128, 256, 32, 4, 2, 0)); // multi-seq Qwen3-Next pp
    test_cases.emplace_back(new test_delta_net(128, 256, 32, 4, 2, 1)); // multi-seq interleaved
    test_cases.emplace_back(new test_delta_net(128,   1,  1, 1, 1, 0)); // degenerate single head
    test_cases.emplace_back(new test_delta_net(128,   4,  4, 2, 2, 1)); // small mixed
    // FUSED_MUL_UNARY with scalar broadcast (Qwen3.5 shared expert
    // single-token decode pattern: shape [1] gate × [n_ff] feature output).
    // The CPU op only allows SILU/SIGMOID for the broadcast variant.
    test_cases.emplace_back(new test_fused_mul_unary(GGML_TYPE_F32, {2048, 1, 1, 1}, GGML_UNARY_OP_SIGMOID, true));
    test_cases.emplace_back(new test_fused_mul_unary(GGML_TYPE_F32, {2048, 1, 1, 1}, GGML_UNARY_OP_SILU,    true));
    // FUSED_RMS_NORM: additional dimension coverage
    test_cases.emplace_back(new test_fused_rms_norm(GGML_TYPE_F32, {128,  1,  1, 1}));
    test_cases.emplace_back(new test_fused_rms_norm(GGML_TYPE_F32, {3072, 4,  1, 1}));
    test_cases.emplace_back(new test_fused_rms_norm(GGML_TYPE_F32, {5120, 32, 1, 1}));

    // FUSED_MUL_UNARY: activation(a) * b
    for (ggml_unary_op uop : {GGML_UNARY_OP_SILU, GGML_UNARY_OP_GELU, GGML_UNARY_OP_RELU}) {
        test_cases.emplace_back(new test_fused_mul_unary(GGML_TYPE_F32, {128, 10, 10, 1}, uop));
        test_cases.emplace_back(new test_fused_mul_unary(GGML_TYPE_F32, {3072, 4,  1, 1}, uop));
    }
    test_cases.emplace_back(new test_fused_mul_unary(GGML_TYPE_F32, {1, 10, 10, 1}, GGML_UNARY_OP_SILU));
    test_cases.emplace_back(new test_fused_mul_unary(GGML_TYPE_F32, {5120, 1, 1, 1}, GGML_UNARY_OP_SILU));
    // F16 FUSED_MUL_UNARY not tested: CPU backend only supports F32

#if 1
    for (ggml_type type_a : base_types) {
        for (ggml_type type_b : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, { 1,  1}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10,  1}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10,  1}, {2, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10, 10}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10, 10}, {2, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10, 10}, {1, 2}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {10, 10}, {2, 2}));

            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, { 1,  1}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10,  1}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10,  1}, {2, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10, 10}, {1, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10, 10}, {2, 1}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10, 10}, {1, 2}));
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 16, 256, {10, 10}, {2, 2}));
        }
    }
#else
    // m = a rows
    // n = b rows
    // k = cols
    std::uniform_int_distribution<> dist_m(1, 128);
    std::uniform_int_distribution<> dist_n(16, 128);
    std::uniform_int_distribution<> dist_k(1, 16);
    for (int i = 0; i < 1000; i++) {
        for (ggml_type type_a : all_types) {
            for (ggml_type type_b : {GGML_TYPE_F32}) {
                int m = dist_m(rng);
                int n = dist_n(rng);
                int k = dist_k(rng) * ggml_blck_size(type_a);
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, m, n, k, { 1,  1}, {1, 1}));
            }
        }
    }
#endif

    for (ggml_type type_a : other_types) {
        for (ggml_type type_b : {GGML_TYPE_F32}) {
            if (ggml_blck_size(type_a) != 256) {
                test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, ggml_blck_size(type_a), {1,  1}, {1, 1}));
            }
            test_cases.emplace_back(new test_mul_mat(type_a, type_b, 16, 1, 256, {1,  1}, {1, 1}));
        }
    }

    // MUL_MAT stress: realistic activation magnitudes for f16acc overflow
    // detection. B∈[-10,10] and B∈[-50,50] capture what deep residual models
    // produce. n=1 tests mul_mat_vec (tg path); n=8 tests mul_mm (pp path).
    // Both use f16acc on Vega (FLOAT_TYPE/ACC_TYPE = float16_t).
    for (float br : {10.0f, 50.0f}) {
        for (ggml_type ta : {GGML_TYPE_Q8_0,
                             GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K,
                             GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                             GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_S}) {
            test_cases.emplace_back(new test_mul_mat_stress(ta, GGML_TYPE_F32, 2048, 1, 2048, br));
            test_cases.emplace_back(new test_mul_mat_stress(ta, GGML_TYPE_F32, 2048, 8, 2048, br));
        }
    }

    for (ggml_type type_a : {GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q4_K}) {
        // m=vocab_size, n=1 (token gen), k=hidden_dim
        test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 32000, 1, 256, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 128256, 1, 256, {1, 1}, {1, 1}));
        // Realistic K dimensions (hidden_dim)
        test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 128256, 1, 3072, {1, 1}, {1, 1}));
        test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 151936, 1, 1536, {1, 1}, {1, 1}));
        // Prompt eval (n>1)
        test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 128256, 6, 3072, {1, 1}, {1, 1}));
    }

    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  64, 2,  128, { 8,  1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  83, 2,  128, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  64, 2,   64, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  83, 2,   64, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32,  64, 45, 128, { 8,  1}, {4, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F16, GGML_TYPE_F32, 128, 45,  64, { 8,  1}, {4, 1}));

    for (ggml_type type_a : base_types) {
        for (ggml_type type_b : {GGML_TYPE_F32 /*, GGML_TYPE_F16 */}) {
            for (int n_mats : {4, 8}) {
                for (int n_used : {1, 2, 4}) {
                    for (bool b : {false, true}) {
                        for (int n : {1, 32}) {
                            int m = 512;
                            int k = 256;
                            test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k));
                        }
                    }
                }
            }
        }
    }

    for (ggml_type type_a : other_types) {
        for (ggml_type type_b : {GGML_TYPE_F32 /*, GGML_TYPE_F16 */}) {
            for (int n_mats : {4}) {
                for (int n_used : {2}) {
                    for (bool b : {false}) {
                        for (int n : {1}) {
                            int m = 512;
                            int k = 256;
                            test_cases.emplace_back(new test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k));
                        }
                    }
                }
            }
        }
    }

    // GGML_OP_MUL_MULTI_ADD
    test_cases.emplace_back(new test_mul_multi_add(128,  6,  1));  // single token, 6 experts
    test_cases.emplace_back(new test_mul_multi_add(128,  6,  4));  // small batch
    test_cases.emplace_back(new test_mul_multi_add(128,  6, 32));  // larger batch
    test_cases.emplace_back(new test_mul_multi_add(256,  8,  1));  // 8 experts
    test_cases.emplace_back(new test_mul_multi_add(5120, 6,  1));  // GLM hidden dim
    test_cases.emplace_back(new test_mul_multi_add(128,  1,  1));  // single expert, single token (degenerate)
    test_cases.emplace_back(new test_mul_multi_add(128,  2,  1));  // 2 experts
    test_cases.emplace_back(new test_mul_multi_add(128, 16,  4));  // many experts
    test_cases.emplace_back(new test_mul_multi_add(3072, 6,  1));  // Nemotron hidden dim
    test_cases.emplace_back(new test_mul_multi_add(3072, 6,  8));  // Nemotron batch
    test_cases.emplace_back(new test_mul_multi_add(  1,  6,  1));  // minimal ne0
    test_cases.emplace_back(new test_mul_multi_add(128,  6, 64));  // large batch

    // GGML_OP_MULTI_ADD — tested below in custom eval section (uses strided view)

    test_cases.emplace_back(new test_sqr());
    test_cases.emplace_back(new test_sqrt());
    test_cases.emplace_back(new test_clamp());

    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, {10, 10,  1,  1}, 5));
    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, {10, 10, 10,  1}, 5));
    test_cases.emplace_back(new test_diag_mask_inf(GGML_TYPE_F32, {10, 10, 10, 10}, 5));

#if 0
    std::uniform_int_distribution<> dist_ne1(1, 50);
    int exponent = 1;
    while (exponent < (1 << 17)) {
        std::uniform_int_distribution<> dist_ne0(exponent, 2*exponent);

        for (int n = 0; n < 10; ++n) {
            int64_t ne0 = dist_ne0(rng);
            int64_t ne1 = dist_ne1(rng);
            test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, GGML_TYPE_F32, {ne0, ne1, 1, 1}, n/2 == 0, 0.1f, ne0 < 1000 ? 4.0f : 0.0f));
        }

        exponent <<= 1;
    }
#endif
    for (bool mask : {false, true}) {
        for (float max_bias : {0.0f, 8.0f}) {
            if (!mask && max_bias > 0.0f) continue;
            for (float scale : {1.0f, 0.1f}) {
                for (int64_t ne0 : {16, 1024}) {
                    for (int64_t ne1 : {16, 1024}) {
                        test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {ne0,   ne1,   1, 1}, mask, scale, max_bias));
                        test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {ne0-1, ne1-1, 1, 1}, mask, scale, max_bias));
                    }
                }
            }
        }
    }
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {16, 2, 32, 1}, true, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {16, 2, 32, 1}, false, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {32, 2, 32, 1}, true,  0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {32, 2, 32, 1}, true,  0.1f, 8.0f));

    // SOFT_MAX: wg512 path (ncols > 1024)
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {2048, 16, 1, 1}, true, 0.1f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {2048, 16, 1, 1}, true, 1.0f, 8.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {4096, 4,  1, 1}, true, 0.088f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {1025, 8,  1, 1}, true, 0.1f, 0.0f));
    // SOFT_MAX: Nemotron attention dimensions (head_dim=128, multi-head)
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {128, 128, 32, 1}, true, 0.088f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {128, 1,   32, 1}, true, 0.088f, 0.0f));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {512, 512,  4, 1}, true, 0.088f, 0.0f));
    // SOFT_MAX: F16 mask (used by flash attention non-causal path)
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {128, 128, 1, 1}, true, 0.088f, 0.0f, GGML_TYPE_F16));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {2048, 16, 1, 1}, true, 0.1f,   0.0f, GGML_TYPE_F16));
    test_cases.emplace_back(new test_soft_max(GGML_TYPE_F32, {16,   16, 1, 1}, true, 1.0f,   0.0f, GGML_TYPE_F16));

    {
        bool all = true;

        for (float v : { 0, 1 }) {
            for (float fs : { 1.0f, 1.4245f }) {
                for (float ef : { 0.0f, 0.7465f }) {
                    for (float af : { 1.0f, 1.4245f }) {
                        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
                            for (bool ff : {false, true}) { // freq_factors
                                test_cases.emplace_back(new test_rope(type, {128,  32, 10, 1}, 128, 0, 512, fs, ef, af, ff, v)); // llama 7B

                                if (all) {
                                    test_cases.emplace_back(new test_rope(type, {128,  40, 10, 1}, 128, 0, 512, fs, ef, af, ff, v)); // llama 13B
                                    test_cases.emplace_back(new test_rope(type, {128,  52, 10, 1}, 128, 0, 512, fs, ef, af, ff, v)); // llama 30B
                                    test_cases.emplace_back(new test_rope(type, {128,  64, 10, 1}, 128, 0, 512, fs, ef, af, ff, v)); // llama 65B
                                }

                                if (all) {
                                    test_cases.emplace_back(new test_rope(type, { 64,   1, 10, 1},  64, 2, 512, fs, ef, af, ff, v)); // neox (falcon 7B)
                                    test_cases.emplace_back(new test_rope(type, { 64,  71, 10, 1},  64, 2, 512, fs, ef, af, ff, v)); // neox (falcon 7B)
                                    test_cases.emplace_back(new test_rope(type, { 64,   8, 10, 1},  64, 2, 512, fs, ef, af, ff, v)); // neox (falcon 40B)
                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 10, 1},  20, 2, 512, fs, ef, af, ff, v)); // neox (stablelm)
                                    test_cases.emplace_back(new test_rope(type, { 80,  32, 10, 1},  32, 2, 512, fs, ef, af, ff, v)); // neox (phi-2)
                                }

                                test_cases.emplace_back(new test_rope(type, { 64, 128, 10, 1},  64, 2, 512, fs, ef, af, ff, v)); // neox (falcon 40B)
                            }
                        }

                        all = false;
                    }
                }
            }
        }
    }

    for (int v : { 0, 1, 2, 3 }) {
        for (int dim : { 0, 1, 2, 3, }) {
            test_cases.emplace_back(new test_concat(GGML_TYPE_F32, {11, 12, 13, 14}, 7, dim, v));
            test_cases.emplace_back(new test_concat(GGML_TYPE_I32, {11, 12, 13, 14}, 7, dim, v));
        }
    }

    // MTP chained-rollout concat pattern: stack multiple [vocab, N] tensors
    // along dim=1. Covers chain lengths 2..5 across all dims for boundary.
    for (int chain_len : { 2, 3, 4, 5 }) {
        for (int dim : { 0, 1, 2 }) {
            test_cases.emplace_back(new test_concat_chain(GGML_TYPE_F32, {32768, 8, 1, 1}, dim, chain_len));
        }
        // smaller-scale sanity at dim=1 only
        test_cases.emplace_back(new test_concat_chain(GGML_TYPE_F32, {1024, 5, 1, 1}, 1, chain_len));
    }

    // MTP argmax→get_rows pattern with set_input/set_output markers on the
    // intermediate (the polaris pattern Fix A ported).
    test_cases.emplace_back(new test_argmax_getrows(GGML_TYPE_F32, GGML_TYPE_F32, 32768, 1, 1024));
    test_cases.emplace_back(new test_argmax_getrows(GGML_TYPE_F32, GGML_TYPE_F32, 32768, 5, 1024));
    test_cases.emplace_back(new test_argmax_getrows(GGML_TYPE_F32, GGML_TYPE_F32, 32768, 32, 1024));

    // MTP LM-head scale matmul: hidden [1024, N] @ lm_head [1024, 32768] → [32768, N]
    // with N in {1, 5, 32}. Covers the chained-rollout's per-iteration LM head.
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 32768, 1, 1024, {1,1}, {1,1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 32768, 5, 1024, {1,1}, {1,1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_F32, GGML_TYPE_F32, 32768, 32, 1024, {1,1}, {1,1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 32768, 1, 1024, {1,1}, {1,1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 32768, 5, 1024, {1,1}, {1,1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 32768, 32, 1024, {1,1}, {1,1}));

    for (ggml_sort_order order : {GGML_SORT_ORDER_ASC, GGML_SORT_ORDER_DESC}) {
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {8, 1, 1, 1}, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {16, 10, 10, 10}, order));
        test_cases.emplace_back(new test_argsort(GGML_TYPE_F32, {60, 10, 10, 10}, order)); // qwen
    }

    test_cases.emplace_back(new test_sum_rows());
    test_cases.emplace_back(new test_upscale());
    test_cases.emplace_back(new test_upscale(GGML_TYPE_F32, { 512, 512, 3, 1 }, 2, true));
    test_cases.emplace_back(new test_upscale_ext());
    test_cases.emplace_back(new test_group_norm());
    test_cases.emplace_back(new test_acc());
    test_cases.emplace_back(new test_pad());
    test_cases.emplace_back(new test_arange());
    test_cases.emplace_back(new test_timestep_embedding());
    test_cases.emplace_back(new test_leaky_relu());

    // flash_attn_ext — matches this fork's constructor (hs, nh, kv, nb, mask, max_bias, softcap, type_KV)
    for (int hs : { 64, 128, 256 }) {
        for (bool mask : { true, false }) {
            for (float max_bias : { 0.0f, 8.0f }) {
                if (!mask && max_bias > 0.0f) continue;
                for (float softcap : { 0.0f, 10.0f }) {
                    if (hs != 128 && softcap != 0.0f) continue;
                    for (int nh : { 4 }) {
                        for (int kv : { 113, 512 }) {
                            for (int nb : { 1, 32 }) {
                                for (ggml_type type_KV : { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K }) {
                                    test_cases.emplace_back(new test_flash_attn_ext(hs, nh, kv, nb, mask, max_bias, softcap, type_KV));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // these tests are disabled to save execution time, but they can be handy for debugging
#if 0
    test_cases.emplace_back(new test_llama(1));
    test_cases.emplace_back(new test_llama(2));
    test_cases.emplace_back(new test_falcon(1));
    test_cases.emplace_back(new test_falcon(2));
#endif

    // run tests
    if (mode == MODE_TEST) {
        ggml_backend_t backend_cpu = ggml_backend_cpu_init();

        size_t n_ok = 0;
        for (auto & test : test_cases) {
            if (test->eval(backend, backend_cpu, op_name)) {
                n_ok++;
            }
        }
        printf("  %zu/%zu tests passed\n", n_ok, test_cases.size());

        // FUSED_UP_GATE: CPU backend ABORTs on this op, so we use a custom
        // decomposed-vs-fused comparison (CPU decomposed vs Vulkan fused).
        if (op_name == nullptr || std::string(op_name) == "FUSED_UP_GATE") {
            printf("\n  === FUSED_UP_GATE (decomposed CPU ref vs fused Vulkan) ===\n");
            size_t fug_ok = 0, fug_total = 0;

            // All 11 supported quant types
            for (ggml_type type_a : {
                    GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
                    GGML_TYPE_Q8_0,
                    GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
                    GGML_TYPE_IQ4_NL}) {
                int64_t blk = ggml_blck_size(type_a);
                int64_t k_small = std::max((int64_t)32, blk);
                int64_t k_large = std::max((int64_t)128, blk * 2);
                for (ggml_unary_op uop : {GGML_UNARY_OP_SILU, GGML_UNARY_OP_GELU, GGML_UNARY_OP_RELU}) {
                    for (int64_t nn : {1, 4}) {
                        fug_total++;
                        if (test_fused_up_gate(type_a, 64, nn, k_small, uop).eval(backend, backend_cpu)) fug_ok++;
                        fug_total++;
                        if (test_fused_up_gate(type_a, 256, nn, k_large, uop).eval(backend, backend_cpu)) fug_ok++;
                    }
                }
            }

            // Edge cases: M=1 (single output row) — use K>=64 so the single
            // output element has enough signal for a stable NMSE comparison.
            fug_total++;
            if (test_fused_up_gate(GGML_TYPE_Q8_0, 1, 1, 64, GGML_UNARY_OP_SILU).eval(backend, backend_cpu)) fug_ok++;
            fug_total++;
            if (test_fused_up_gate(GGML_TYPE_Q4_K, 1, 4, 256, GGML_UNARY_OP_SILU).eval(backend, backend_cpu)) fug_ok++;

            // Edge cases: large N (batch)
            fug_total++;
            if (test_fused_up_gate(GGML_TYPE_Q8_0, 64, 16, 64, GGML_UNARY_OP_SILU).eval(backend, backend_cpu)) fug_ok++;
            fug_total++;
            if (test_fused_up_gate(GGML_TYPE_Q8_0, 64, 32, 64, GGML_UNARY_OP_SILU).eval(backend, backend_cpu)) fug_ok++;

            // Edge cases: M not multiple of tile size (non-aligned)
            fug_total++;
            if (test_fused_up_gate(GGML_TYPE_Q8_0, 17, 1, 32, GGML_UNARY_OP_SILU).eval(backend, backend_cpu)) fug_ok++;
            fug_total++;
            if (test_fused_up_gate(GGML_TYPE_Q8_0, 33, 3, 64, GGML_UNARY_OP_GELU).eval(backend, backend_cpu)) fug_ok++;
            fug_total++;
            if (test_fused_up_gate(GGML_TYPE_Q4_0, 127, 1, 32, GGML_UNARY_OP_RELU).eval(backend, backend_cpu)) fug_ok++;

            // Larger realistic dims
            fug_total++;
            if (test_fused_up_gate(GGML_TYPE_Q8_0, 512, 1, 256, GGML_UNARY_OP_SILU).eval(backend, backend_cpu)) fug_ok++;
            fug_total++;
            if (test_fused_up_gate(GGML_TYPE_Q8_0, 512, 8, 256, GGML_UNARY_OP_SILU).eval(backend, backend_cpu)) fug_ok++;
            fug_total++;
            if (test_fused_up_gate(GGML_TYPE_Q8_0, 1024, 1, 256, GGML_UNARY_OP_SILU).eval(backend, backend_cpu)) fug_ok++;

            // Nemotron-3-Nano dims (intermediate=9216, hidden=3072)
            fug_total++;
            if (test_fused_up_gate(GGML_TYPE_Q8_0, 9216, 1, 3072, GGML_UNARY_OP_SILU).eval(backend, backend_cpu)) fug_ok++;

            printf("  FUSED_UP_GATE: %zu/%zu passed\n", fug_ok, fug_total);
            if (fug_ok != fug_total) {
                n_ok = 0; // force overall failure
            }
        }

        // BATCH_INVARIANCE probes — hunt for ops whose pos-0 output
        // differs between n_tokens=1 and n_tokens=N on the same backend.
        if (op_name == nullptr || std::string(op_name) == "BATCH_INVARIANCE") {
            printf("\n  === BATCH_INVARIANCE (same backend, n=1 vs n=N, pos-0 byte-identical) ===\n");
            size_t bi_ok = 0, bi_total = 0;

            // MUL_MAT: narrow search to find which shape/pipeline triggers the bug.
            for (ggml_type ta : {GGML_TYPE_F16, GGML_TYPE_Q4_K, GGML_TYPE_Q8_0}) {
                for (int64_t N : {2, 4}) {
                    bi_total++; if (test_batch_invariance_mul_mat(ta,  256,  256, N).eval(backend)) bi_ok++;
                    bi_total++; if (test_batch_invariance_mul_mat(ta,  512,  512, N).eval(backend)) bi_ok++;
                    bi_total++; if (test_batch_invariance_mul_mat(ta, 1024,  512, N).eval(backend)) bi_ok++;
                    bi_total++; if (test_batch_invariance_mul_mat(ta, 2048,  128, N).eval(backend)) bi_ok++;
                    bi_total++; if (test_batch_invariance_mul_mat(ta, 2048,  256, N).eval(backend)) bi_ok++;
                    bi_total++; if (test_batch_invariance_mul_mat(ta, 2048,  512, N).eval(backend)) bi_ok++;
                    bi_total++; if (test_batch_invariance_mul_mat(ta, 2048, 1024, N).eval(backend)) bi_ok++;
                }
            }

            // SOFT_MAX: small (fits wg=64), larger (triggers wg512 variant).
            for (int64_t K : {64, 512, 4096}) {
                for (int64_t N : {2, 4}) {
                    bi_total++; if (test_batch_invariance_soft_max(K, N).eval(backend)) bi_ok++;
                }
            }

            // RMS_NORM: Qwen-like head_dim.
            for (int64_t K : {64, 128, 256}) {
                for (int64_t N : {2, 4}) {
                    bi_total++; if (test_batch_invariance_rms_norm(K, N).eval(backend)) bi_ok++;
                }
            }

            // Unary activations.
            for (ggml_unary_op uop : {GGML_UNARY_OP_SILU, GGML_UNARY_OP_GELU, GGML_UNARY_OP_RELU}) {
                bi_total++; if (test_batch_invariance_unary(uop, 1024, 4).eval(backend)) bi_ok++;
            }

            // FUSED_UP_GATE (dense) — same spec-const pipeline-variant pattern as mul_mat_vec.
            for (ggml_type ta : {GGML_TYPE_Q4_K, GGML_TYPE_Q8_0}) {
                for (int64_t N : {2, 4}) {
                    bi_total++; if (test_batch_invariance_fused_up_gate(ta, 256, 256, N).eval(backend)) bi_ok++;
                    bi_total++; if (test_batch_invariance_fused_up_gate(ta, 2048, 512, N).eval(backend)) bi_ok++;
                }
            }

            // MOE_FUSED_UP_GATE — MoE expert-routed variant.
            for (ggml_type ta : {GGML_TYPE_Q4_K, GGML_TYPE_Q8_0}) {
                for (int64_t N : {2, 4, 8}) {
                    bi_total++; if (test_batch_invariance_moe_fused_up_gate(ta, 256, 256, 16, 2, N).eval(backend)) bi_ok++;
                    bi_total++; if (test_batch_invariance_moe_fused_up_gate(ta, 2048, 512, 16, 2, N).eval(backend)) bi_ok++;
                    // Qwen3.5-35B-A3B-ish shape: K=5120, M=2560, n_experts=128, n_used=8.
                    bi_total++; if (test_batch_invariance_moe_fused_up_gate(ta, 5120, 2560, 128, 8, N).eval(backend)) bi_ok++;
                }
            }

            // Per-position BI: fused_moe's pos-1 output in a batched dispatch
            // must match a solo-at-pos-1 call. The pos-0-only test above
            // misses a shader bug where pos-1+ outputs diverge massively from
            // their sequential counterparts (full-model evidence max|Δ|=5.1
            // between batched pos=1 and sequential-through-1 on 35B-A3B).
            for (ggml_type ta : {GGML_TYPE_Q4_K, GGML_TYPE_Q8_0}) {
                bi_total++; if (test_pos1_invariance_moe_fused_up_gate(ta, 256, 256, 16, 2).eval(backend)) bi_ok++;
                bi_total++; if (test_pos1_invariance_moe_fused_up_gate(ta, 2048, 512, 16, 2).eval(backend)) bi_ok++;
                bi_total++; if (test_pos1_invariance_moe_fused_up_gate(ta, 5120, 2560, 128, 8).eval(backend)) bi_ok++;
            }

            // MUL_MAT_ID — non-fused MoE mat-mat (the `-no-fmoe` path).
            for (ggml_type ta : {GGML_TYPE_Q4_K, GGML_TYPE_Q8_0}) {
                for (int64_t N : {2, 4}) {
                    bi_total++; if (test_batch_invariance_mul_mat_id(ta,  256, 256, 16, 2, N).eval(backend)) bi_ok++;
                    bi_total++; if (test_batch_invariance_mul_mat_id(ta,  256, 512, 16, 2, N).eval(backend)) bi_ok++;
                    bi_total++; if (test_batch_invariance_mul_mat_id(ta, 2048, 256, 16, 2, N).eval(backend)) bi_ok++;
                    bi_total++; if (test_batch_invariance_mul_mat_id(ta, 2048, 512, 16, 2, N).eval(backend)) bi_ok++;
                }
            }

            // FLASH_ATTN — exercise Vulkan pipeline variants on RDNA2:
            //   {cm2/cm1/scalar} × {f16acc/f32acc} × {small_rows/large_rows} × {aligned/unaligned}.
            // Qwen3.5 shapes: head_dim=128, nh_q=16, nh_kv∈{8,2} (GQA), kv_ctx=512.
            // N∈{1,2,4,8} drives small_rows vs large_rows pipeline selection.
            for (bool f32acc : {false, true}) {
                for (int64_t nh_kv : {8, 2}) {
                    for (int64_t N : {2, 4, 8}) {
                        bi_total++; if (test_batch_invariance_flash_attn(
                            /*hs=*/128, /*nh_q=*/16, /*nh_kv=*/nh_kv,
                            /*kv=*/512, /*N=*/N, /*f32acc=*/f32acc,
                            /*use_mask=*/true).eval(backend)) bi_ok++;
                    }
                }
            }

            printf("  BATCH_INVARIANCE: %zu/%zu passed\n", bi_ok, bi_total);
            if (bi_ok != bi_total) n_ok = 0;
        }

        // MOE_FUSED_UP_GATE — MoE variant of FUSED_UP_GATE (compare-vs-CPU)
        if (op_name == nullptr || std::string(op_name) == "MOE_FUSED_UP_GATE") {
            printf("\n  === MOE_FUSED_UP_GATE (Vulkan moe_fused_up_gate vs CPU ggml_moe_up_gate) ===\n");
            size_t mfg_ok = 0, mfg_total = 0;

            // Coverage matrix: a few quants × {SILU, GELU} × {batch=1, batch=8} ×
            // {n_expert_used=1, 2, 4} × {small, medium, target-shape}.
            // The Qwen3.5-35B-A3B shape (k=2048, m=512, n_experts=256, n_used=8)
            // is too large for a single test_backend_ops case (would allocate
            // hundreds of MB of weights), so we cover representative shapes
            // up to the row_ids 4096-cap.
            // Binary-search shapes to localize bugs:
            // (k, m, n_exp, n_used, n_tok)
            //  - 1×1: simplest, no row_ids iteration
            //  - 1×2: 2 tokens, 1 slot each (cross-token via row_ids)
            //  - 2×1: 1 token, 2 slots (same-token slot indexing)
            //  - 2×2: full multi-token multi-slot
            mfg_total++;
            if (test_moe_fused_up_gate(GGML_TYPE_Q8_0, 128, 128, 4, 1, 1, GGML_UNARY_OP_SILU).eval(backend, backend_cpu)) mfg_ok++;
            mfg_total++;
            if (test_moe_fused_up_gate(GGML_TYPE_Q8_0, 128, 128, 4, 1, 2, GGML_UNARY_OP_SILU).eval(backend, backend_cpu)) mfg_ok++;
            mfg_total++;
            if (test_moe_fused_up_gate(GGML_TYPE_Q8_0, 128, 128, 4, 2, 1, GGML_UNARY_OP_SILU).eval(backend, backend_cpu)) mfg_ok++;
            mfg_total++;
            if (test_moe_fused_up_gate(GGML_TYPE_Q8_0, 128, 128, 4, 2, 2, GGML_UNARY_OP_SILU).eval(backend, backend_cpu)) mfg_ok++;

            // Block-32 quants (k must be a multiple of 32).
            const ggml_type block32_types[] = {
                GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL,
            };
            for (ggml_type ta : block32_types) {
                for (ggml_unary_op uop : {GGML_UNARY_OP_SILU, GGML_UNARY_OP_GELU}) {
                    mfg_total++;
                    if (test_moe_fused_up_gate(ta, 64, 64, 4, 2, 4, uop).eval(backend, backend_cpu)) mfg_ok++;
                    mfg_total++;
                    if (test_moe_fused_up_gate(ta, 256, 256, 8, 2, 8, uop).eval(backend, backend_cpu)) mfg_ok++;
                    mfg_total++;
                    if (test_moe_fused_up_gate(ta, 128, 128, 4, 1, 1, uop).eval(backend, backend_cpu)) mfg_ok++;
                }
            }
            // Superblock-256 quants (k must be a multiple of 256).
            const ggml_type block256_types[] = {
                GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS,
            };
            for (ggml_type ta : block256_types) {
                for (ggml_unary_op uop : {GGML_UNARY_OP_SILU, GGML_UNARY_OP_GELU}) {
                    mfg_total++;
                    if (test_moe_fused_up_gate(ta, 256, 256, 8, 2, 8, uop).eval(backend, backend_cpu)) mfg_ok++;
                    mfg_total++;
                    if (test_moe_fused_up_gate(ta, 256, 128, 4, 1, 1, uop).eval(backend, backend_cpu)) mfg_ok++;
                }
            }

            // Tighter Qwen3.5-style proportions (smaller, but same shape ratios:
            // k > m, n_used = sqrt(n_experts)). Stays under the 4096 row cap.
            mfg_total++;
            if (test_moe_fused_up_gate(GGML_TYPE_IQ2_S, 512, 128, 16, 4, 8, GGML_UNARY_OP_SILU).eval(backend, backend_cpu)) mfg_ok++;
            mfg_total++;
            if (test_moe_fused_up_gate(GGML_TYPE_IQ3_XXS, 512, 128, 16, 4, 8, GGML_UNARY_OP_SILU).eval(backend, backend_cpu)) mfg_ok++;

            printf("  MOE_FUSED_UP_GATE: %zu/%zu passed\n", mfg_ok, mfg_total);
            if (mfg_ok != mfg_total) {
                n_ok = 0; // force overall failure
            }
        }

        // GROUPED_TOPK: per-row group-aware top-k
        if (op_name == nullptr || std::string(op_name) == "GROUPED_TOPK") {
            printf("\n  === GROUPED_TOPK (Vulkan vs CPU iqk_grouped_top_k) ===\n");
            size_t gt_ok = 0, gt_total = 0;
            // Qwen3.5-35B-A3B shape: 256 experts in 8 groups of 32, top-4 groups,
            // top-2 within each group for scoring, top-8 experts overall.
            for (int64_t n_tok : {1, 2, 8, 64}) {
                gt_total++;
                if (test_grouped_topk(256, 8, 4, 2, 8, n_tok).eval(backend, backend_cpu)) gt_ok++;
            }
            // Smaller / different shapes for sanity coverage.
            gt_total++; if (test_grouped_topk(128, 8, 2, 2, 4, 4).eval(backend, backend_cpu)) gt_ok++;
            gt_total++; if (test_grouped_topk(64,  4, 2, 2, 4, 4).eval(backend, backend_cpu)) gt_ok++;
            gt_total++; if (test_grouped_topk(32,  4, 2, 1, 2, 1).eval(backend, backend_cpu)) gt_ok++;
            printf("  GROUPED_TOPK: %zu/%zu passed\n", gt_ok, gt_total);
            if (gt_ok != gt_total) {
                n_ok = 0;
            }
        }

        // MULTI_ADD: uses strided view_2d, needs custom eval
        if (op_name == nullptr || std::string(op_name) == "MULTI_ADD") {
            printf("\n  === MULTI_ADD (strided view_2d, CPU ref vs Vulkan) ===\n");
            size_t ma_ok = 0, ma_total = 0;
            struct { int64_t ne0; int64_t ne1; int n_experts; } ma_cases[] = {
                {128,    4,  6},  // 6 experts, 4 tokens
                {128,    1,  6},  // single token
                {256,    4,  8},  // 8 experts
                {5120,   1,  6},  // GLM hidden dim
                {128,    1,  1},  // single expert (identity)
                {128,    1,  2},  // 2 experts
                {128,   32,  6},  // large batch
                {128,    4, 16},  // many experts
                {  1,    4,  6},  // minimal ne0
                {3072,   1,  6},  // Nemotron hidden dim
                {3072,   8,  6},  // Nemotron batch
                {128,    4, 32},  // many experts, batch
            };
            for (auto & c : ma_cases) {
                ma_total++;
                if (test_multi_add(c.ne0, c.ne1, c.n_experts).eval(backend, backend_cpu)) ma_ok++;
            }
            printf("  MULTI_ADD: %zu/%zu passed\n", ma_ok, ma_total);
            if (ma_ok != ma_total) {
                n_ok = 0; // force overall failure
            }
        }

        ggml_backend_free(backend_cpu);

        return n_ok == test_cases.size();
    }

    if (mode == MODE_PERF) {
        for (auto & test : test_cases) {
            test->eval_perf(backend, op_name);
        }
        return true;
    }

    GGML_ABORT("fatal error");
    return false;
}

static void usage(char ** argv) {
    printf("Usage: %s [mode] [-o op] [-b backend]\n", argv[0]);
    printf("  valid modes are: test (compare with CPU backend for correctness) or perf (performance evaluation)\n");
    printf("  op names are as given by ggml_op_desc()\n");
}

int main(int argc, char ** argv) {
    test_mode mode = MODE_TEST;
    const char * op_name_filter = NULL;
    const char * backend_filter = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "test") == 0) {
            mode = MODE_TEST;
        } else if (strcmp(argv[i], "perf") == 0) {
            mode = MODE_PERF;
        } else if (strcmp(argv[i], "-o") == 0) {
            if (i + 1 < argc) {
                op_name_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (i + 1 < argc) {
                backend_filter = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else {
            usage(argv);
            return 1;
        }
    }

    // enumerate backends
    printf("Testing %zu backends\n\n", ggml_backend_reg_get_count());

    size_t n_ok = 0;

    for (size_t i = 0; i < ggml_backend_reg_get_count(); i++) {
        printf("Backend %zu/%zu (%s)\n", i + 1, ggml_backend_reg_get_count(), ggml_backend_reg_get_name(i));

        if (backend_filter != NULL && strcmp(backend_filter, ggml_backend_reg_get_name(i)) != 0) {
            printf("  Skipping\n");
            n_ok++;
            continue;
        }

        ggml_backend_t backend = ggml_backend_reg_init_backend(i, NULL);
        GGML_ASSERT(backend != NULL);

        if (backend_filter == NULL && ggml_backend_is_cpu(backend)) {
            printf("  Skipping CPU backend\n");
            ggml_backend_free(backend);
            n_ok++;
            continue;
        }

        printf("  Backend name: %s\n", ggml_backend_name(backend));

        bool ok = test_backend(backend, mode, op_name_filter);

        printf("  Backend %s: ", ggml_backend_name(backend));
        if (ok) {
            printf("\033[1;32mOK\033[0m\n");
            n_ok++;
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }

        printf("\n");

        ggml_backend_free(backend);
    }

    printf("%zu/%zu backends passed\n", n_ok, ggml_backend_reg_get_count());

    if (n_ok != ggml_backend_reg_get_count()) {
        printf("\033[1;31mFAIL\033[0m\n");
        return 1;
    }

    ggml_quantize_free();

    printf("\033[1;32mOK\033[0m\n");
    return 0;
}
