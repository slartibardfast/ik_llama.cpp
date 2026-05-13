//
// Copyright (C) 2023-2025 The llama.cpp authors
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <iterator>
#include <map>
#include <numeric>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ggml.h"
#include "llama.h"
#include "common.h"
#include "perplexity.h"
#include "speculative.h"
#include "ggml-cuda.h"
#include "ggml-sycl.h"

#ifdef GGML_USE_CANN
#include "ggml-cann.h"
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

// utils
static uint64_t get_time_ns() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& str, const std::pair<T1, T2>& item) {
    str << '{' << item.first << ", " << item.second << '}';
    return str;
}

template<class T>
static std::string join(const std::vector<T> & values, const std::string & delim) {
    std::ostringstream str;
    for (size_t i = 0; i < values.size(); i++) {
        str << values[i];
        if (i < values.size() - 1) {
            str << delim;
        }
    }
    return str.str();
}

template<typename T, typename F>
static std::vector<std::string> transform_to_str(const std::vector<T> & values, F f) {
    std::vector<std::string> str_values;
    std::transform(values.begin(), values.end(), std::back_inserter(str_values), f);
    return str_values;
}

template<typename T>
static T avg(const std::vector<T> & v) {
    if (v.empty()) {
        return 0;
    }
    T sum = std::accumulate(v.begin(), v.end(), T(0));
    return sum / (T)v.size();
}

template<typename T>
static T stdev(const std::vector<T> & v) {
    if (v.size() <= 1) {
        return 0;
    }
    T mean = avg(v);
    T sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), T(0));
    T stdev = std::sqrt(sq_sum / (T)(v.size() - 1) - mean * mean * (T)v.size() / (T)(v.size() - 1));
    return stdev;
}

static std::string get_cpu_info() {
    std::string id;
#ifdef __linux__
    FILE * f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char buf[1024];
        while (fgets(buf, sizeof(buf), f)) {
            if (strncmp(buf, "model name", 10) == 0) {
                char * p = strchr(buf, ':');
                if (p) {
                    p++;
                    while (std::isspace(*p)) {
                        p++;
                    }
                    while (std::isspace(p[strlen(p) - 1])) {
                        p[strlen(p) - 1] = '\0';
                    }
                    id = p;
                    break;
                }
            }
        }
        fclose(f);
    }
#elif defined(_WIN32)
    HKEY hKey;
    if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                     TEXT("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0"),
                     0,
                     KEY_READ,
                     &hKey) != ERROR_SUCCESS) {
        // fail to open registry key
        return "";
    }
    char cpu_brand[256];
    DWORD cpu_brand_size = sizeof(cpu_brand);
    if (RegQueryValueExA(hKey,
                        TEXT("ProcessorNameString"),
                        NULL,
                        NULL,
                        (LPBYTE)cpu_brand,
                        &cpu_brand_size) == ERROR_SUCCESS) {
        id.assign(cpu_brand, cpu_brand_size);
    }
    RegCloseKey(hKey);
#endif
    // TODO: other platforms
    return id;
}

static std::string get_gpu_info() {
    std::string id;
#ifdef GGML_USE_CUDA
    int count = ggml_backend_cuda_get_device_count();
    for (int i = 0; i < count; i++) {
        char buf[128];
        ggml_backend_cuda_get_device_description(i, buf, sizeof(buf));
        id += buf;
        if (i < count - 1) {
            id += "/";
        }
    }
#endif
#ifdef GGML_USE_SYCL
    int count = ggml_backend_sycl_get_device_count();
    for (int i = 0; i < count; i++) {
        char buf[128];
        ggml_sycl_get_device_description(i, buf, sizeof(buf));
        id += buf;
        if (i < count - 1) {
            id += "/";
        }
    }
#endif
#ifdef GGML_USE_CANN
    uint32_t count = ggml_backend_cann_get_device_count();
    for (uint32_t i = 0; i < count; i++) {
        char buf[128];
        ggml_backend_cann_get_device_description(i, buf, sizeof(buf));
        id += buf;
        if (i < count - 1) {
            id += "/";
        }
    }
#endif
    // TODO: other backends
    return id;
}

// command line params
enum output_formats {NONE, CSV, JSON, MARKDOWN, SQL};

static const char * output_format_str(output_formats format) {
    switch (format) {
        case NONE:     return "none";
        case CSV:      return "csv";
        case JSON:     return "json";
        case MARKDOWN: return "md";
        case SQL:      return "sql";
        default: GGML_ABORT("invalid output format");
    }
}

static bool output_format_from_str(const std::string & s, output_formats & format) {
    if (s == "none") {
        format = NONE;
    } else if (s == "csv") {
        format = CSV;
    } else if (s == "json") {
        format = JSON;
    } else if (s == "md") {
        format = MARKDOWN;
    } else if (s == "sql") {
        format = SQL;
    } else {
        return false;
    }
    return true;
}

static const char * split_mode_str(llama_split_mode mode) {
    switch (mode) {
        case LLAMA_SPLIT_MODE_NONE:  return "none";
        case LLAMA_SPLIT_MODE_LAYER: return "layer";
        case LLAMA_SPLIT_MODE_GRAPH: return "graph";
        default: GGML_ABORT("invalid split mode");
    }
}

static std::string pair_str(const std::pair<int, int> & p) {
    static char buf[32];
    snprintf(buf, sizeof(buf), "%d,%d", p.first, p.second);
    return buf;
}

// Ser = Smart Expert Reduction
using Ser = std::pair<int,float>;

// Spec-aware bench fields (T8 Phase 1). See PHASE_DFLASH.md Phase 1.1/1.2.
//
//   spec_type       sweep over { none, mtp, dflash, draft, ngram-* } drivers
//   n_draft         draft chain length / DFlash BLOCK_SIZE
//   spec_model      drafter GGUF (DFlash/Draft only; MTP uses inline heads)
//   prompt_files    real prompts (one row per file); needed for spec accept-rate
//   ppl_of_output   re-decode generated output under target → corpus PPL
//                   captures cumulative numerical drift across spec methods
struct cmd_params {
    std::vector<std::string> model;
    std::vector<int> n_prompt;
    std::vector<int> n_gen;
    std::vector<std::pair<int, int>> n_pg;
    std::vector<std::pair<int, int>> n_gp;
    std::vector<int> n_batch;
    std::vector<int> n_ubatch;
    std::vector<common_speculative_type> spec_type;
    std::vector<int> n_draft;
    std::vector<std::string> spec_model;
    std::vector<std::string> prompt_files;
    bool ppl_of_output = false;
    std::vector<ggml_type> type_k;
    std::vector<ggml_type> type_v;
    std::vector<std::pair<int,int>> n_threads;
    std::vector<int> n_gpu_layers;
    std::vector<std::string> rpc_servers;
    std::vector<llama_split_mode> split_mode;
    std::vector<int> main_gpu;
    std::vector<bool> no_kv_offload;
    std::vector<bool> flash_attn;
    std::vector<int> mla_attn;
    std::vector<int> attn_max_batch;
    std::vector<Ser> ser;
    std::vector<bool> reuse;
    std::vector<std::vector<float>> tensor_split;
    std::vector<bool> use_mmap;
    std::vector<bool> embeddings;
    std::vector<llama_model_tensor_buft_override> buft_overrides;
    ggml_numa_strategy numa;
    std::string cuda_params;
    int reps;
    bool verbose;
    bool warmup;
    bool repack = false;
    bool fmoe = true;
    bool ger = false;     // ger = Grouped Expert Routing
    bool no_fug = false;
    bool use_thp = false;
    bool no_ooae = false;
    bool mqkv = false;
    bool muge = false;
    bool defer_experts = false;
    bool rcache = false;
    bool sas = false;
    int  max_gpu = 0;
    bool print_overrides = false;
    bool fit = false;
    int  fit_margin = 0;
    output_formats output_format;
    output_formats output_format_stderr;
};

static const cmd_params cmd_params_defaults = {
    /* model                */ {"models/7B/ggml-model-q4_0.gguf"},
    /* n_prompt             */ {512},
    /* n_gen                */ {128},
    /* n_pg                 */ {},
    /* n_gp                 */ {},
    /* n_batch              */ {2048},
    /* n_ubatch             */ {512},
    /* spec_type            */ {COMMON_SPECULATIVE_TYPE_NONE},
    /* n_draft              */ {0},     // 0 = method-specific default applied later
    /* spec_model           */ {""},
    /* prompt_files         */ {""},    // empty path = synthetic n_prompt prefix
    /* ppl_of_output        */ false,
    /* type_k               */ {GGML_TYPE_F16},
    /* type_v               */ {GGML_TYPE_F16},
    /* n_threads            */ {{cpu_get_num_math(), cpu_get_num_math()}},
    /* n_gpu_layers         */ {999},
    /* rpc_servers          */ {""},
    /* split_mode           */ {LLAMA_SPLIT_MODE_LAYER},
    /* main_gpu             */ {0},
    /* no_kv_offload        */ {false},
    /* flash_attn           */ {true},
    /* mla_attn             */ {3},
    /* attn_max_batch       */ {0},
    /* ser                  */ {{-1,0.0f}},
    /* reuse                */ {true},
    /* tensor_split         */ {std::vector<float>(llama_max_devices(), 0.0f)},
    /* use_mmap             */ {true},
    /* embeddings           */ {false},
    /* buft_overrides       */ {},
    /* numa                 */ GGML_NUMA_STRATEGY_DISABLED,
    /* cuda_params          */ {},
    /* reps                 */ 5,
    /* verbose              */ false,
    /* warmup               */ true,
    /* repack               */ false,
    /* fmoe                 */ true,
    /* ger                  */ false,
    /* no_fug               */ false,
    /* use_thp              */ false,
    /* no_ooae              */ false,
    /* mqkv                 */ false,
    /* muge                 */ false,
    /* defer_experts        */ false,
    /* rcache               */ false,
    /* sas                  */ false,
    /* max_gpu              */ 0,
    /* print_overrides      */ false,
    /* fit                  */ false,
    /* fit_margin           */ 0,
    /* output_format        */ MARKDOWN,
    /* output_format_stderr */ NONE,
};

static void print_usage(int /* argc */, char ** argv) {
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("options:\n");
    printf("  -h, --help\n");
    printf("  -m, --model <filename>              (default: %s)\n", join(cmd_params_defaults.model, ",").c_str());
    printf("  -p, --n-prompt <n>                  (default: %s)\n", join(cmd_params_defaults.n_prompt, ",").c_str());
    printf("  -n, --n-gen <n>                     (default: %s)\n", join(cmd_params_defaults.n_gen, ",").c_str());
    printf("  -pg <pp,tg>                         (default: %s)\n", join(transform_to_str(cmd_params_defaults.n_pg, pair_str), ",").c_str());
    printf("  -gp <pp,tg>                         (default: %s)\n", join(transform_to_str(cmd_params_defaults.n_gp, pair_str), ",").c_str());
    printf("  -b, --batch-size <n>                (default: %s)\n", join(cmd_params_defaults.n_batch, ",").c_str());
    printf("  -ub, --ubatch-size <n>              (default: %s)\n", join(cmd_params_defaults.n_ubatch, ",").c_str());
    printf("  -ctk, --cache-type-k <t>            (default: %s)\n", join(transform_to_str(cmd_params_defaults.type_k, ggml_type_name), ",").c_str());
    printf("  -ctv, --cache-type-v <t>            (default: %s)\n", join(transform_to_str(cmd_params_defaults.type_v, ggml_type_name), ",").c_str());
    printf("  -t, --threads <n>                   (default: %s)\n", join(cmd_params_defaults.n_threads, ",").c_str());
    printf("  -tgb, --threads-gen-batch <n1,n2>   (default: %s)\n", join(cmd_params_defaults.n_threads, ",").c_str());
    printf("  -ngl, --n-gpu-layers <n>            (default: %s)\n", join(cmd_params_defaults.n_gpu_layers, ",").c_str());
    printf("  --n-cpu-moe <n>                     (default: none)\n");
    printf("  -rpc, --rpc <rpc_servers>           (default: %s)\n", join(cmd_params_defaults.rpc_servers, ",").c_str());
    printf("  -sm, --split-mode <none|layer|graph>(default: %s)\n", join(transform_to_str(cmd_params_defaults.split_mode, split_mode_str), ",").c_str());
    printf("  -mg, --main-gpu <i>                 (default: %s)\n", join(cmd_params_defaults.main_gpu, ",").c_str());
    printf("  -nkvo, --no-kv-offload <0|1>        (default: %s)\n", join(cmd_params_defaults.no_kv_offload, ",").c_str());
    printf("  -fa, --flash-attn <0|1>             (default: %s)\n", join(cmd_params_defaults.flash_attn, ",").c_str());
    printf("  -mla, --mla-attn <0|1|2>            (default: %s)\n", join(cmd_params_defaults.mla_attn, ",").c_str());
    printf("  -amb, --attn-max-batch <i>          (default: %s)\n", join(cmd_params_defaults.attn_max_batch, ",").c_str());
    printf("  -ser, --smart-expert-reduction <i,f>(default: %s)\n", join(cmd_params_defaults.attn_max_batch, ",").c_str());
    printf("  -gr, --graph-reuse <0|1>            (default: %s)\n", join(cmd_params_defaults.reuse, ",").c_str());
    printf("  -mmp, --mmap <0|1>                  (default: %s)\n", join(cmd_params_defaults.use_mmap, ",").c_str());
    printf("  --numa <distribute|isolate|numactl> (default: disabled)\n");
    printf("  -embd, --embeddings <0|1>           (default: %s)\n", join(cmd_params_defaults.embeddings, ",").c_str());
    printf("  -ts, --tensor-split <ts0/ts1/..>    (default: 0)\n");
    printf("  -r, --repetitions <n>               (default: %d)\n", cmd_params_defaults.reps);
    printf("  -o, --output <csv|json|md|sql>      (default: %s)\n", output_format_str(cmd_params_defaults.output_format));
    printf("  -oe, --output-err <csv|json|md|sql> (default: %s)\n", output_format_str(cmd_params_defaults.output_format_stderr));
    printf("  -v, --verbose                       (default: %s)\n", cmd_params_defaults.verbose ? "1" : "0");
    printf("  -w, --warmup <0|1>                  (default: %s)\n", cmd_params_defaults.warmup ? "1" : "0");
    printf("  -rtr, --run-time-repack <0|1>       (default: %s)\n", cmd_params_defaults.repack ? "1" : "0");
    printf("  -cuda, --cuda-params <string>       (default: %s)\n", cmd_params_defaults.cuda_params.c_str());
    printf("  -mqkv, --merge-qkv                  (default: %s)\n", cmd_params_defaults.mqkv ? "1" : "0");
    printf("  -muge, --merge-up-gate-experts      (default: %s)\n", cmd_params_defaults.muge ? "1" : "0");
    printf("  --defer-experts                     (Linux only, default: %s)\n", cmd_params_defaults.defer_experts ? "1" : "0");
    printf("  -rcache, --rope-cache               (default: %s)\n", cmd_params_defaults.rcache ? "1" : "0");
    printf("  -thp, --transparent-huge-pages <0|1> (default: %s)\n", cmd_params_defaults.use_thp? "1" : "0");
    printf("  -ot, --override-tensor pattern      (default: none)\n");
    printf("  -fmoe, --fused-moe <0|1>            (default: %s)\n", cmd_params_defaults.fmoe? "1" : "0");
    printf("  -ger, --grouped-expert-routing <0|1>(default: %s)\n", cmd_params_defaults.ger ? "1" : "0");
    printf("  -no-fug, --no-fused-up-gate <0|1>   (default: %s)\n", cmd_params_defaults.no_fug? "1" : "0");
    printf("  -no-ooae, --no-offload-only-active-experts <0|1>   (default: %s)\n", cmd_params_defaults.no_ooae? "1" : "0");
    printf("  -sas, --scheduler-async <0|1>       (default: %s)\n", cmd_params_defaults.sas ? "1" : "0");
    printf("  --fit <0|1>                         (default: %s)\n", cmd_params_defaults.fit ? "1" : "0");
    printf("  --fit-margin N                      (default: %d)\n", cmd_params_defaults.fit_margin);
    printf("  --max-gpu <N>                       (default: %d)\n", cmd_params_defaults.max_gpu);
    printf("        --print-overrides <0|1>       (default: %s)\n", cmd_params_defaults.print_overrides ? "1" : "0");
    printf("\n");
    printf("speculative decoding (T8):\n");
    printf("  --spec <name[,name,...]>            spec method(s): none, mtp, dflash, draft, ngram-simple,\n");
    printf("                                      ngram-map-k, ngram-map-k4v, ngram-mod, ngram-cache, suffix, eagle3\n");
    printf("                                      (default: none)\n");
    printf("  -nd, --draft <N[,N,...]>            draft chain length or DFlash BLOCK_SIZE (0 = method default)\n");
    printf("  --spec-model <path>                 drafter GGUF (DFlash/draft; MTP uses inline heads)\n");
    printf("  --prompt-file <path>                real prompt for spec accept-rate; may be repeated\n");
    printf("  --ppl-of-output                     after each tg row, re-decode generated output and report\n");
    printf("                                      target PPL — quality bound for spec method comparison\n");
    printf("\n");
    printf("Multiple values can be given for each parameter by separating them with ',' or by specifying the parameter multiple times.\n");
}

static ggml_type ggml_type_from_name(const std::string & s) {
    if (s == "f16") {
        return GGML_TYPE_F16;
    }
    if (s == "bf16") {
        return GGML_TYPE_BF16;
    }
    if (s == "q8_0") {
        return GGML_TYPE_Q8_0;
    }
    if (s == "q4_0") {
        return GGML_TYPE_Q4_0;
    }
    if (s == "q4_1") {
        return GGML_TYPE_Q4_1;
    }
    if (s == "q5_0") {
        return GGML_TYPE_Q5_0;
    }
    if (s == "q5_1") {
        return GGML_TYPE_Q5_1;
    }
    if (s == "iq4_nl") {
        return GGML_TYPE_IQ4_NL;
    }
    if (s == "q6_0") {
        return GGML_TYPE_Q6_0;
    }
    if (s == "q8_KV") {
        return GGML_TYPE_Q8_KV;
    }

    return GGML_TYPE_COUNT;
}

namespace {
bool parse_buft_overrides(const std::string& value, std::vector<llama_model_tensor_buft_override>& overrides) {
    /* static */ std::map<std::string, ggml_backend_buffer_type_t> buft_list;
    if (buft_list.empty()) {
        // enumerate all the devices and add their buffer types to the list
        for (size_t i = 0; i < ggml_backend_reg_get_count(); ++i) {
            //auto * dev = ggml_backend_reg_get_name(i);
            auto * buft = ggml_backend_reg_get_default_buffer_type(i);
            if (buft) {
                buft_list[ggml_backend_buft_name(buft)] = buft;
            }
        }
    }
    for (const auto & override : string_split<std::string>(value, ',')) {
        std::string::size_type pos = override.find('=');
        if (pos == std::string::npos) {
            fprintf(stderr, "Invalid buft override argument %s\n", value.c_str());
            return false;
        }
        std::string tensor_name = override.substr(0, pos);
        std::string buffer_type = override.substr(pos + 1);
        if (buft_list.find(buffer_type) == buft_list.end()) {
            fprintf(stderr, "Available buffer types:\n");
            for (const auto & it : buft_list) {
                fprintf(stderr, "  %s\n", ggml_backend_buft_name(it.second));
            }
            return false;
        }
        overrides.push_back({strdup(tensor_name.c_str()), buft_list.at(buffer_type)});
    }
    return true;
}
bool add_cpu_buft_overrides(const char * arg, std::vector<llama_model_tensor_buft_override>& overrides) {
    int n_layers = std::stoi(arg);
    if (n_layers < 0) {
        fprintf(stderr, "error: Invalid value for --n-cpu-moe: %s\n", arg);
        return false;
    }
    for (int32_t l = 0; l < n_layers; ++l) {
        std::string pattern = "blk\\." + std::to_string(l) + "\\.(ffn_(up|down|gate)_exps\\.weight)";
        overrides.push_back({strdup(pattern.c_str()), ggml_backend_cpu_buffer_type()});
    }
    return true;
}

template<class T1, class T2>
std::vector<std::pair<T1,T2>> string_split_pairs(const std::string & str, char delim) {
    std::vector<std::pair<T1,T2>> values;
    std::istringstream str_stream(str);
    std::string token;
    T1 first_value;
    int i = 0;
    while (std::getline(str_stream, token, delim)) {
        std::istringstream token_stream(token);
        if (i%2 == 0) {
            token_stream >> first_value;
            if (token_stream.fail()) return {};
        } else {
            T2 value;
            token_stream >> value;
            if (token_stream.fail()) return {};
            values.emplace_back(first_value, value);
        }
        i++;
    }
    return values;
}
bool operator==(const llama_model_tensor_buft_override & lhs, const llama_model_tensor_buft_override & rhs) {
    return lhs.buft == rhs.buft &&
          ((lhs.pattern == nullptr && rhs.pattern == nullptr) || strcmp(lhs.pattern, rhs.pattern) == 0);
}
bool operator==(const std::vector<llama_model_tensor_buft_override> & lhs, const std::vector<llama_model_tensor_buft_override> & rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (int i = 0; i < int(lhs.size()); ++i) {
        if (!(lhs[i] == rhs[i])) return false;
    }
    return true;
}
}

static cmd_params parse_cmd_params(int argc, char ** argv) {
    cmd_params params;
    std::string arg;
    bool invalid_param = false;
    const std::string arg_prefix = "--";
    const char split_delim = ',';

    params.verbose = cmd_params_defaults.verbose;
    params.output_format = cmd_params_defaults.output_format;
    params.output_format_stderr = cmd_params_defaults.output_format_stderr;
    params.reps = cmd_params_defaults.reps;
    params.numa = cmd_params_defaults.numa;
    params.warmup = cmd_params_defaults.warmup;

    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            exit(0);
        } else if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<std::string>(argv[i], split_delim);
            params.model.insert(params.model.end(), p.begin(), p.end());
        } else if (arg == "-p" || arg == "--n-prompt") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<int>(argv[i], split_delim);
            params.n_prompt.insert(params.n_prompt.end(), p.begin(), p.end());
        } else if (arg == "-n" || arg == "--n-gen") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<int>(argv[i], split_delim);
            params.n_gen.insert(params.n_gen.end(), p.begin(), p.end());
        } else if (arg == "-pg") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<std::string>(argv[i], ',');
            if (p.size() != 2) {
                invalid_param = true;
                break;
            }
            params.n_pg.push_back({std::stoi(p[0]), std::stoi(p[1])});
        } else if (arg == "-gp") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<std::string>(argv[i], ',');
            if (p.size() != 2) {
                invalid_param = true;
                break;
            }
            params.n_gp.push_back({ std::stoi(p[0]), std::stoi(p[1]) });
        } else if (arg == "-b" || arg == "--batch-size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<int>(argv[i], split_delim);
            params.n_batch.insert(params.n_batch.end(), p.begin(), p.end());
        } else if (arg == "-ub" || arg == "--ubatch-size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<int>(argv[i], split_delim);
            params.n_ubatch.insert(params.n_ubatch.end(), p.begin(), p.end());
        } else if (arg == "-ctk" || arg == "--cache-type-k") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<std::string>(argv[i], split_delim);
            std::vector<ggml_type> types;
            for (const auto & t : p) {
                ggml_type gt = ggml_type_from_name(t);
                if (gt == GGML_TYPE_COUNT) {
                    invalid_param = true;
                    break;
                }
                types.push_back(gt);
            }
            params.type_k.insert(params.type_k.end(), types.begin(), types.end());
        } else if (arg == "-ctv" || arg == "--cache-type-v") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<std::string>(argv[i], split_delim);
            std::vector<ggml_type> types;
            for (const auto & t : p) {
                ggml_type gt = ggml_type_from_name(t);
                if (gt == GGML_TYPE_COUNT) {
                    invalid_param = true;
                    break;
                }
                types.push_back(gt);
            }
            params.type_v.insert(params.type_v.end(), types.begin(), types.end());
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<int>(argv[i], split_delim);
            params.n_threads.reserve(params.n_threads.size() + p.size());
            for (auto t : p) params.n_threads.push_back({t, t});
            //params.n_threads.insert(params.n_threads.end(), p.begin(), p.end());
        } else if (arg == "-tgb" || arg == "--threads-gen-batch") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto ps = string_split<std::string>(argv[i], ';');
            for (auto& s : ps) {
                auto p = string_split<int>(s.c_str(), ',');
                if (p.size() != 2) {
                    invalid_param = true;
                    break;
                }
                params.n_threads.push_back({p[0], p[1]});
            }
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<int>(argv[i], split_delim);
            params.n_gpu_layers.insert(params.n_gpu_layers.end(), p.begin(), p.end());
        } else if (arg == "-rpc" || arg == "--rpc") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.rpc_servers.push_back(argv[i]);
        } else if (arg == "-sm" || arg == "--split-mode") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<std::string>(argv[i], split_delim);
            std::vector<llama_split_mode> modes;
            for (const auto & m : p) {
                llama_split_mode mode;
                if (m == "none") {
                    mode = LLAMA_SPLIT_MODE_NONE;
                } else if (m == "layer") {
                    mode = LLAMA_SPLIT_MODE_LAYER;
                } else if (m == "graph") {
                    mode = LLAMA_SPLIT_MODE_GRAPH;
                } else {
                    invalid_param = true;
                    break;
                }
                modes.push_back(mode);
            }
            params.split_mode.insert(params.split_mode.end(), modes.begin(), modes.end());
        } else if (arg == "-mg" || arg == "--main-gpu") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.main_gpu = string_split<int>(argv[i], split_delim);
        } else if (arg == "-nkvo" || arg == "--no-kv-offload") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<bool>(argv[i], split_delim);
            params.no_kv_offload.insert(params.no_kv_offload.end(), p.begin(), p.end());
        } else if (arg == "--numa") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            } else {
                std::string value(argv[i]);
                /**/ if (value == "distribute" || value == "" ) { params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE; }
                else if (value == "isolate")                    { params.numa = GGML_NUMA_STRATEGY_ISOLATE; }
                else if (value == "numactl")                    { params.numa = GGML_NUMA_STRATEGY_NUMACTL; }
                else { invalid_param = true; break; }
            }
        } else if (arg == "-fa" || arg == "--flash-attn") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<bool>(argv[i], split_delim);
            params.flash_attn.insert(params.flash_attn.end(), p.begin(), p.end());
        } else if (arg == "-mla" || arg == "--mla-attn") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<int>(argv[i], split_delim);
            params.mla_attn.insert(params.mla_attn.end(), p.begin(), p.end());
        } else if (arg == "-amb" || arg == "--attn-max-batch") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<int>(argv[i], split_delim);
            params.attn_max_batch.insert(params.attn_max_batch.end(), p.begin(), p.end());
        } else if (arg == "-gr" || arg == "--graph-reuse") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<bool>(argv[i], split_delim);
            params.reuse.insert(params.reuse.end(), p.begin(), p.end());
        } else if (arg == "-ser" || arg == "--smart-expert-reduction") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split_pairs<int,float>(argv[i], split_delim);
            params.ser.insert(params.ser.end(), p.begin(), p.end());
        } else if (arg == "-mmp" || arg == "--mmap") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<bool>(argv[i], split_delim);
            params.use_mmap.insert(params.use_mmap.end(), p.begin(), p.end());
        } else if (arg == "-embd" || arg == "--embeddings") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<bool>(argv[i], split_delim);
            params.embeddings.insert(params.embeddings.end(), p.begin(), p.end());
        } else if (arg == "-ts" || arg == "--tensor-split") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            for (auto ts : string_split<std::string>(argv[i], split_delim)) {
                // split string by ; and /
                const std::regex regex{R"([;/]+)"};
                std::sregex_token_iterator it{ts.begin(), ts.end(), regex, -1};
                std::vector<std::string> split_arg{it, {}};
                GGML_ASSERT(split_arg.size() <= llama_max_devices());

                std::vector<float> tensor_split(llama_max_devices());
                for (size_t i = 0; i < llama_max_devices(); ++i) {
                    if (i < split_arg.size()) {
                        tensor_split[i] = std::stof(split_arg[i]);
                    } else {
                        tensor_split[i] = 0.0f;
                    }
                }
                params.tensor_split.push_back(tensor_split);
            }
        } else if (arg == "-r" || arg == "--repetitions") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.reps = std::stoi(argv[i]);
        } else if (arg == "-o" || arg == "--output") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            invalid_param = !output_format_from_str(argv[i], params.output_format);
        } else if (arg == "-oe" || arg == "--output-err") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            invalid_param = !output_format_from_str(argv[i], params.output_format_stderr);
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "-w" || arg == "--warmup") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.warmup = std::stoi(argv[i]);
        } else if (arg == "-rtr" || arg == "--run-time-repack") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.repack = std::stoi(argv[i]);
        } else if (arg == "-cuda" || arg == "--cuda-params") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.cuda_params = argv[i];
        } else if (arg == "-mqkv" || arg == "--merge-qkv") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.mqkv = std::stoi(argv[i]);
        } else if (arg == "-muge" || arg == "--merge-up-gate-exps") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.muge = std::stoi(argv[i]);
        } else if (arg == "--defer-experts") {
            params.defer_experts = true;
        } else if (arg == "-sas" || arg == "--scheduler-async") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.sas = std::stoi(argv[i]);
        } else if (arg == "--fit") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.fit = std::stoi(argv[i]);
        } else if (arg == "--fit-margin") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.fit_margin = std::stoi(argv[i]);
        } else if (arg == "--max-gpu") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.max_gpu = std::stoi(argv[i]);
        } else if (arg == "-rcache" || arg == "--rope-cache") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.rcache = std::stoi(argv[i]);
        } else if (arg == "-thp" || arg == "--transparent-huge-pages") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.use_thp = std::stoi(argv[i]);
        } else if (arg == "-fmoe" || arg == "--fused-moe") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.fmoe = std::stoi(argv[i]);
        } else if (arg == "-ger" || arg == "--grouped-expert-routing") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.ger = std::stoi(argv[i]);
        } else if (arg == "-no-fug" || arg == "--no-fused-up-gate") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.no_fug = std::stoi(argv[i]);
        } else if (arg == "-no-ooae" || arg == "--no-offload-only-active-experts") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.no_ooae = std::stoi(argv[i]);
        } else if (arg == "-ot" || arg == "--override-tensor") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            if (!parse_buft_overrides(std::string{argv[i]}, params.buft_overrides)) {
                fprintf(stderr, "error: Invalid tensor buffer type override: %s\n", argv[i]);
                invalid_param = true;
                break;
            }
        } else if (arg == "--n-cpu-moe") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            if (!add_cpu_buft_overrides(argv[i], params.buft_overrides)) {
                invalid_param = true;
                break;
            }
        } else if (arg == "--print-overrides") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.print_overrides = std::stoi(argv[i]);
        } else if (arg == "--spec") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto names = string_split<std::string>(argv[i], split_delim);
            for (const auto & name : names) {
                common_speculative_type st = common_speculative_type_from_name(name);
                if (st == COMMON_SPECULATIVE_TYPE_COUNT) {
                    fprintf(stderr, "error: unknown spec method '%s' (valid: %s)\n",
                            name.c_str(), common_speculative_type_name_str().c_str());
                    invalid_param = true;
                    break;
                }
                params.spec_type.push_back(st);
            }
            if (invalid_param) break;
        } else if (arg == "-nd" || arg == "--draft") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            auto p = string_split<int>(argv[i], split_delim);
            params.n_draft.insert(params.n_draft.end(), p.begin(), p.end());
        } else if (arg == "--spec-model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.spec_model.push_back(argv[i]);
        } else if (arg == "--prompt-file") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.prompt_files.push_back(argv[i]);
        } else if (arg == "--ppl-of-output") {
            params.ppl_of_output = true;
        } else {
            invalid_param = true;
            break;
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        print_usage(argc, argv);
        exit(1);
    }

    // set defaults
    if (params.model.empty())        { params.model = cmd_params_defaults.model; }
    if (params.n_prompt.empty())     { params.n_prompt = cmd_params_defaults.n_prompt; }
    if (params.n_gen.empty())        { params.n_gen = cmd_params_defaults.n_gen; }
    if (params.n_pg.empty())         { params.n_pg = cmd_params_defaults.n_pg; }
    if (params.n_gp.empty())         { params.n_gp = cmd_params_defaults.n_gp; }
    if (params.n_batch.empty())      { params.n_batch = cmd_params_defaults.n_batch; }
    if (params.n_ubatch.empty())     { params.n_ubatch = cmd_params_defaults.n_ubatch; }
    if (params.type_k.empty())       { params.type_k = cmd_params_defaults.type_k; }
    if (params.type_v.empty())       { params.type_v = cmd_params_defaults.type_v; }
    if (params.n_gpu_layers.empty()) { params.n_gpu_layers = cmd_params_defaults.n_gpu_layers; }
    if (params.rpc_servers.empty())  { params.rpc_servers = cmd_params_defaults.rpc_servers; }
    if (params.split_mode.empty())   { params.split_mode = cmd_params_defaults.split_mode; }
    if (params.main_gpu.empty())     { params.main_gpu = cmd_params_defaults.main_gpu; }
    if (params.no_kv_offload.empty()){ params.no_kv_offload = cmd_params_defaults.no_kv_offload; }
    if (params.flash_attn.empty())   { params.flash_attn = cmd_params_defaults.flash_attn; }
    if (params.mla_attn.empty())     { params.mla_attn = cmd_params_defaults.mla_attn; }
    if (params.attn_max_batch.empty()){ params.attn_max_batch = cmd_params_defaults.attn_max_batch; }
    if (params.reuse.empty())        { params.reuse = cmd_params_defaults.reuse; }
    if (params.ser.empty())          { params.ser = cmd_params_defaults.ser; }
    if (params.tensor_split.empty()) { params.tensor_split = cmd_params_defaults.tensor_split; }
    if (params.use_mmap.empty())     { params.use_mmap = cmd_params_defaults.use_mmap; }
    if (params.embeddings.empty())   { params.embeddings = cmd_params_defaults.embeddings; }
    if (params.n_threads.empty())    { params.n_threads = cmd_params_defaults.n_threads; }
    if (params.spec_type.empty())    { params.spec_type = cmd_params_defaults.spec_type; }
    if (params.n_draft.empty())      { params.n_draft = cmd_params_defaults.n_draft; }
    if (params.spec_model.empty())   { params.spec_model = cmd_params_defaults.spec_model; }
    if (params.prompt_files.empty()) { params.prompt_files = cmd_params_defaults.prompt_files; }
    if (!params.buft_overrides.empty()) params.buft_overrides.emplace_back(llama_model_tensor_buft_override{nullptr, nullptr});

    return params;
}

enum test_kind_type {
    // measure mean prompt processing rate without token generation
    TEST_KIND_PP,
    // measure mean token generation rate without prompt processing
    TEST_KIND_TG,
    // measure mean prompt processing and token generation rate
    TEST_KIND_PG,
    // measure mean token generation rate after processing prompt of given length
    TEST_KIND_GP,
};

struct cmd_params_instance {
    test_kind_type test_kind;
    std::string model;
    int n_prompt;
    int n_gen;
    int n_batch;
    int n_ubatch;
    ggml_type type_k;
    ggml_type type_v;
    std::pair<int,int> n_threads;
    int n_gpu_layers;
    std::string rpc_servers;
    llama_split_mode split_mode;
    int main_gpu;
    bool no_kv_offload;
    bool flash_attn;
    int  mla_attn;
    int  attn_max_batch;
    bool reuse;
    Ser  ser;
    std::vector<float> tensor_split;
    std::string cuda_params;
    bool use_mmap;
    bool embeddings;
    bool repack = false;
    bool fmoe = true;
    bool ger = false;
    bool no_fug = false;
    bool use_thp = false;
    bool no_ooae = false;
    bool mqkv = false;
    bool muge = false;
    bool defer_experts = false;
    bool rcache = false;
    bool sas = false;
    int max_gpu = 0;
    bool fit = false;
    int  fit_margin = 0;
    const llama_model_tensor_buft_override* buft_overrides;
    // T8 spec-aware bench
    common_speculative_type spec_type = COMMON_SPECULATIVE_TYPE_NONE;
    int         n_draft = 0;
    std::string spec_model;
    std::string prompt_file;
    bool        ppl_of_output = false;

    llama_model_params to_llama_mparams() const {
        llama_model_params mparams = llama_model_default_params();

        mparams.n_gpu_layers = n_gpu_layers;
        if (!rpc_servers.empty()) {
            mparams.rpc_servers = rpc_servers.c_str();
        }
        mparams.split_mode = split_mode;
        mparams.main_gpu = main_gpu;
        mparams.tensor_split = tensor_split.data();
        mparams.use_mmap = use_mmap;
        mparams.repack_tensors = repack;
        mparams.use_thp = use_thp;
        mparams.merge_qkv = mqkv;
        mparams.merge_up_gate_exps = muge;
        mparams.defer_experts = defer_experts;
        mparams.tensor_buft_overrides = buft_overrides;
        mparams.mla = mla_attn;
        mparams.max_gpu = max_gpu;
        mparams.fit = fit;
        mparams.fit_margin = fit_margin;
        mparams.type_k = type_k;
        mparams.type_v = type_v;

        return mparams;
    }

    bool equal_mparams(const cmd_params_instance & other) const {
        return model == other.model &&
               n_gpu_layers == other.n_gpu_layers &&
               rpc_servers == other.rpc_servers &&
               split_mode == other.split_mode &&
               main_gpu == other.main_gpu &&
               use_mmap == other.use_mmap &&
               repack == other.repack &&
               mqkv == other.mqkv &&
               muge == other.muge &&
               defer_experts == other.defer_experts &&
               use_thp == other.use_thp &&
               sas == other.sas &&
               fit == other.fit &&
               fit_margin == other.fit_margin &&
               max_gpu == other.max_gpu &&
               tensor_split == other.tensor_split;
    }

    llama_context_params to_llama_cparams() const {
        llama_context_params cparams = llama_context_default_params();

        cparams.n_ctx = n_prompt + n_gen;
        cparams.n_batch = n_batch;
        cparams.n_ubatch = n_ubatch;
        cparams.type_k = type_k;
        cparams.type_v = type_v;
        cparams.offload_kqv = !no_kv_offload;
        cparams.flash_attn = flash_attn;
        cparams.mla_attn = mla_attn;
        cparams.attn_max_batch = attn_max_batch;
        cparams.graph_reuse = reuse;
        cparams.fused_moe_up_gate = fmoe;
        cparams.grouped_expert_routing = ger;
        cparams.rope_cache = rcache;
        cparams.fused_up_gate = !no_fug;
        cparams.only_active_experts = !no_ooae;
        cparams.min_experts = ser.first;
        cparams.thresh_experts = ser.second;
        cparams.embeddings = embeddings;
        cparams.cuda_params = (void *)cuda_params.data();
        cparams.scheduler_async = sas;

        return cparams;
    }
};

static std::vector<cmd_params_instance> get_cmd_params_instances(const cmd_params & params) {
    std::vector<cmd_params_instance> instances;

    // this ordering minimizes the number of times that each model needs to be reloaded
    for (const auto & m : params.model)
    for (const auto & nl : params.n_gpu_layers)
    for (const auto & rpc : params.rpc_servers)
    for (const auto & sm : params.split_mode)
    for (const auto & mg : params.main_gpu)
    for (const auto & ts : params.tensor_split)
    for (const auto & mmp : params.use_mmap)
    for (const auto & embd : params.embeddings)
    for (const auto & nb : params.n_batch)
    for (const auto & nub : params.n_ubatch)
    for (const auto & tk : params.type_k)
    for (const auto & tv : params.type_v)
    for (const auto & nkvo : params.no_kv_offload)
    for (const auto & fa : params.flash_attn)
    for (const auto & mla : params.mla_attn)
    for (const auto & amb : params.attn_max_batch)
    for (const auto & reuse : params.reuse)
    for (const auto & ser : params.ser)
    for (const auto & nt : params.n_threads)
    for (const auto & st : params.spec_type)
    for (const auto & nd : params.n_draft)
    for (const auto & sm_path : params.spec_model)
    for (const auto & prompt_file : params.prompt_files) {
        for (const auto & n_prompt : params.n_prompt) {
            if (n_prompt == 0) {
                continue;
            }
            // PP-only rows are spec-independent (prefill happens before spec init).
            // When sweeping multiple --spec methods, only emit one PP row to avoid
            // wasting compute on identical prefills.
            if (st != params.spec_type.front()) {
                continue;
            }
            cmd_params_instance instance = {
                /* .test_kind    = */ TEST_KIND_PP,
                /* .model        = */ m,
                /* .n_prompt     = */ n_prompt,
                /* .n_gen        = */ 0,
                /* .n_batch      = */ nb,
                /* .n_ubatch     = */ nub,
                /* .type_k       = */ tk,
                /* .type_v       = */ tv,
                /* .n_threads    = */ nt,
                /* .n_gpu_layers = */ nl,
                /* .rpc_servers  = */ rpc,
                /* .split_mode   = */ sm,
                /* .main_gpu     = */ mg,
                /* .no_kv_offload= */ nkvo,
                /* .flash_attn   = */ fa,
                /* .mla_attn     = */ mla,
                /* .attn_max_b   = */ amb,
                /* .reuse        = */ reuse,
                /* .ser          = */ ser,
                /* .tensor_split = */ ts,
                /* .cuda_params  = */ params.cuda_params,
                /* .use_mmap     = */ mmp,
                /* .embeddings   = */ embd,
                /* .repack       = */ params.repack,
                /* .fmoe         = */ params.fmoe,
                /* .ger          = */ params.ger,
                /* .no_fug       = */ params.no_fug,
                /* .use_thp      = */ params.use_thp,
                /* .no_ooae      = */ params.no_ooae,
                /* .mqkv         = */ params.mqkv,
                /* .muge         = */ params.muge,
                /* .defer_experts= */ params.defer_experts,
                /* .rcache       = */ params.rcache,
                /* .sas          = */ params.sas,
                /* .max_gpu      = */ params.max_gpu,
                /* .fit          = */ params.fit,
                /* .git_margin   = */ params.fit_margin,
                /* .buft_overrides=*/ params.buft_overrides.data(),
                /* .spec_type    = */ st,
                /* .n_draft      = */ nd,
                /* .spec_model   = */ sm_path,
                /* .prompt_file  = */ prompt_file,
                /* .ppl_of_output*/ params.ppl_of_output,
            };
            instances.push_back(instance);
        }

        for (const auto & n_gen : params.n_gen) {
            if (n_gen == 0) {
                continue;
            }
            cmd_params_instance instance = {
                /* .test_kind    = */ TEST_KIND_TG,
                /* .model        = */ m,
                /* .n_prompt     = */ 0,
                /* .n_gen        = */ n_gen,
                /* .n_batch      = */ nb,
                /* .n_ubatch     = */ nub,
                /* .type_k       = */ tk,
                /* .type_v       = */ tv,
                /* .n_threads    = */ nt,
                /* .n_gpu_layers = */ nl,
                /* .rpc_servers  = */ rpc,
                /* .split_mode   = */ sm,
                /* .main_gpu     = */ mg,
                /* .no_kv_offload= */ nkvo,
                /* .flash_attn   = */ fa,
                /* .mla_attn     = */ mla,
                /* .attn_max_b   = */ amb,
                /* .reuse        = */ reuse,
                /* .ser          = */ ser,
                /* .tensor_split = */ ts,
                /* .cuda_params  = */ params.cuda_params,
                /* .use_mmap     = */ mmp,
                /* .embeddings   = */ embd,
                /* .repack       = */ params.repack,
                /* .fmoe         = */ params.fmoe,
                /* .ger          = */ params.ger,
                /* .no_fug       = */ params.no_fug,
                /* .use_thp      = */ params.use_thp,
                /* .no_ooae      = */ params.no_ooae,
                /* .mqkv         = */ params.mqkv,
                /* .muge         = */ params.muge,
                /* .defer_experts= */ params.defer_experts,
                /* .rcache       = */ params.rcache,
                /* .sas          = */ params.sas,
                /* .max_gpu      = */ params.max_gpu,
                /* .fit          = */ params.fit,
                /* .git_margin   = */ params.fit_margin,
                /* .buft_overrides=*/ params.buft_overrides.data(),
                /* .spec_type    = */ st,
                /* .n_draft      = */ nd,
                /* .spec_model   = */ sm_path,
                /* .prompt_file  = */ prompt_file,
                /* .ppl_of_output*/ params.ppl_of_output,
            };
            instances.push_back(instance);
        }

        for (const auto & n_pg : params.n_pg) {
            if (n_pg.first == 0 && n_pg.second == 0) {
                continue;
            }
            cmd_params_instance instance = {
                /* .test_kind    = */ TEST_KIND_PG,
                /* .model        = */ m,
                /* .n_prompt     = */ n_pg.first,
                /* .n_gen        = */ n_pg.second,
                /* .n_batch      = */ nb,
                /* .n_ubatch     = */ nub,
                /* .type_k       = */ tk,
                /* .type_v       = */ tv,
                /* .n_threads    = */ nt,
                /* .n_gpu_layers = */ nl,
                /* .rpc_servers  = */ rpc,
                /* .split_mode   = */ sm,
                /* .main_gpu     = */ mg,
                /* .no_kv_offload= */ nkvo,
                /* .flash_attn   = */ fa,
                /* .mla_attn     = */ mla,
                /* .attn_max_b   = */ amb,
                /* .reuse        = */ reuse,
                /* .ser          = */ ser,
                /* .tensor_split = */ ts,
                /* .cuda_params  = */ params.cuda_params,
                /* .use_mmap     = */ mmp,
                /* .embeddings   = */ embd,
                /* .repack       = */ params.repack,
                /* .fmoe         = */ params.fmoe,
                /* .ger          = */ params.ger,
                /* .no_fug       = */ params.no_fug,
                /* .use_thp      = */ params.use_thp,
                /* .no_ooae      = */ params.no_ooae,
                /* .mqkv         = */ params.mqkv,
                /* .muge         = */ params.muge,
                /* .defer_experts= */ params.defer_experts,
                /* .rcache       = */ params.rcache,
                /* .sas          = */ params.sas,
                /* .max_gpu      = */ params.max_gpu,
                /* .fit          = */ params.fit,
                /* .git_margin   = */ params.fit_margin,
                /* .buft_overrides=*/ params.buft_overrides.data(),
                /* .spec_type    = */ st,
                /* .n_draft      = */ nd,
                /* .spec_model   = */ sm_path,
                /* .prompt_file  = */ prompt_file,
                /* .ppl_of_output*/ params.ppl_of_output,
            };
            instances.push_back(instance);
        }

        for (const auto & n_gp : params.n_gp) {
            if (n_gp.first == 0 && n_gp.second == 0) {
                continue;
            }
            cmd_params_instance instance = {
                /* .test_kind    = */ TEST_KIND_GP,
                /* .model        = */ m,
                /* .n_prompt     = */ n_gp.first,
                /* .n_gen        = */ n_gp.second,
                /* .n_batch      = */ nb,
                /* .n_ubatch     = */ nub,
                /* .type_k       = */ tk,
                /* .type_v       = */ tv,
                /* .n_threads    = */ nt,
                /* .n_gpu_layers = */ nl,
                /* .rpc_servers  = */ rpc,
                /* .split_mode   = */ sm,
                /* .main_gpu     = */ mg,
                /* .no_kv_offload= */ nkvo,
                /* .flash_attn   = */ fa,
                /* .mla_attn     = */ mla,
                /* .attn_max_b   = */ amb,
                /* .reuse        = */ reuse,
                /* .ser          = */ ser,
                /* .tensor_split = */ ts,
                /* .cuda_params  = */ params.cuda_params,
                /* .use_mmap     = */ mmp,
                /* .embeddings   = */ embd,
                /* .repack       = */ params.repack,
                /* .fmoe         = */ params.fmoe,
                /* .ger          = */ params.ger,
                /* .no_fug       = */ params.no_fug,
                /* .use_thp      = */ params.use_thp,
                /* .no_ooae      = */ params.no_ooae,
                /* .mqkv         = */ params.mqkv,
                /* .muge         = */ params.muge,
                /* .defer_experts= */ params.defer_experts,
                /* .rcache       = */ params.rcache,
                /* .sas          = */ params.sas,
                /* .max_gpu      = */ params.max_gpu,
                /* .fit          = */ params.fit,
                /* .git_margin   = */ params.fit_margin,
                /* .buft_overrides=*/ params.buft_overrides.data(),
                /* .spec_type    = */ st,
                /* .n_draft      = */ nd,
                /* .spec_model   = */ sm_path,
                /* .prompt_file  = */ prompt_file,
                /* .ppl_of_output*/ params.ppl_of_output,
            };
            instances.push_back(instance);
        }
    }

    return instances;
}

struct test {
    static const std::string build_commit;
    static const int build_number;
    static const bool cuda;
    static const bool vulkan;
    static const bool kompute;
    static const bool metal;
    static const bool sycl;
    static const bool gpu_blas;
    static const bool blas;
    static const std::string cpu_info;
    static const std::string gpu_info;
    std::string model_filename;
    std::string model_type;
    uint64_t model_size;
    uint64_t model_n_params;
    int n_batch;
    int n_ubatch;
    std::pair<int,int> n_threads;
    bool has_rpc;
    ggml_type type_k;
    ggml_type type_v;
    int n_gpu_layers;
    llama_split_mode split_mode;
    int main_gpu;
    bool no_kv_offload;
    bool flash_attn;
    int  mla_attn;
    int  attn_max_batch;
    bool reuse;
    Ser  ser;
    std::vector<float> tensor_split;
    std::string cuda_params;
    bool use_mmap;
    bool embeddings;
    bool repack = false;
    bool fmoe = false;
    bool ger = false;
    bool no_fug = false;
    bool use_thp = false;
    bool no_ooae = false;
    bool mqkv = false;
    bool muge = false;
    bool defer_experts = false;
    bool rcache = false;
    bool sas = false;
    bool max_gpu = 0;
    bool fit = false;
    int  fit_margin = 0;
    std::string override_tensor;
    int n_prompt;
    int n_gen;
    std::string test_time;
    std::vector<uint64_t> samples_ns;
    test_kind_type  test_kind;
    std::string     test_label;
    // T8 spec-aware bench result fields. Populated for TG-class kinds; left
    // at defaults for PP. accept_rate / mean_accept are 0 for spec=none.
    common_speculative_type spec_type = COMMON_SPECULATIVE_TYPE_NONE;
    int         n_draft       = 0;
    std::string spec_model;
    std::string prompt_file;
    int         n_drafts      = 0;     // total draft cycles across reps
    int         n_accepted    = 0;     // total accepted draft tokens across reps
    int         n_draft_total = 0;     // total candidate draft tokens (= cycles * BS)
    double      ppl_of_output = 0.0;   // avg corpus-PPL of generated output (target re-decode)
    bool        has_ppl       = false;

    test(const cmd_params_instance & inst, const llama_model * lmodel, const llama_context * ctx) {
        model_filename = inst.model;
        char buf[128];
        llama_model_desc(lmodel, buf, sizeof(buf));
        model_type = buf;
        model_size = llama_model_size(lmodel);
        model_n_params = llama_model_n_params(lmodel);
        n_batch = inst.n_batch;
        n_ubatch = inst.n_ubatch;
        n_threads = inst.n_threads;
        has_rpc = !inst.rpc_servers.empty();
        type_k = inst.type_k;
        type_v = inst.type_v;
        n_gpu_layers = inst.n_gpu_layers;
        split_mode = inst.split_mode;
        main_gpu = inst.main_gpu;
        no_kv_offload = inst.no_kv_offload;
        flash_attn = inst.flash_attn;
        mla_attn = inst.mla_attn;
        attn_max_batch = inst.attn_max_batch;
        reuse = inst.reuse;
        ser = inst.ser;
        tensor_split = inst.tensor_split;
        cuda_params = inst.cuda_params;
        use_mmap = inst.use_mmap;
        embeddings = inst.embeddings;
        repack = inst.repack;
        mqkv = inst.mqkv;
        muge = inst.muge;
        defer_experts = inst.defer_experts;
        fmoe = inst.fmoe;
        ger = inst.ger;
        rcache = inst.rcache;
        sas = inst.sas;
        max_gpu = inst.max_gpu;
        fit = inst.fit;
        fit_margin = inst.fit_margin;
        no_fug = inst.no_fug;
        use_thp = inst.use_thp;
        no_ooae = inst.no_ooae;
        if (inst.buft_overrides) {
            const auto * bo = inst.buft_overrides;
            while (bo->pattern) {
                if (!override_tensor.empty()) {
                    override_tensor += ",";
                }
                override_tensor += bo->pattern;
                override_tensor += "=";
                override_tensor += ggml_backend_buft_name(bo->buft);
                ++bo;
            }
        }

        n_prompt = inst.n_prompt;
        n_gen = inst.n_gen;
        test_kind = inst.test_kind;
        spec_type = inst.spec_type;
        n_draft = inst.n_draft;
        spec_model = inst.spec_model;
        prompt_file = inst.prompt_file;
        // RFC 3339 date-time format
        time_t t = time(NULL);
        std::strftime(buf, sizeof(buf), "%FT%TZ", gmtime(&t));
        test_time = buf;

        // prepare test label for printing
        switch (test_kind) {
            case TEST_KIND_PP:
                snprintf(buf, sizeof(buf), "pp%d", n_prompt);
                break;
            case TEST_KIND_TG:
                snprintf(buf, sizeof(buf), "tg%d", n_gen);
                break;
            case TEST_KIND_PG:
                snprintf(buf, sizeof(buf), "pp%d+tg%d", n_prompt, n_gen);
                break;
            case TEST_KIND_GP:
                snprintf(buf, sizeof(buf), "tg%d@pp%d", n_gen, n_prompt);
                break;
            default:
                snprintf(buf, sizeof(buf), "unknown");
                break;
        }
        test_label = buf;

        (void) ctx;
    }

    uint64_t avg_ns() const {
        return ::avg(samples_ns);
    }

    uint64_t stdev_ns() const {
        return ::stdev(samples_ns);
    }

    std::vector<double> get_ts() const {
        int n_tokens = (test_kind == TEST_KIND_GP ? 0 : n_prompt) + n_gen;
        std::vector<double> ts;
        std::transform(samples_ns.begin(), samples_ns.end(), std::back_inserter(ts), [n_tokens](uint64_t t) { return 1e9 * n_tokens / t; });
        return ts;
    }

    double avg_ts() const {
        return ::avg(get_ts());
    }

    double stdev_ts() const {
        return ::stdev(get_ts());
    }

    static std::string get_backend() {
        if (cuda) {
            return GGML_CUDA_NAME;
        }
        if (vulkan) {
            return "Vulkan";
        }
        if (kompute) {
            return "Kompute";
        }
        if (metal) {
            return "Metal";
        }
        if (sycl) {
            return GGML_SYCL_NAME;
        }
        if (gpu_blas) {
            return "GPU BLAS";
        }
        if (blas) {
            return "BLAS";
        }

        return "CPU";
    }

    enum field_type {STRING, BOOL, INT, FLOAT};

    static field_type get_field_type(const std::string & field) {
        if (field == "build_number" || field == "n_batch" || field == "n_ubatch" ||
            field == "n_threads" ||
            field == "model_size" || field == "model_n_params" ||
            field == "n_gpu_layers" || field == "main_gpu" ||
            field == "n_prompt" || field == "n_gen" || field == "mla_attn" || field == "attn_max_batch" ||
            field == "avg_ns" || field == "stddev_ns" || field == "max_gpu" ||
            field == "n_draft" || field == "n_drafts" || field == "n_accepted" || field == "n_draft_total") {
            return INT;
        }
        if (field == "cuda" || field == "vulkan" || field == "kompute" || field == "metal" ||
            field == "gpu_blas" || field == "blas" || field == "sycl" || field == "no_kv_offload" ||
            field == "flash_attn" || field == "use_mmap" || field == "embeddings" || field == "repack" || field == "use_thp" ||
            field == "fused_moe" || field == "grouped_er" || field == "no_fused_up_gate" || field == "no_ooae" || field == "mqkv" ||
            field == "rcache" || field == "reuse" || field == "muge" || field == "defer_experts" || field == "sas") {
            return BOOL;
        }
        if (field == "avg_ts" || field == "stddev_ts" ||
            field == "accept_rate" || field == "mean_accept" || field == "target_ppl_of_output") {
            return FLOAT;
        }
        return STRING;
    }

    std::vector<std::string> get_values() const {
        std::string tensor_split_str;
        int max_nonzero = 0;
        for (size_t i = 0; i < llama_max_devices(); i++) {
            if (tensor_split[i] > 0) {
                max_nonzero = i;
            }
        }
        for (int i = 0; i <= max_nonzero; i++) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.2f", tensor_split[i]);
            tensor_split_str += buf;
            if (i < max_nonzero) {
                tensor_split_str += "/";
            }
        }
        auto ser_to_string = [] (const Ser& ser) {
            std::ostringstream str;
            str << ser.first << ',' << ser.second;
            return str.str();
        };
        bool is_gen = n_gen > 0;
        char buf_ar[32];
        char buf_ma[32];
        char buf_ppl[32];
        snprintf(buf_ar,  sizeof(buf_ar),  "%.4f", accept_rate());
        snprintf(buf_ma,  sizeof(buf_ma),  "%.3f", mean_accept());
        snprintf(buf_ppl, sizeof(buf_ppl), "%.4f", ppl_of_output);
        std::vector<std::string> values = {
            build_commit, std::to_string(build_number),
            std::to_string(cuda), std::to_string(vulkan), std::to_string(kompute),
            std::to_string(metal), std::to_string(sycl), std::to_string(has_rpc), std::to_string(gpu_blas), std::to_string(blas),
            cpu_info, gpu_info,
            model_filename, model_type, std::to_string(model_size), std::to_string(model_n_params),
            std::to_string(n_batch), std::to_string(n_ubatch),
            std::to_string(is_gen ? n_threads.first : n_threads.second), ggml_type_name(type_k), ggml_type_name(type_v),
            std::to_string(n_gpu_layers), split_mode_str(split_mode),
            std::to_string(main_gpu), std::to_string(no_kv_offload), std::to_string(flash_attn),
            std::to_string(mla_attn), std::to_string(attn_max_batch), ser_to_string(ser), std::to_string(reuse),
            tensor_split_str, std::to_string(use_mmap), std::to_string(embeddings),
            std::to_string(repack), std::to_string(mqkv), std::to_string(muge), std::to_string(defer_experts), std::to_string(fmoe), std::to_string(ger),
            std::to_string(no_fug), std::to_string(use_thp), std::to_string(no_ooae), std::to_string(rcache), std::to_string(sas),
            std::to_string(max_gpu),
            cuda_params, override_tensor,
            std::to_string(n_prompt), std::to_string(n_gen), test_time,
            std::to_string(avg_ns()), std::to_string(stdev_ns()),
            std::to_string(avg_ts()), std::to_string(stdev_ts()),
            test_label,
            common_speculative_type_to_str(spec_type),
            std::to_string(n_draft),
            spec_model,
            prompt_file,
            std::to_string(n_drafts),
            std::to_string(n_accepted),
            std::to_string(n_draft_total),
            buf_ar,
            buf_ma,
            has_ppl ? std::string(buf_ppl) : std::string("-")
        };
        return values;
    }

    // T8: accept_rate = accepted / total_candidates, mean_accept = accepted / drafts
    double accept_rate() const {
        return n_draft_total > 0 ? double(n_accepted) / double(n_draft_total) : 0.0;
    }
    double mean_accept() const {
        return n_drafts > 0 ? double(n_accepted) / double(n_drafts) : 0.0;
    }

    static const std::vector<std::string> & get_fields() {
        static const std::vector<std::string> fields = {
            "build_commit", "build_number",
            "cuda", "vulkan", "kompute", "metal", "sycl", "rpc", "gpu_blas", "blas",
            "cpu_info", "gpu_info",
            "model_filename", "model_type", "model_size", "model_n_params",
            "n_batch", "n_ubatch",
            "n_threads", "type_k", "type_v",
            "n_gpu_layers", "split_mode",
            "main_gpu", "no_kv_offload", "flash_attn", "mla_attn", "attn_max_batch", "ser", "reuse",
            "tensor_split", "use_mmap", "embeddings", "repack", "mqkv", "muge", "defer_experts", "fused_moe", "grouped_er",
            "no_fused_up_gate", "use_thp", "no_ooae", "rcache", "sas", "max_gpu", "cuda_params", "override_tensor",
            "n_prompt", "n_gen", "test_time",
            "avg_ns", "stddev_ns",
            "avg_ts", "stddev_ts", "test",
            "spec", "n_draft", "spec_model", "prompt_file",
            "n_drafts", "n_accepted", "n_draft_total",
            "accept_rate", "mean_accept", "target_ppl_of_output",
        };
        return fields;
    }

    std::map<std::string, std::string> get_map() const {
        std::map<std::string, std::string> map;
        auto fields = get_fields();
        auto values = get_values();
        std::transform(fields.begin(), fields.end(), values.begin(),
                std::inserter(map, map.end()), std::make_pair<const std::string &, const std::string &>);
        return map;
    }
};

const std::string test::build_commit = LLAMA_COMMIT;
const int         test::build_number = LLAMA_BUILD_NUMBER;
const bool        test::cuda         = !!ggml_cpu_has_cuda();
const bool        test::vulkan       = !!ggml_cpu_has_vulkan();
const bool        test::kompute      = !!ggml_cpu_has_kompute();
const bool        test::metal        = !!ggml_cpu_has_metal();
const bool        test::gpu_blas     = !!ggml_cpu_has_gpublas();
const bool        test::blas         = !!ggml_cpu_has_blas();
const bool        test::sycl         = !!ggml_cpu_has_sycl();
const std::string test::cpu_info     = get_cpu_info();
const std::string test::gpu_info     = get_gpu_info();

struct printer {
    virtual ~printer() {}

    FILE * fout;
    virtual void print_header(const cmd_params & params) { (void) params; }
    virtual void print_test(const test & t) = 0;
    virtual void print_footer() { }
};

struct csv_printer : public printer {
    static std::string escape_csv(const std::string & field) {
        std::string escaped = "\"";
        for (auto c : field) {
            if (c == '"') {
                escaped += "\"";
            }
            escaped += c;
        }
        escaped += "\"";
        return escaped;
    }

    void print_header(const cmd_params & params) override  {
        std::vector<std::string> fields = test::get_fields();
        fprintf(fout, "%s\n", join(fields, ",").c_str());
        (void) params;
    }

    void print_test(const test & t) override {
        std::vector<std::string> values = t.get_values();
        std::transform(values.begin(), values.end(), values.begin(), escape_csv);
        fprintf(fout, "%s\n", join(values, ",").c_str());
    }
};

struct json_printer : public printer {
    bool first = true;

    static std::string escape_json(const std::string & value) {
        std::string escaped;
        for (auto c : value) {
            if (c == '"') {
                escaped += "\\\"";
            } else if (c == '\\') {
                escaped += "\\\\";
            } else  if (c <= 0x1f) {
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", c);
                escaped += buf;
            } else {
                escaped += c;
            }
        }
        return escaped;
    }

    static std::string format_value(const std::string & field, const std::string & value) {
        switch (test::get_field_type(field)) {
            case test::STRING:
                return "\"" + escape_json(value) + "\"";
            case test::BOOL:
                return value == "0" ? "false" : "true";
            default:
                return value;
        }
    }

    void print_header(const cmd_params & params) override {
        fprintf(fout, "[\n");
        (void) params;
    }

    void print_fields(const std::vector<std::string> & fields, const std::vector<std::string> & values) {
        assert(fields.size() == values.size());
        for (size_t i = 0; i < fields.size(); i++) {
            fprintf(fout, "    \"%s\": %s,\n", fields.at(i).c_str(), format_value(fields.at(i), values.at(i)).c_str());
        }
    }

    void print_test(const test & t) override {
        if (first) {
            first = false;
        } else {
            fprintf(fout, ",\n");
        }
        fprintf(fout, "  {\n");
        print_fields(test::get_fields(), t.get_values());
        fprintf(fout, "    \"samples_ns\": [ %s ],\n", join(t.samples_ns, ", ").c_str());
        fprintf(fout, "    \"samples_ts\": [ %s ]\n", join(t.get_ts(), ", ").c_str());
        fprintf(fout, "  }");
        fflush(fout);
    }

    void print_footer() override {
        fprintf(fout, "\n]\n");
    }
};

struct markdown_printer : public printer {
    std::vector<std::string> fields;
    bool skipped_overrides = false;

    static int get_field_width(const std::string & field) {
        if (field == "model") {
            return -30;
        }
        if (field == "t/s") {
            return 16;
        }
        if (field == "size" || field == "params") {
            return 10;
        }
        if (field == "n_gpu_layers") {
            return 3;
        }
        if (field == "n_threads") {
            return 7;
        }
        if (field == "n_batch") {
            return 7;
        }
        if (field == "n_ubatch") {
            return 8;
        }
        if (field == "type_k" || field == "type_v") {
            return 6;
        }
        if (field == "split_mode") {
            return 5;
        }
        if (field == "flash_attn") {
            return 2;
        }
        if (field == "mla_attn") {
            return 3;
        }
        if (field == "attn_max_batch") {
            return 5;
        }
        if (field == "reuse") {
            return 2;
        }
        if (field == "ser") {
            return 10;
        }
        if (field == "use_mmap") {
            return 4;
        }
        if (field == "repack") {
            return 3;
        }
        if (field == "mqkv") {
            return 4;
        }
        if (field == "muge") {
            return 4;
        }
        if (field == "defer_experts") {
            return 5;
        }
        if (field == "sas") {
            return 3;
        }
        if (field == "max_gpu") {
            return 7;
        }
        if (field == "use_thp") {
            return 3;
        }
        if (field == "fused_moe") {
            return 4;
        }
        if (field == "grouped_er") {
            return 3;
        }
        if (field == "rcache") {
            return 6;
        }
        if (field == "no_fused_up_gate") {
            return 6;
        }
        if (field == "no_ooae") {
            return 7;
        }
        if (field == "test") {
            return 13;
        }
        if (field == "spec") {
            return 12;
        }
        if (field == "n_draft") {
            return 3;
        }
        if (field == "accept_rate" || field == "mean_accept") {
            return 8;
        }
        if (field == "target_ppl_of_output") {
            return 10;
        }
        if (field == "prompt_file") {
            return -24;
        }

        int width = std::max((int)field.length(), 10);

        if (test::get_field_type(field) == test::STRING) {
            return -width;
        }
        return width;
    }

    static std::string get_field_display_name(const std::string & field) {
        if (field == "n_gpu_layers") {
            return "ngl";
        }
        if (field == "split_mode") {
            return "sm";
        }
        if (field == "n_threads") {
            return "threads";
        }
        if (field == "no_kv_offload") {
            return "nkvo";
        }
        if (field == "flash_attn") {
            return "fa";
        }
        if (field == "mla_attn") {
            return "mla";
        }
        if (field == "attn_max_batch") {
            return "amb";
        }
        if (field == "reuse") {
            return "gr";
        }
        if (field == "ser") {
            return "ser";
        }
        if (field == "use_mmap") {
            return "mmap";
        }
        if (field == "repack") {
            return "rtr";
        }
        if (field == "mqkv") {
            return "mqkv";
        }
        if (field == "muge") {
            return "muge";
        }
        if (field == "defer_experts") {
            return "defer";
        }
        if (field == "sas") {
            return "sas";
        }
        if (field == "max_gpu") {
            return "max_gpu";
        }
        if (field == "use_thp") {
            return "thp";
        }
        if (field == "fused_moe") {
            return "fmoe";
        }
        if (field == "grouped_er") {
            return "ger";
        }
        if (field == "rcache") {
            return "rcache";
        }
        if (field == "no_fused_up_gate") {
            return "no-fug";
        }
        if (field == "no_ooae") {
            return "no-ooae";
        }
        if (field == "embeddings") {
            return "embd";
        }
        if (field == "tensor_split") {
            return "ts";
        }
        if (field == "cuda_params") {
            return "cuda";
        }
        if (field == "override_tensor") {
            return "ot";
        }
        if (field == "spec") {
            return "spec";
        }
        if (field == "n_draft") {
            return "nd";
        }
        if (field == "accept_rate") {
            return "acc%";
        }
        if (field == "mean_accept") {
            return "ma";
        }
        if (field == "target_ppl_of_output") {
            return "ppl_out";
        }
        if (field == "prompt_file") {
            return "prompt";
        }
        return field;
    }

    void print_header(const cmd_params & params) override {
        // select fields to print
        fields.emplace_back("model");
        fields.emplace_back("size");
        fields.emplace_back("params");
        fields.emplace_back("backend");
        bool is_cpu_backend = test::get_backend() == "CPU" || test::get_backend() == "BLAS";
        if (!is_cpu_backend) {
            fields.emplace_back("n_gpu_layers");
        }
        if (params.n_threads.size() > 1 || params.n_threads != cmd_params_defaults.n_threads || is_cpu_backend) {
            fields.emplace_back("n_threads");
        }
        if (params.n_batch.size() > 1 || params.n_batch != cmd_params_defaults.n_batch) {
            fields.emplace_back("n_batch");
        }
        if (params.n_ubatch.size() > 1 || params.n_ubatch != cmd_params_defaults.n_ubatch) {
            fields.emplace_back("n_ubatch");
        }
        if (params.type_k.size() > 1 || params.type_k != cmd_params_defaults.type_k) {
            fields.emplace_back("type_k");
        }
        if (params.type_v.size() > 1 || params.type_v != cmd_params_defaults.type_v) {
            fields.emplace_back("type_v");
        }
        if (params.main_gpu.size() > 1 || params.main_gpu != cmd_params_defaults.main_gpu) {
            fields.emplace_back("main_gpu");
        }
        if (params.split_mode.size() > 1 || params.split_mode != cmd_params_defaults.split_mode) {
            fields.emplace_back("split_mode");
        }
        if (params.no_kv_offload.size() > 1 || params.no_kv_offload != cmd_params_defaults.no_kv_offload) {
            fields.emplace_back("no_kv_offload");
        }
        if (params.flash_attn.size() > 1 || params.flash_attn != cmd_params_defaults.flash_attn) {
            fields.emplace_back("flash_attn");
        }
        if (params.mla_attn.size() > 1 || params.mla_attn != cmd_params_defaults.mla_attn) {
            fields.emplace_back("mla_attn");
        }
        if (params.attn_max_batch.size() > 1 || params.attn_max_batch != cmd_params_defaults.attn_max_batch) {
            fields.emplace_back("attn_max_batch");
        }
        if (params.reuse.size() > 1 || params.reuse != cmd_params_defaults.reuse) {
            fields.emplace_back("reuse");
        }
        if (params.ser.size() > 1 || params.ser != cmd_params_defaults.ser) {
            fields.emplace_back("ser");
        }
        if (params.tensor_split.size() > 1 || params.tensor_split != cmd_params_defaults.tensor_split) {
            fields.emplace_back("tensor_split");
        }
        if (params.use_mmap.size() > 1 || params.use_mmap != cmd_params_defaults.use_mmap) {
            fields.emplace_back("use_mmap");
        }
        if (params.embeddings.size() > 1 || params.embeddings != cmd_params_defaults.embeddings) {
            fields.emplace_back("embeddings");
        }
        if (params.cuda_params != cmd_params_defaults.cuda_params) {
            fields.emplace_back("cuda_params");
        }
        if (!(params.buft_overrides == cmd_params_defaults.buft_overrides)) {
            if (params.print_overrides) {
                fields.emplace_back("override_tensor");
            } else {
                skipped_overrides = true;
            }
        }
        if (params.repack != cmd_params_defaults.repack) {
            fields.emplace_back("repack");
        }
        if (params.mqkv != cmd_params_defaults.mqkv) {
            fields.emplace_back("mqkv");
        }
        if (params.sas != cmd_params_defaults.sas) {
            fields.emplace_back("sas");
        }
        if (params.max_gpu != cmd_params_defaults.max_gpu) {
            fields.emplace_back("max_gpu");
        }
        if (params.muge != cmd_params_defaults.muge) {
            fields.emplace_back("muge");
        }
        if (params.defer_experts != cmd_params_defaults.defer_experts) {
            fields.emplace_back("defer_experts");
        }
        if (params.use_thp != cmd_params_defaults.use_thp) {
            fields.emplace_back("use_thp");
        }
        if (params.fmoe != cmd_params_defaults.fmoe) {
            fields.emplace_back("fused_moe");
        }
        if (params.ger != cmd_params_defaults.ger) {
            fields.emplace_back("grouped_er");
        }
        if (params.rcache != cmd_params_defaults.rcache) {
            fields.emplace_back("rcache");
        }
        if (params.no_fug != cmd_params_defaults.no_fug) {
            fields.emplace_back("no_fused_up_gate");
        }
        if (params.no_ooae != cmd_params_defaults.no_ooae) {
            fields.emplace_back("no_ooae");
        }
        // T8 spec columns: emit only when user opted in (any non-default value).
        const bool any_spec = params.spec_type.size() > 1 ||
            (params.spec_type.size() == 1 && params.spec_type.front() != COMMON_SPECULATIVE_TYPE_NONE);
        if (any_spec) {
            fields.emplace_back("spec");
            fields.emplace_back("n_draft");
            fields.emplace_back("accept_rate");
            fields.emplace_back("mean_accept");
        }
        const bool any_prompt = params.prompt_files.size() > 1 ||
            (params.prompt_files.size() == 1 && !params.prompt_files.front().empty());
        if (any_prompt) {
            fields.emplace_back("prompt_file");
        }
        if (params.ppl_of_output) {
            fields.emplace_back("target_ppl_of_output");
        }
        fields.emplace_back("test");
        fields.emplace_back("t/s");

        fprintf(fout, "|");
        for (const auto & field : fields) {
            fprintf(fout, " %*s |", get_field_width(field), get_field_display_name(field).c_str());
        }
        fprintf(fout, "\n");
        fprintf(fout, "|");
        for (const auto & field : fields) {
            int width = get_field_width(field);
            fprintf(fout, " %s%s |", std::string(std::abs(width) - 1, '-').c_str(), width > 0 ? ":" : "-");
        }
        fprintf(fout, "\n");
    }

    void print_test(const test & t) override {
        std::map<std::string, std::string> vmap = t.get_map();

        fprintf(fout, "|");
        for (const auto & field : fields) {
            if (skipped_overrides && field == "override_tensor") {
                continue;
            }
            std::string value;
            char buf[128];
            if (field == "prompt_file") {
                // Display basename only — full path crowds the row.
                const std::string & pf = t.prompt_file;
                auto slash = pf.find_last_of('/');
                value = (slash == std::string::npos) ? pf : pf.substr(slash + 1);
            } else if (field == "target_ppl_of_output") {
                value = t.has_ppl ? vmap.at(field) : std::string("-");
            } else if (field == "accept_rate") {
                char buf2[32];
                snprintf(buf2, sizeof(buf2), "%.3f", t.accept_rate());
                value = buf2;
            } else if (field == "mean_accept") {
                char buf2[32];
                snprintf(buf2, sizeof(buf2), "%.2f", t.mean_accept());
                value = buf2;
            } else if (field == "spec") {
                value = common_speculative_type_to_str(t.spec_type);
            } else if (field == "model") {
                value = t.model_type;
            } else if (field == "size") {
                if (t.model_size < 1024*1024*1024) {
                    snprintf(buf, sizeof(buf), "%.2f MiB", t.model_size / 1024.0 / 1024.0);
                } else {
                    snprintf(buf, sizeof(buf), "%.2f GiB", t.model_size / 1024.0 / 1024.0 / 1024.0);
                }
                value = buf;
            } else if (field == "params") {
                if (t.model_n_params < 1000*1000*1000) {
                    snprintf(buf, sizeof(buf), "%.2f M", t.model_n_params / 1e6);
                } else {
                    snprintf(buf, sizeof(buf), "%.2f B", t.model_n_params / 1e9);
                }
                value = buf;
            } else if (field == "backend") {
                value = test::get_backend();
                if (t.has_rpc) {
                    value += "+RPC";
                }
            } else if (field == "test") {
                //if (t.n_prompt > 0 && t.n_gen == 0) {
                //    snprintf(buf, sizeof(buf), "pp%d", t.n_prompt);
                //} else if (t.n_gen > 0 && t.n_prompt == 0) {
                //    snprintf(buf, sizeof(buf), "tg%d", t.n_gen);
                //} else {
                //    snprintf(buf, sizeof(buf), "pp%d+tg%d", t.n_prompt, t.n_gen);
                //}
                //value = buf;
                value = t.test_label;
            } else if (field == "t/s") {
                snprintf(buf, sizeof(buf), "%.2f ± %.2f", t.avg_ts(), t.stdev_ts());
                value = buf;
            } else if (vmap.find(field) != vmap.end()) {
                value = vmap.at(field);
            } else {
                assert(false);
                exit(1);
            }

            int width = get_field_width(field);
            if (field == "t/s") {
                // HACK: the utf-8 character is 2 bytes
                width += 1;
            }
            fprintf(fout, " %*s |", width, value.c_str());
        }
        fprintf(fout, "\n");
    }

    void print_footer() override {
        fprintf(fout, "\nbuild: %s (%d)\n", test::build_commit.c_str(), test::build_number);
    }
};

struct sql_printer : public printer {
    static std::string escape_sql(const std::string & value) {
        std::string escaped;
        for (auto c : value) {
            if (c == '\'') {
                escaped += "''";
            } else {
                escaped += c;
            }
        }
        return escaped;
    }
    static std::string get_sql_field_type(const std::string & field) {
        switch (test::get_field_type(field)) {
            case test::STRING:
                return "TEXT";
            case test::BOOL:
            case test::INT:
                return "INTEGER";
            case test::FLOAT:
                return "REAL";
            default:
                assert(false);
                exit(1);
        }
    }

    void print_header(const cmd_params & params) override {
        std::vector<std::string> fields = test::get_fields();
        fprintf(fout, "CREATE TABLE IF NOT EXISTS test (\n");
        for (size_t i = 0; i < fields.size(); i++) {
            fprintf(fout, "  %s %s%s\n", fields.at(i).c_str(), get_sql_field_type(fields.at(i)).c_str(),  i < fields.size() - 1 ? "," : "");
        }
        fprintf(fout, ");\n");
        fprintf(fout, "\n");
        (void) params;
    }

    void print_test(const test & t) override {
        fprintf(fout, "INSERT INTO test (%s) ", join(test::get_fields(), ", ").c_str());
        fprintf(fout, "VALUES (");
        std::vector<std::string> values = t.get_values();
        std::transform(values.begin(), values.end(), values.begin(), escape_sql);
        for (size_t i = 0; i < values.size(); i++) {
            fprintf(fout, "'%s'%s", values.at(i).c_str(), i < values.size() - 1 ? ", " : "");
        }
        fprintf(fout, ");\n");
    }
};

// Greedy argmax over a single logit row (n_vocab floats).
static llama_token bench_greedy_argmax(const float * logits, int n_vocab) {
    llama_token best = 0;
    float bv = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    return best;
}

// Decode a real prompt as prefill (one batch per n_batch chunk; final logit only).
// Returns true on success.
static bool bench_prefill_real(llama_context * ctx, const std::vector<llama_token> & tokens,
                               int n_batch, int n_threads) {
    llama_set_n_threads(ctx, n_threads, n_threads);
    const int n_total = (int) tokens.size();
    int n_processed = 0;
    while (n_processed < n_total) {
        int n_tokens = std::min(n_batch, n_total - n_processed);
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (int i = 0; i < n_tokens; i++) {
            int pos = n_processed + i;
            bool logits_here = (pos == n_total - 1);
            common_batch_add(batch, tokens[pos], pos, {0}, logits_here);
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return false;
        }
        llama_batch_free(batch);
        n_processed += n_tokens;
    }
    llama_synchronize(ctx);
    return true;
}

// Spec-aware token generation. Drives `common_speculative_draft` + verify
// batch + accept-prefix loop, mirroring examples/dflash-speculative-simple
// and the production server's MTP path.
//
// On entry: ctx must already have the prompt prefilled; the last logit row
// holds the predictive distribution for the first generated token.
//
// Outputs: n_drafts (cycle count), n_accepted (sum of accepted prefix lengths),
// n_draft_total (sum of candidate lengths). When out_generated != nullptr,
// captures the emitted token sequence (for PPL-of-output).
static void test_gen_spec(
        llama_context * ctx,
        int n_gen,
        int n_threads,
        common_speculative * spec,
        common_params_speculative & spec_params,
        const std::vector<llama_token> & prompt_tokens,
        int & n_drafts,
        int & n_accepted,
        int & n_draft_total,
        std::vector<llama_token> * out_generated)
{
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model * model = llama_get_model(ctx);
    const int n_vocab = llama_n_vocab(model);
    const int n_prompt = (int) prompt_tokens.size();

    // First token: greedy argmax over the last logit row from prefill.
    llama_token id_last;
    {
        float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) return;
        id_last = bench_greedy_argmax(logits, n_vocab);
    }

    std::vector<llama_token> emitted;
    emitted.reserve(n_gen);
    emitted.push_back(id_last);

    const bool is_dflash = (spec_params.type == COMMON_SPECULATIVE_TYPE_DFLASH);
    std::vector<llama_token> prompt_tgt;

    while ((int) emitted.size() < n_gen) {
        prompt_tgt.assign(prompt_tokens.begin(), prompt_tokens.end());
        prompt_tgt.insert(prompt_tgt.end(), emitted.begin(), emitted.end() - 1);

        const llama_tokens draft = common_speculative_draft(spec, spec_params, prompt_tgt, id_last);
        n_drafts++;
        n_draft_total += (int) draft.size();

        if (draft.empty()) {
            // Fall back: single-token decode at the anchor position.
            llama_token tok = id_last;
            llama_decode(ctx, llama_batch_get_one(&tok, 1, (llama_pos) prompt_tgt.size(), 0));
            llama_synchronize(ctx);
            float * logits = llama_get_logits_ith(ctx, 0);
            if (!logits) break;
            llama_token nxt = bench_greedy_argmax(logits, n_vocab);
            emitted.push_back(nxt);
            id_last = nxt;
            if (llama_token_is_eog(model, nxt)) break;
            continue;
        }

        const llama_pos P = (llama_pos) prompt_tgt.size();

        if (is_dflash) {
            if (!llama_spec_ckpt_save(ctx, 0)) break;
        }

        const int verify_bs = (int) draft.size() + 1;
        llama_batch batch = llama_batch_init(verify_bs, 0, 1);
        common_batch_add(batch, id_last, P, {0}, true);
        for (size_t k = 0; k < draft.size(); k++) {
            common_batch_add(batch, draft[k], P + 1 + (llama_pos) k, {0}, true);
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            break;
        }
        std::vector<llama_token> sampled(verify_bs);
        for (int k = 0; k < verify_bs; k++) {
            float * logits = llama_get_logits_ith(ctx, k);
            if (!logits) { sampled[k] = -1; break; }
            sampled[k] = bench_greedy_argmax(logits, n_vocab);
        }
        llama_batch_free(batch);

        int n_acc = 0;
        for (size_t k = 0; k < draft.size(); k++) {
            if (draft[k] == sampled[k]) n_acc++;
            else break;
        }
        common_speculative_accept(spec, (uint16_t) n_acc);
        n_accepted += n_acc;

        llama_token bonus = sampled[n_acc];

        if (is_dflash) {
            if (!llama_spec_ckpt_restore(ctx, 0, P, n_acc)) break;
            llama_dflash_trim_extract(ctx, P + n_acc + 1, -1);
        } else {
            // Roll back rejected positions from target KV. The verify batch
            // wrote [P .. P+BS]; we keep [P .. P+n_acc] (id_last + n_acc drafts).
            llama_kv_cache_seq_rm(ctx, 0, P + n_acc + 1, -1);
        }

        for (int k = 0; k < n_acc; k++) emitted.push_back(draft[k]);
        emitted.push_back(bonus);
        id_last = bonus;

        if (llama_token_is_eog(model, bonus)) break;
    }

    if (out_generated) {
        *out_generated = emitted;
    }
    (void) n_prompt;
}

// Compute corpus PPL of [prompt + generated] under the target. Re-decodes
// the full sequence in fresh KV cache state, captures per-position logits at
// the positions that predict generated[0..n_gen-1], then runs the shared
// NLL/PPL kernel from common/perplexity.h.
//
// Returns exp(mean_nll); 0.0 on failure.
static double compute_ppl_of_output(
        llama_context * ctx,
        const std::vector<llama_token> & prompt_tokens,
        const std::vector<llama_token> & generated)
{
    const int n_prompt = (int) prompt_tokens.size();
    const int n_gen = (int) generated.size();
    if (n_prompt < 1 || n_gen < 1) return 0.0;

    const llama_model * model = llama_get_model(ctx);
    const int n_vocab = llama_n_vocab(model);

    llama_kv_cache_clear(ctx);

    std::vector<llama_token> all_tokens = prompt_tokens;
    all_tokens.insert(all_tokens.end(), generated.begin(), generated.end());
    const int n_total = (int) all_tokens.size();

    // Flatten per-position logits into [n_gen, n_vocab].
    std::vector<float> logits;
    logits.reserve((size_t) n_gen * (size_t) n_vocab);

    const int n_batch_cap = (int) llama_n_batch(ctx);
    int n_processed = 0;
    while (n_processed < n_total) {
        int n_tokens = std::min(n_batch_cap, n_total - n_processed);
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (int i = 0; i < n_tokens; i++) {
            int pos = n_processed + i;
            // Need predictive logits at [n_prompt-1 .. n_prompt+n_gen-2].
            bool need = (pos >= n_prompt - 1 && pos < n_prompt + n_gen - 1);
            common_batch_add(batch, all_tokens[pos], pos, {0}, need);
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            return 0.0;
        }
        // Gather logits in batch order. With logits-mask only on `need`
        // positions, llama_get_logits_ith(i) returns row for the i-th
        // `need` slot in the batch.
        int need_slot = 0;
        for (int i = 0; i < n_tokens; i++) {
            int pos = n_processed + i;
            if (pos >= n_prompt - 1 && pos < n_prompt + n_gen - 1) {
                float * row = llama_get_logits_ith(ctx, need_slot++);
                if (!row) {
                    llama_batch_free(batch);
                    return 0.0;
                }
                logits.insert(logits.end(), row, row + n_vocab);
            }
        }
        llama_batch_free(batch);
        n_processed += n_tokens;
    }

    if ((int) logits.size() != n_gen * n_vocab) return 0.0;

    // process_logits scores tokens[1..n_token] under logits[0..n_token-1].
    // tokens[0] is unused by the loop; place the prompt's last token there
    // for clarity. tokens[1..n_gen] = generated[0..n_gen-1].
    std::vector<llama_token> seq(n_gen + 1);
    seq[0] = prompt_tokens.back();
    for (int i = 0; i < n_gen; i++) seq[i + 1] = generated[i];

    double nll = 0.0, nll2 = 0.0;
    std::vector<float> logit_hist(n_gen), prob_hist(n_gen);

    int n_workers = std::max(1, (int) std::thread::hardware_concurrency() - 1);
    std::vector<std::thread> workers(n_workers);
    process_logits(n_vocab, logits.data(), seq.data(), n_gen, workers, nll, nll2,
                   logit_hist.data(), prob_hist.data());

    return std::exp(nll / (double) n_gen);
}

static void test_prompt(llama_context * ctx, int n_prompt, int n_past, int n_batch, int n_threads) {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model * model = llama_get_model(ctx);
    const int32_t n_vocab = llama_n_vocab(model);

    std::vector<llama_token> tokens(n_batch);

    int n_processed = 0;

    while (n_processed < n_prompt) {
        int n_tokens = std::min(n_prompt - n_processed, n_batch);
        tokens[0] = n_processed == 0 && llama_add_bos_token(model) ? llama_token_bos(model) : std::rand() % n_vocab;
        for (int i = 1; i < n_tokens; i++) {
            tokens[i] = std::rand() % n_vocab;
        }
        llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tokens, n_past + n_processed, 0));
        n_processed += n_tokens;
    }

    llama_synchronize(ctx);
}

static void test_gen(llama_context * ctx, int n_gen, int n_past, int n_threads) {
    llama_set_n_threads(ctx, n_threads, n_threads);

    const llama_model * model = llama_get_model(ctx);
    const int32_t n_vocab = llama_n_vocab(model);

    llama_token token = llama_add_bos_token(model) ? llama_token_bos(model) : std::rand() % n_vocab;

    for (int i = 0; i < n_gen; i++) {
        llama_decode(ctx, llama_batch_get_one(&token, 1, n_past + i, 0));
        llama_synchronize(ctx);
        token = std::rand() % n_vocab;
    }
}

static void llama_null_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

static std::unique_ptr<printer> create_printer(output_formats format) {
    switch (format) {
        case NONE:
            return nullptr;
        case CSV:
            return std::unique_ptr<printer>(new csv_printer());
        case JSON:
            return std::unique_ptr<printer>(new json_printer());
        case MARKDOWN:
            return std::unique_ptr<printer>(new markdown_printer());
        case SQL:
            return std::unique_ptr<printer>(new sql_printer());
    }
    GGML_ABORT("fatal error");
}

int main(int argc, char ** argv) {
    // try to set locale for unicode characters in markdown
    setlocale(LC_CTYPE, ".UTF-8");

#if !defined(NDEBUG)
    fprintf(stderr, "warning: asserts enabled, performance may be affected\n");
#endif

#if (defined(_MSC_VER) && defined(_DEBUG)) || (!defined(_MSC_VER) && !defined(__OPTIMIZE__))
    fprintf(stderr, "warning: debug build, performance may be affected\n");
#endif

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    fprintf(stderr, "warning: sanitizer enabled, performance may be affected\n");
#endif

    cmd_params params = parse_cmd_params(argc, argv);

    // initialize llama.cpp
    if (!params.verbose) {
        llama_log_set(llama_null_log_callback, NULL);
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize printer
    std::unique_ptr<printer> p = create_printer(params.output_format);
    std::unique_ptr<printer> p_err = create_printer(params.output_format_stderr);

    if (p) {
        p->fout = stdout;
        p->print_header(params);
    }

    if (p_err) {
        p_err->fout = stderr;
        p_err->print_header(params);
    }

    std::vector<cmd_params_instance> params_instances = get_cmd_params_instances(params);

    llama_model * lmodel = nullptr;
    const cmd_params_instance * prev_inst = nullptr;

    for (const auto & inst : params_instances) {
        // keep the same model between tests when possible
        if (!lmodel || !prev_inst || !inst.equal_mparams(*prev_inst)) {
            if (lmodel) {
                llama_free_model(lmodel);
            }

            lmodel = llama_model_load_from_file(inst.model.c_str(), inst.to_llama_mparams());
            if (lmodel == NULL) {
                fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, inst.model.c_str());
                return 1;
            }
            prev_inst = &inst;
        }

        llama_context * ctx = llama_init_from_model(lmodel, inst.to_llama_cparams());
        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, inst.model.c_str());
            llama_free_model(lmodel);
            return 1;
        }

        test t(inst, lmodel, ctx);

        // ----- T8 spec init (once per instance, after ctx created) -----
        common_speculative * spec = nullptr;
        common_params_speculative spec_params;
        std::vector<llama_token> prompt_tokens;
        bool spec_init_failed = false;

        if (!inst.prompt_file.empty()) {
            std::ifstream f(inst.prompt_file);
            if (!f) {
                fprintf(stderr, "%s: error: failed to open prompt file '%s'\n", __func__, inst.prompt_file.c_str());
                spec_init_failed = true;
            } else {
                std::stringstream ss;
                ss << f.rdbuf();
                prompt_tokens = common_tokenize(ctx, ss.str(), /*add_special=*/true, /*parse_special=*/true);
                if (prompt_tokens.empty()) {
                    fprintf(stderr, "%s: error: empty prompt after tokenization (%s)\n", __func__, inst.prompt_file.c_str());
                    spec_init_failed = true;
                }
            }
        }

        if (!spec_init_failed && inst.spec_type != COMMON_SPECULATIVE_TYPE_NONE) {
            spec_params.type = inst.spec_type;
            int nd = inst.n_draft;
            if (nd <= 0) {
                nd = (inst.spec_type == COMMON_SPECULATIVE_TYPE_DFLASH) ? 4 : 3;
            }
            spec_params.n_max = nd;

            if (inst.spec_type == COMMON_SPECULATIVE_TYPE_MTP) {
                if (llama_model_n_nextn_layer(lmodel) <= 0) {
                    fprintf(stderr, "warning: model has no MTP nextn layers; skipping --spec mtp row\n");
                    spec_init_failed = true;
                } else {
                    // Mirror server-context.cpp:294-331 MTP init recipe.
                    spec_params.cparams_dft = inst.to_llama_cparams();
                    spec_params.cparams_dft.mtp = true;
                    spec_params.cparams_dft.mtp_op_type = MTP_OP_WARMUP;
                    spec_params.cparams_dft.embeddings = true;
                    llama_set_embeddings(ctx, true);
                }
            } else if (inst.spec_type == COMMON_SPECULATIVE_TYPE_DFLASH) {
                if (inst.spec_model.empty()) {
                    fprintf(stderr, "error: --spec dflash requires --spec-model PATH\n");
                    spec_init_failed = true;
                } else {
                    spec_params.mparams_dft.path = inst.spec_model;
                }
            } else if (!inst.spec_model.empty()) {
                spec_params.mparams_dft.path = inst.spec_model;
            }

            if (!spec_init_failed) {
                spec = common_speculative_init(spec_params, ctx, /*seq_id=*/0);
                if (!spec) {
                    fprintf(stderr, "error: failed to init spec for type=%s\n",
                            common_speculative_type_to_str(inst.spec_type).c_str());
                    spec_init_failed = true;
                }
            }

            if (!spec_init_failed && inst.spec_type == COMMON_SPECULATIVE_TYPE_DFLASH) {
                const int max_tokens = (int) spec_params.n_max + 1;
                int rc = llama_spec_ckpt_init(ctx, LLAMA_SPEC_CKPT_AUTO, max_tokens);
                if (rc < 0) {
                    fprintf(stderr, "warning: llama_spec_ckpt_init returned %d\n", rc);
                }
            }
        }

        if (spec_init_failed) {
            if (spec) common_speculative_free(spec);
            llama_free(ctx);
            continue;
        }

        llama_kv_cache_clear(ctx);

        // warmup run
        if (params.warmup) {
            if (t.n_prompt > 0) {
                //test_prompt(ctx, std::min(t.n_batch, std::min(t.n_prompt, 32)), 0, t.n_batch, t.n_threads);
                test_prompt(ctx, 1, 0, t.n_batch, t.n_threads.second);
            }
            if (t.n_gen > 0) {
                test_gen(ctx, 1, 0, t.n_threads.first);
            }
        }

        // ----- Per-rep measurement -----
        const bool use_real_prompt = !prompt_tokens.empty();
        double ppl_sum = 0.0;
        int    ppl_n   = 0;

        for (int i = 0; i < params.reps; i++) {
            llama_kv_cache_clear(ctx);

            // Prefill: real prompt (if --prompt-file) or random (legacy).
            // PP/PG include prefill in the timer; TG/GP exclude it.
            const bool prefill_inside_timer =
                (t.test_kind == TEST_KIND_PP || t.test_kind == TEST_KIND_PG);

            uint64_t t_start = 0;
            if (prefill_inside_timer) {
                t_start = get_time_ns();
            }
            if (use_real_prompt) {
                if (!bench_prefill_real(ctx, prompt_tokens, t.n_batch, t.n_threads.second)) {
                    fprintf(stderr, "error: prefill decode failed\n");
                    break;
                }
            } else if (t.n_prompt > 0) {
                test_prompt(ctx, t.n_prompt, 0, t.n_batch, t.n_threads.second);
            }
            if (!prefill_inside_timer) {
                t_start = get_time_ns();
            }

            std::vector<llama_token> generated;
            if (t.n_gen > 0) {
                if (spec) {
                    int rep_n_drafts = 0;
                    int rep_n_accept = 0;
                    int rep_n_draft_total = 0;
                    test_gen_spec(ctx, t.n_gen, t.n_threads.first,
                                  spec, spec_params, prompt_tokens,
                                  rep_n_drafts, rep_n_accept, rep_n_draft_total,
                                  &generated);
                    t.n_drafts      += rep_n_drafts;
                    t.n_accepted    += rep_n_accept;
                    t.n_draft_total += rep_n_draft_total;
                } else {
                    const int n_past_after_prefill = use_real_prompt
                        ? (int) prompt_tokens.size()
                        : t.n_prompt;
                    test_gen(ctx, t.n_gen, n_past_after_prefill, t.n_threads.first);
                }
            }

            uint64_t t_ns = get_time_ns() - t_start;
            t.samples_ns.push_back(t_ns);

            // Optional second pass: corpus PPL of generated output under target.
            // Only meaningful when we have a real prompt + spec-driven coherent output.
            if (inst.ppl_of_output && spec && use_real_prompt && !generated.empty()) {
                double ppl = compute_ppl_of_output(ctx, prompt_tokens, generated);
                if (std::isfinite(ppl) && ppl > 0.0) {
                    ppl_sum += ppl;
                    ppl_n   += 1;
                }
            }
        }

        if (ppl_n > 0) {
            t.ppl_of_output = ppl_sum / (double) ppl_n;
            t.has_ppl = true;
        }

        if (p) {
            p->print_test(t);
            fflush(p->fout);
        }

        if (p_err) {
            p_err->print_test(t);
            fflush(p_err->fout);
        }

        llama_print_timings(ctx);

        if (spec) common_speculative_free(spec);
        llama_free(ctx);
    }

    llama_free_model(lmodel);

    if (p) {
        p->print_footer();
    }

    if (p_err) {
        p_err->print_footer();
    }

    llama_backend_free();

    return 0;
}
