// dflash-extract: dump per-layer residual-stream snapshots from a single
// forward pass through a Qwen 3.5/3.6 target model. Used to compare the
// ik_llama.cpp DFlash extract hook against a vLLM-side reference.
//
// Output: one .npy float32 file per requested layer, shape [n_tokens, n_embd],
// row-major little-endian.

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int32_t> parse_layer_list(const std::string & s) {
    std::vector<int32_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        out.push_back(std::atoi(tok.c_str()));
    }
    return out;
}

// Minimal float32 .npy v1.0 writer. Header dict + raw row-major data.
static bool write_npy_f32(const std::string & path,
                          const std::vector<float> & data,
                          int64_t n_tokens,
                          int64_t n_embd) {
    std::ostringstream dict;
    dict << "{'descr': '<f4', 'fortran_order': False, 'shape': ("
         << n_tokens << ", " << n_embd << "), }";
    std::string s = dict.str();
    // pad so that magic + version + header_len + dict + '\n' aligns to 64
    const size_t prelude = 10; // magic(6) + ver(2) + header_len(2)
    size_t total = prelude + s.size() + 1;
    size_t pad = (64 - (total % 64)) % 64;
    s.append(pad, ' ');
    s.push_back('\n');
    uint16_t header_len = (uint16_t) s.size();

    std::ofstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "dflash-extract: failed to open %s for writing\n", path.c_str());
        return false;
    }
    const char magic[] = "\x93NUMPY";
    f.write(magic, 6);
    const char ver[2] = { 0x01, 0x00 };
    f.write(ver, 2);
    f.write((const char *) &header_len, 2);
    f.write(s.data(), s.size());
    f.write((const char *) data.data(), data.size() * sizeof(float));
    return f.good();
}

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m MODEL --prompt-file FILE --extract-layers L1,L2,... "
        "--out-prefix PREFIX [--n-tokens N]\n"
        "  Loads MODEL, tokenises the contents of FILE, configures the DFlash\n"
        "  extract hook at the listed layer indices, decodes once, and dumps\n"
        "  each captured residual to PREFIX-layerN.npy (shape [n_tokens, n_embd]).\n"
        "  --n-tokens optionally caps the prompt to the first N tokens.\n",
        argv0);
}

// Parse our extra flags out of argv, leaving the remaining flags for
// gpt_params_parse so users can pass standard llama-server-style options
// (--ngl, --split-mode, --tensor-split, --device, etc.) through.
static std::vector<char *> strip_extract_flags(int argc, char ** argv,
                                               std::string & layer_arg,
                                               std::string & prompt_file,
                                               std::string & out_prefix,
                                               int & n_tokens_cap) {
    std::vector<char *> kept;
    kept.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--extract-layers" && i + 1 < argc)      { layer_arg = argv[++i]; }
        else if (a == "--prompt-file" && i + 1 < argc)    { prompt_file = argv[++i]; }
        else if (a == "--out-prefix" && i + 1 < argc)     { out_prefix = argv[++i]; }
        else if (a == "--n-tokens-cap" && i + 1 < argc)   { n_tokens_cap = std::atoi(argv[++i]); }
        else                                              { kept.push_back(argv[i]); }
    }
    return kept;
}

int main(int argc, char ** argv) {
    std::string layer_arg;
    std::string prompt_file;
    std::string out_prefix;
    int n_tokens_cap = -1;

    std::vector<char *> remaining = strip_extract_flags(argc, argv,
        layer_arg, prompt_file, out_prefix, n_tokens_cap);

    gpt_params params;
    params.n_ctx   = 4096;
    params.n_batch = 4096;
    params.n_ubatch = 4096;
    if (!gpt_params_parse((int) remaining.size(), remaining.data(), params)) {
        print_usage(argv[0]);
        return 2;
    }

    if (prompt_file.empty() || layer_arg.empty() || out_prefix.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    std::vector<int32_t> layers = parse_layer_list(layer_arg);
    if (layers.empty() || layers.size() > 16) {
        fprintf(stderr, "--extract-layers must list 1..16 indices\n");
        return 2;
    }

    std::ifstream pf(prompt_file);
    if (!pf) {
        fprintf(stderr, "could not open prompt file %s\n", prompt_file.c_str());
        return 1;
    }
    std::stringstream pbuf;
    pbuf << pf.rdbuf();
    std::string prompt = pbuf.str();

    llama_backend_init();

    llama_init_result init = llama_init_from_gpt_params(params);
    llama_model   * model = init.model;
    llama_context * ctx   = init.context;
    if (!model || !ctx) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }

    std::vector<int32_t> tokens = ::common_tokenize(ctx, prompt, true, true);
    if (n_tokens_cap > 0 && (int) tokens.size() > n_tokens_cap) {
        tokens.resize(n_tokens_cap);
    }
    fprintf(stderr, "dflash-extract: prompt tokenised to %zu tokens\n", tokens.size());

    // Configure the extract hook before decode.
    llama_set_dflash_extract_layers(ctx, layers.data(), (int32_t) layers.size());

    // Build a single-sequence batch over all prompt tokens. Request logits
    // on the last token so the standard decode path completes cleanly.
    llama_batch batch = llama_batch_init((int) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], (llama_pos) i, { 0 }, i + 1 == tokens.size());
    }

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        return 1;
    }

    const int n_embd = llama_model_n_embd(model);
    const int n_tok  = (int) tokens.size();

    int n_ok = 0;
    for (size_t k = 0; k < layers.size(); ++k) {
        std::vector<float> buf((size_t) n_tok * (size_t) n_embd, 0.0f);
        size_t got = llama_get_dflash_extract_data(ctx, (int32_t) k, buf.data(), buf.size());
        if (got == 0) {
            fprintf(stderr, "dflash-extract: layer %d returned 0 elements — hook miss\n", layers[k]);
            continue;
        }
        if (got != buf.size()) {
            fprintf(stderr, "dflash-extract: layer %d got %zu of %zu expected — truncating\n",
                    layers[k], got, buf.size());
        }
        char path[1024];
        std::snprintf(path, sizeof(path), "%s-layer%d.npy", out_prefix.c_str(), layers[k]);
        if (!write_npy_f32(path, buf, n_tok, n_embd)) {
            fprintf(stderr, "dflash-extract: failed to write %s\n", path);
            continue;
        }
        fprintf(stderr, "dflash-extract: wrote %s (shape [%d, %d])\n", path, n_tok, n_embd);
        ++n_ok;
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    return n_ok == (int) layers.size() ? 0 : 1;
}
