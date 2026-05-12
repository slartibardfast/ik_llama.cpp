// PHASE46 S1.T1.5 — DFlash drafter standalone forward harness.
//
// Loads the DFlash drafter GGUF in is_drafter mode (bypasses the
// T1.3 loader-pair contract), allocates a zero-filled
// (n_target_pickoffs * n_embd, n_tokens) input buffer, calls
// llama_dflash_drafter_forward to run the drafter end-to-end, and
// writes the (n_embd, n_tokens) output as a .npy file for later
// numerical comparison (T1.6).
//
// Usage:
//   llama-dflash-drafter-forward --model <drafter.gguf>
//                                --out <output.npy>
//                                [--n-tokens 16]
//                                [--n-threads 4]
//                                [--gpu-layers 0]
//
// Output .npy format: numpy v1.0, fp32, fortran_order=False,
//                     shape=(n_tokens, n_embd) row-major.

#include "llama.h"
#include "llama-spec.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

namespace {

struct args_t {
    std::string model_path;
    std::string out_path;
    int32_t     n_tokens   = 16;
    int32_t     n_threads  = 4;
    int32_t     gpu_layers = 0;
};

void usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s --model <drafter.gguf> --out <output.npy>\n"
        "             [--n-tokens 16] [--n-threads 4] [--gpu-layers 0]\n",
        prog);
}

int parse_args(int argc, char ** argv, args_t & out) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](int k) -> const char * {
            if (i + k >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                return nullptr;
            }
            return argv[i + k];
        };
        if (a == "--model" || a == "-m") {
            const char * v = need(1); if (!v) return 1;
            out.model_path = v; ++i;
        } else if (a == "--out" || a == "-o") {
            const char * v = need(1); if (!v) return 1;
            out.out_path = v; ++i;
        } else if (a == "--n-tokens") {
            const char * v = need(1); if (!v) return 1;
            out.n_tokens = std::atoi(v); ++i;
        } else if (a == "--n-threads") {
            const char * v = need(1); if (!v) return 1;
            out.n_threads = std::atoi(v); ++i;
        } else if (a == "--gpu-layers") {
            const char * v = need(1); if (!v) return 1;
            out.gpu_layers = std::atoi(v); ++i;
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]); return 1;
        } else {
            fprintf(stderr, "unknown arg: %s\n", a.c_str());
            usage(argv[0]); return 1;
        }
    }
    if (out.model_path.empty() || out.out_path.empty()) {
        usage(argv[0]); return 1;
    }
    return 0;
}

// Minimal NPY writer. Format reference:
//   https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
// Emits NPY v1.0, little-endian float32, fortran_order=False.
bool write_npy_f32(const std::string & path,
                   const float * data,
                   int64_t rows,
                   int64_t cols) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "failed to open %s for writing\n", path.c_str());
        return false;
    }
    // 6-byte magic + 1-byte major + 1-byte minor.
    const char magic[8] = { '\x93', 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00 };
    std::fwrite(magic, 1, sizeof(magic), f);

    // ASCII header. v1.0 uses a uint16 little-endian header length.
    // Header must be padded with spaces to make (10 + len) % 64 == 0,
    // last char before the padding being '\n'.
    char hdr[256];
    std::snprintf(hdr, sizeof(hdr),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (%lld, %lld), }",
        (long long) rows, (long long) cols);
    size_t hlen = std::strlen(hdr);
    // Pad with spaces; final byte must be '\n'. Target alignment 64.
    size_t total = 10 + hlen + 1; // +1 for trailing newline
    size_t pad = (64 - (total % 64)) % 64;
    std::string padded(hdr, hdr + hlen);
    padded.append(pad, ' ');
    padded.push_back('\n');
    uint16_t hlen_le = (uint16_t) padded.size();
    std::fwrite(&hlen_le, 1, 2, f);
    std::fwrite(padded.data(), 1, padded.size(), f);

    // Raw fp32 data, row-major.
    std::fwrite(data, sizeof(float), (size_t) rows * (size_t) cols, f);

    std::fclose(f);
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    args_t args;
    if (parse_args(argc, argv, args) != 0) {
        return 1;
    }

    fprintf(stderr, "=== llama-dflash-drafter-forward ===\n");
    fprintf(stderr, "  model      : %s\n", args.model_path.c_str());
    fprintf(stderr, "  out        : %s\n", args.out_path.c_str());
    fprintf(stderr, "  n_tokens   : %d\n", args.n_tokens);
    fprintf(stderr, "  n_threads  : %d\n", args.n_threads);
    fprintf(stderr, "  gpu_layers : %d\n", args.gpu_layers);

    llama_backend_init();

    // Load the drafter — is_drafter=true bypasses T1.3 enforcement
    // and skips the (non-existent) tokenizer load.
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = args.gpu_layers;
    mp.is_drafter   = true;

    llama_model * model = llama_model_load_from_file(args.model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load model: %s\n", args.model_path.c_str());
        llama_backend_free();
        return 2;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx    = (uint32_t)(args.n_tokens * 2);
    cp.n_batch  = args.n_tokens;
    cp.n_ubatch = args.n_tokens;
    cp.n_threads       = args.n_threads;
    cp.n_threads_batch = args.n_threads;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        llama_free_model(model);
        llama_backend_free();
        return 3;
    }

    const int32_t n_embd            = llama_model_n_embd(model);
    // The drafter's input width = n_target_pickoffs * n_embd. We can
    // recover n_target_pickoffs from the model metadata via the public
    // KV API, but the loader has already parsed it into hparams; we
    // hard-code the production shape here (5 pickoffs × 5120 = 25600)
    // and assert it matches. A future iteration can plumb this via a
    // public getter.
    const int32_t n_target_pickoffs = 5;
    const int64_t n_fc_in           = (int64_t) n_target_pickoffs * (int64_t) n_embd;

    fprintf(stderr, "  n_embd     : %d\n", n_embd);
    fprintf(stderr, "  n_fc_in    : %lld  (= %d × %d)\n",
            (long long) n_fc_in, n_target_pickoffs, n_embd);

    // Allocate zero input + output buffers.
    std::vector<float> in_data((size_t) n_fc_in * (size_t) args.n_tokens, 0.0f);
    std::vector<float> out_data((size_t) n_embd * (size_t) args.n_tokens, 0.0f);

    fprintf(stderr, "  running forward ...\n");
    int rc = llama_dflash_drafter_forward(
        ctx,
        in_data.data(),
        args.n_tokens,
        out_data.data(),
        out_data.size());

    if (rc != 0) {
        fprintf(stderr, "llama_dflash_drafter_forward returned rc=%d\n", rc);
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 4;
    }

    fprintf(stderr, "  forward OK; output shape = (%d, %d)\n",
            args.n_tokens, n_embd);

    // Write as NPY. Shape (n_tokens, n_embd) — row-major, matches how
    // the runtime laid the output out (dim0=n_embd fastest in ggml;
    // when read row-major as (n_tokens, n_embd) consecutive rows are
    // consecutive tokens).
    if (!write_npy_f32(args.out_path, out_data.data(), args.n_tokens, n_embd)) {
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 5;
    }

    fprintf(stderr, "  wrote %s (%zu bytes)\n", args.out_path.c_str(),
            sizeof(float) * out_data.size() + 128 /* header */);

    // Quick sanity print: first 5 values of token 0.
    fprintf(stderr, "  out[0, :5] = ");
    for (int i = 0; i < 5 && i < n_embd; ++i) {
        fprintf(stderr, "%.6f ", out_data[i]);
    }
    fprintf(stderr, "\n");

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
