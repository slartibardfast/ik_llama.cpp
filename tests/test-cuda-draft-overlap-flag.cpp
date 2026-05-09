// PHASE46 E.1 — verify the LLAMA_DRAFT_OVERLAP env gate.
//
// Forks two children to dodge the once-cached static initializer:
//   * env unset → ggml_backend_cuda_draft_overlap_enabled() returns 0
//   * env set   → returns nonzero
//
// Default-off behavior is the load-bearing claim (Phase 38 E ground truth:
// async overlap measures negative on sm_75; the flag must be opt-in).

#include "ggml-cuda.h"

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#define CHECK(cond) do {                                         \
    if (!(cond)) {                                               \
        std::fprintf(stderr, "Assertion failed at %s:%d: %s\n",  \
                __FILE__, __LINE__, #cond);                      \
        std::abort();                                            \
    }                                                            \
} while (0)

static int wait_child(pid_t pid) {
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

int main() {
    // child A — env unset, must report disabled (exit 0)
    pid_t a = fork();
    if (a == 0) {
        unsetenv("LLAMA_DRAFT_OVERLAP");
        int v = ggml_backend_cuda_draft_overlap_enabled();
        std::_Exit(v != 0 ? 10 : 0);
    }
    int sa = wait_child(a);
    if (sa != 0) {
        std::fprintf(stderr, "child A: env-unset returned enabled=%d\n", sa);
        return 1;
    }

    // child B — env set, must report enabled (exit 1)
    pid_t b = fork();
    if (b == 0) {
        setenv("LLAMA_DRAFT_OVERLAP", "1", 1);
        int v = ggml_backend_cuda_draft_overlap_enabled();
        std::_Exit(v != 0 ? 1 : 11);
    }
    int sb = wait_child(b);
    if (sb != 1) {
        std::fprintf(stderr, "child B: env-set returned %d (expected 1)\n", sb);
        return 1;
    }

    // child C — env set to "0", must report disabled
    pid_t c = fork();
    if (c == 0) {
        setenv("LLAMA_DRAFT_OVERLAP", "0", 1);
        int v = ggml_backend_cuda_draft_overlap_enabled();
        std::_Exit(v != 0 ? 12 : 0);
    }
    int sc = wait_child(c);
    if (sc != 0) {
        std::fprintf(stderr, "child C: env=0 returned enabled=%d\n", sc);
        return 1;
    }

    std::printf("test-cuda-draft-overlap-flag: PASS\n");
    return 0;
}
