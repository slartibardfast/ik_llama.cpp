// test-paged-multi-seq-byte-identity.cpp
//
// Binding test extending test-multi-seq-decode-byte-identity (now in
// tree, commit ae5dee44) with paged-KV dispatch.
//
// Drives a 2-seq decode with identical prompts under paged allocation,
// asserts byte-identical output to two serial single-seq decodes at
// the same starting state.
//
// Binds:
//   paged_block_allocator.allium::BlockUniquelyOwned (mid-decode)
//   paged_read_path.allium::KLoopCanonicalOrder
//   paged_read_path.allium::BugCAbsenceByConstruction
//   paged_write_path.allium::ProductionByteIdentityAtNStream2
//
// RED-bound on HEAD: requires the full paged path (T5.5 + T5.6) to
// exercise multi-seq through the new layout.
//
// Body sketch (T5.6 author):
//   1. Load Qwen 3.6 27B Q4_0.
//   2. Configure n_seq_max=2; both seqs receive identical prompt.
//   3. Run 200-token decode via llama_decode with paged WRITE+READ.
//   4. Save per-seq logits.
//   5. Reset; run two serial single-seq decodes (n_seq_max=1) with the
//      same prompt.
//   6. Compare per-seq logits byte-by-byte; assert byte-identical.

// Includes deferred to T5.6:
//   #include "llama.h"
//   #include "common.h"
#include <cstdio>

#ifndef LLAMA_PAGED_KV_LANDED
#define LLAMA_PAGED_KV_LANDED 0
#endif

int main() {
    if (!LLAMA_PAGED_KV_LANDED) {
        printf("FAIL: T5.6 paged multi-seq byte-identity test not yet active.\n");
        printf("      Full paged path (T5.5 + T5.6) has not yet landed.\n");
        printf("      Test transitions PASS at T5.6 close.\n");
        return 1;
    }
    printf("FAIL: T5.6 implementation body not yet written even though flag set.\n");
    return 1;
}
