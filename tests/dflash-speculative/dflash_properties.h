// dflash_properties.h
//
// Lightweight property-based testing helpers for DFlash spec
// invariants in ik_llama.cpp.
//
// No external library — uses std::mt19937 with a fixed seed
// (DFLASH_PROPERTY_SEED) so failures are deterministic and
// reproducible. Sample count is fixed at DFLASH_PROPERTY_SAMPLES.
//
// Pattern (modeled on tests/test-backend-ops.cpp's init_tensor_uniform):
//
//   int main() {
//       std::mt19937 rng(DFLASH_PROPERTY_SEED);
//       for (int sample = 0; sample < DFLASH_PROPERTY_SAMPLES; ++sample) {
//           int  n_layers = uniform_int(rng, 3, 8);
//           int  hidden   = uniform_int(rng, 1024, 8192);
//           // ... drive contract under test ...
//           DFLASH_REQUIRE(fc_weight_shape.rows == n_layers * hidden,
//                          "FuseProjectionFcWeight");
//       }
//       return 0;
//   }
//
// One macro per check: DFLASH_REQUIRE(cond, <Allium name>) where
// the second argument is the verbatim string-literal name of an
// Allium @invariant from dflash.allium (e.g. "FuseProjectionFcWeight",
// "AnchorPosPreserved"). scripts/check-bindings.py enforces that
// every cited name is a real Allium invariant — typos and renames
// fail the CI gate. On failure the macro prints the sample index,
// the cited invariant name, and the file:line.
//
// Coverage policy: every test in tests/dflash-speculative/ that
// uses random generation MUST cite a spec invariant in each
// DFLASH_REQUIRE. The point is binding to the spec, not numerical
// fuzzing. If you can't name the invariant the check binds, write
// it as a concrete unit test instead.
//
// Spec: specs/dflash/dflash.allium

#pragma once

#include <cstdio>
#include <cstdint>
#include <random>

namespace dflash_properties {

// Fixed across the suite; bump only when a known-passing test
// starts failing on a new shape (then the bumped seed becomes
// the regression record).
constexpr uint64_t DFLASH_PROPERTY_SEED    = 42ULL;
constexpr int      DFLASH_PROPERTY_SAMPLES = 128;

inline int uniform_int(std::mt19937 & rng, int lo, int hi) {
    std::uniform_int_distribution<int> d(lo, hi);
    return d(rng);
}

inline double uniform_real(std::mt19937 & rng, double lo, double hi) {
    std::uniform_real_distribution<double> d(lo, hi);
    return d(rng);
}

} // namespace dflash_properties

// Use inside main() loops. Bails with rc=1 on first failure;
// prints the sample index and the invariant name being bound.
#define DFLASH_REQUIRE(cond, invariant)                                                                              \
    do {                                                                                                             \
        if (!(cond)) {                                                                                               \
            std::fprintf(stderr,                                                                                     \
                         "FAIL: sample=%d invariant=%s cond=" #cond " at %s:%d\n",                                   \
                         sample, (invariant), __FILE__, __LINE__);                                                   \
            return 1;                                                                                                \
        }                                                                                                            \
    } while (0)
