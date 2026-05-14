// fattn-per-slot-kv-sm75-stub.cpp
//
// Temporary stub for `fattn_per_slot_kv_sm75_launch` (the kernel's launcher).
// Returns -1 immediately. Used to keep the test binary linkable while the
// kernel implementation is in flight; once the real launcher exists in
// `ggml/src/ggml-cuda/fattn-per-slot-kv-sm75.cu`, this stub is REMOVED
// from the build (CMakeLists.txt switches sources).
//
// RED state per feedback_test_first_discipline: every test config will
// fail with the message "kernel launcher returned non-zero: -1" until
// the kernel lands.

#include "fattn-per-slot-kv-sm75-reference.h"

extern "C" int fattn_per_slot_kv_sm75_launch(
    const fattn_per_slot_kv_sm75::Half * /*Q*/,
    const fattn_per_slot_kv_sm75::Half * /*K*/,
    const fattn_per_slot_kv_sm75::Half * /*V*/,
    const fattn_per_slot_kv_sm75::Half * /*mask*/,
    const int32_t * /*slot_seq_lens*/,
    float * /*out_final*/,
    const fattn_per_slot_kv_sm75::AttnConfig & /*cfg*/
) {
    return -1;  // KERNEL NOT IMPLEMENTED — test should fail with this rc.
}
