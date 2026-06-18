#llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE common)
    install(TARGETS ${TEST_TARGET} RUNTIME)
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_target_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS} get-model.cpp)

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    target_link_libraries(${TEST_TARGET} PRIVATE common)
    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS} get-model.cpp)

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
add_executable(test-tokenizer-0 test-tokenizer-0.cpp)
target_link_libraries(test-tokenizer-0 PRIVATE common)
install(TARGETS test-tokenizer-0 RUNTIME)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-starcoder.gguf)

if (LLAMA_LLGUIDANCE)
    llama_target_and_test(test-grammar-llguidance.cpp ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf)
endif ()

if (NOT WIN32)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API
    #llama_target_and_test(test-sampling.cpp)
    #llama_target_and_test(test-grammar-parser.cpp)
    #llama_target_and_test(test-grammar-integration.cpp)
    #llama_target_and_test(test-llama-grammar.cpp)
    #llama_target_and_test(test-chat.cpp)
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    #if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
    #    llama_target_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
    #    target_include_directories(test-json-schema-to-grammar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples/server)
    #endif()


    # build test-tokenizer-1-bpe target once and add many tests
    # add_executable(test-tokenizer-1-bpe test-tokenizer-1-bpe.cpp)
    # target_link_libraries(test-tokenizer-1-bpe PRIVATE common)
    # install(TARGETS test-tokenizer-1-bpe RUNTIME)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    # add_executable(test-tokenizer-1-spm test-tokenizer-1-spm.cpp)
    # target_link_libraries(test-tokenizer-1-spm PRIVATE common)
    # install(TARGETS test-tokenizer-1-spm RUNTIME)

    # llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-baichuan.gguf)

    # llama_target_and_test(test-double-float.cpp) # SLOW
endif()

#llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp )
llama_build_and_test(test-json-partial.cpp )
#llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)
llama_build_and_test(test-regex-partial.cpp)
llama_build_and_test(test-q4-0-ar16-cpu.cpp)
if (GGML_CUDA)
    llama_build_and_test(test-q4-0-ar16-cuda.cpp)
    llama_build_and_test(test-m1-gemv-byte-identity.cpp)
    llama_build_and_test(test-mmvq-cross-ncols.cpp)
endif()
llama_build_and_test(test-qnext-seq-pattern.cpp)
if (GGML_CUDA)
    llama_build_and_test(test-qnext-multislot-dispatch.cpp)
    llama_build_and_test(test-cuda-graph-cache-bounded.cpp)
    llama_build_and_test(test-cuda-graph-probe-hit-monotonic.cpp)
    llama_build_and_test(test-cuda-graph-probe-distinct-shapes.cpp)
    llama_build_and_test(test-cuda-graph-probe-destroy-frees.cpp)
    llama_build_and_test(test-cuda-graph-probe-schema.cpp)
    llama_build_and_test(test-cuda-graph-update-state-mutation.cpp)
    llama_build_and_test(test-cuda-graph-comparator-op-params.cpp)
    llama_build_and_test(test-cuda-pool-graceful-oom.cpp)
    # PHASE_CUDA_NATIVE_DISPATCH C0 — calibration framework binding test.
    # Covers idempotent calibrate, env overrides, no-ops-registered,
    # synthetic-probe registration, deterministic results, cache
    # round-trip, FORCE_RECALIBRATE bypass.
    llama_build_and_test(test-cuda-calibration-framework.cpp)

    # PHASE_CUDA_NATIVE_DISPATCH C1 — single-threaded compute_splits.
    # Binds HostThreadIsExactlyOne over the lifetime of N graph computes
    # on a multi-backend (2× CUDA) sched configured for split-mode graph
    # + has_reduce — the path that used to enter the openmp parallel
    # block. Asserts ggml_backend_sched_dispatch_thread_count() == 1
    # across 100 iterations.
    llama_build_and_test(test-single-threaded-dispatch.cpp)

    # PHASE_CUDA_NATIVE_DISPATCH C4 — outer cudaStreamBeginCapture wrap.
    # Drives ggml_cuda_outer_capture_begin/_end_and_launch directly on a
    # 2-CUDA-backend setup. Asserts:
    #   - outer_capture_count() advances by N after N round-trips
    #   - output is bit-identical across N captured replays
    # Requires 2 CUDA devices; skip otherwise.
    llama_build_and_test(test-multi-device-graph-capture.cpp)

    # PHASE_CUDA_NATIVE_DISPATCH C5 — split end-capture / launch-exec /
    # destroy-exec API used by the sched outer-graph cache. Captures
    # once, replays N times via _launch_exec, asserts bit-identical
    # output across all replays.
    llama_build_and_test(test-multi-device-graph-cache.cpp)

    # PHASE_CUDA_NATIVE_DISPATCH C7 — sched-integrated outer capture on
    # a libmgpu-shape graph (cross-device GGML_OP_REDUCE). Asserts:
    #   - outer_capture_count advances with the dispatch loop firing
    #     OR (if multi-backend branch didn't trigger) the sequential
    #     fallback produces deterministic output
    #   - output bit-identical across all iters
    #   - sched->outer_capture_disabled never goes true mid-test
    llama_build_and_test(test-libmgpu-subgraph-capture.cpp)

    # PHASE_CUDA_NATIVE_DISPATCH C8-C11 — bind that all four calibrated
    # ops (MATMUL_STREAM_SPLIT, REDUCE_CROSS_DEVICE, PEER_COPY, GRAPH_
    # CAPTURE) are registered at module load via the static-initializer
    # in op_calibration_probes.cu, calibrate to SIZE_MAX on the default-
    # wins stub probes, and produce deterministic thresholds across
    # fresh backends.
    llama_build_and_test(test-calibration-ops-registered.cpp)

    # mtp_fused_draft.allium — RED-first scaffolding. Each test fails
    # to compile or link until its referenced API symbols land.
    llama_build_and_test(mtp-fused/test-mtp-fused-symbols.cpp)
    llama_build_and_test(mtp-fused/test-mtp-fused-step-count-bound.cpp)
    llama_build_and_test(mtp-fused/test-mtp-fused-single-compute.cpp)
    llama_build_and_test(mtp-fused/test-mtp-fused-argmax-correctness.cpp)
    llama_build_and_test(mtp-fused/test-mtp-fused-determinism.cpp)
    llama_build_and_test(mtp-fused/test-mtp-fused-kv-coverage.cpp)
    llama_build_and_test(mtp-fused/test-mtp-fused-kv-chop-rewrite.cpp)
    llama_build_and_test(mtp-fused/test-mtp-fused-prob-populated.cpp)
    llama_build_and_test(mtp-fused/test-mtp-fused-split-argmax.cpp)
    llama_build_and_test(mtp-fused/test-mtp-fused-chain-residual.cpp)

    # mtp_ubatch_hook.allium — RED-first scaffolding for the per-ubatch
    # MTP-KV hook. The .sh integration tests live alongside but are
    # not registered in ctest; they're run manually like
    # tests/mtp-rollout-investigation/.
    llama_build_and_test(mtp-ubatch-hook/test-hook-tag-tensor.cpp)
    llama_build_and_test(mtp-ubatch-hook/test-hook-fires-once.cpp)
    llama_build_and_test(mtp-ubatch-hook/test-hook-no-secondary-decode.cpp)
    llama_build_and_test(mtp-ubatch-hook/test-hook-cross-ubatch-pairing.cpp)
    llama_build_and_test(mtp-ubatch-hook/test-hook-idempotent-chop.cpp)
    llama_build_and_test(mtp-ubatch-hook/test-hook-acceptance-vs-update-accepted.cpp)

    # mtp_verify_accept.allium — RED-first scaffolding for the verify-step
    # accept-decision contract that sits between the per-ubatch hook and
    # the post-accept KV chop. Host-side; no GPU needed for the algorithm
    # itself, gated alongside sister bundles for symmetry.
    # Disabled 2026-05-14: pre-existing symbol mismatch (LLAMA_ACCEPT_MODE_*
    # and llama_accept_decision no longer in the header). Unrelated to Phase 2.
    # llama_build_and_test(mtp-verify-accept/test-accept-symbols.cpp)
    llama_build_and_test(mtp-verify-accept/test-accept-logits-shape.cpp)
    llama_build_and_test(mtp-verify-accept/test-accept-longest-prefix.cpp)
    llama_build_and_test(mtp-verify-accept/test-accept-bonus-correctness.cpp)
    llama_build_and_test(mtp-verify-accept/test-accept-bonus-position.cpp)
    llama_build_and_test(mtp-verify-accept/test-accept-determinism.cpp)
    llama_build_and_test(mtp-verify-accept/test-accept-progress.cpp)
    llama_build_and_test(mtp-verify-accept/test-accept-probabilistic-rejected.cpp)
    llama_build_and_test(mtp-verify-accept/test-accept-committed-shape.cpp)

    # dflash_speculative.allium — RED-first scaffolding for DFlash
    # speculative decoding (z-lab/Qwen3.6-27B-DFlash drafter on
    # Qwen3.6-27B target). See yarn-agentic/dflash_speculative.allium
    # and PHASE_DFLASH_DESIGN.md. Symbols test binds at compile time;
    # behavioural tests skip (exit 77) until implementation lands.
    llama_build_and_test(dflash-speculative/test-dflash-symbols.cpp)
    llama_build_and_test(dflash-speculative/test-dflash-load-vocab-identity.cpp)
    # Smoke test for bench's spec init paths (none / mtp / dflash). Skips
    # (exit 77) when LLAMA_TEST_TARGET unset. See PHASE_DFLASH.md Phase 1
    # closure binding #7.
    llama_build_and_test(dflash-speculative/test-llama-bench-spec-init.cpp)
    # T9 (revised) — vanilla np>1 validity harness. Gates the DFlash
    # multi-slot API extension; if vanilla fails validity, DFlash multi-slot
    # work is moot. See PHASE_DFLASH.md T9.
    llama_build_and_test(dflash-speculative/test-np-validity-vanilla.cpp)
    # PLAN.md D1 — per-layer per-slot residual capture at np ∈ {1,2,4,8}.
    # Single-decode capture writes n_embd floats per (layer × slot) to disk.
    # Drives D2 first-divergent-layer localization. Requires extract-layers
    # cap ≥ n_layer (cap is 80; Qwen 3.6 27B is 65 — fits).
    llama_build_and_test(dflash-speculative/test-deltanet-d1-capture.cpp)
    # PLAN.md S2.3 — op-level intermediate capture at a specified target
    # layer. Installs a custom cb_eval via llama_context_params.cb_eval
    # that captures every named tensor whose suffix matches "-<L>" for L
    # in the configured target layer set. Localizes WHICH op within a
    # first-divergent layer (S2.1 named the layer; S2.3 names the op).
    llama_build_and_test(dflash-speculative/test-deltanet-s23-op-capture.cpp)

    # PLAN.md S2.5.b — RED → GREEN unit test for fattn_per_slot_kv_sm75
    # (specs/deltanet/fattn-per-slot-kv-sm75.md).
    #
    # The launcher `fattn_per_slot_kv_sm75_launch` is implemented in
    # ggml/src/ggml-cuda/fattn-per-slot-kv-sm75.cu (Phase 1 = naïve scalar
    # device kernel; Phase 2 = SoTA PTX path). The test links against ggml
    # to pull in the launcher symbol from the compiled ggml-cuda library.
    set(TEST_TARGET test-fattn-per-slot-kv-sm75)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-fattn-per-slot-kv-sm75.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # PHASE_FATTN_KV_SPLIT_SM75 step 3 — KV-split decode kernel unit tests:
    # f64 reference (q4_0 + f16), rep-determinism, padding/chunk-count
    # invariance at straddle lengths, block-table permutation invariance,
    # and batch invariance. The launcher
    # `fattn_per_slot_kv_split_sm75_test_launch` is implemented in
    # ggml/src/ggml-cuda/fattn-per-slot-kv-split-sm75.cu.
    set(TEST_TARGET test-fattn-per-slot-kv-split-sm75)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-fattn-per-slot-kv-split-sm75.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # NP-invariance unit test on the production dispatch path
    # (specs/deltanet/fattn-per-slot-kv-sm75.md §15.7). Drives the
    # ggml_flash_attn_ext_per_slot_kv op through the CUDA backend with
    # two K-cache occupancies and asserts byte-identity. Separate from
    # test-fattn-per-slot-kv-sm75 which validates the deprecated bespoke
    # kernels against a host scalar reference.
    set(TEST_TARGET test-fattn-per-slot-kv-dispatch-np-invariance)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-fattn-per-slot-kv-dispatch-np-invariance.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # Phase CX.A — ne[1] sweep test for the wmma fp16 frag_c_VKQ claim.
    # The companion dispatch-np-invariance test only varies K-cache stride
    # at ne[1]=1; this varies ne[1] ∈ {1, 2, 4, 8} with row-0 input
    # invariant, exposing inter-row WMMA fp16 rounding if present.
    set(TEST_TARGET test-fattn-per-slot-kv-ncols-invariance)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-fattn-per-slot-kv-ncols-invariance.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # Phase CX.B — RMSNorm batch-shape invariance.
    set(TEST_TARGET test-rmsnorm-batch-shape-invariance)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-rmsnorm-batch-shape-invariance.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # Phase CX.C — RoPE batch-shape invariance.
    set(TEST_TARGET test-rope-batch-shape-invariance)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-rope-batch-shape-invariance.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # DeltaNet (linear-attention) shape invariance at production geometry.
    set(TEST_TARGET test-deltanet-shape-invariance)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-deltanet-shape-invariance.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # K1 — DeltaNet save_all_steps self-consistency at production geometry.
    # Binds the kernel-side cause narrowed by the P0.A.3 bisect ladder: the
    # last per-step state (save_all_steps=true) must byte-equal the final
    # state (save_all_steps=false) for the same inputs and n_tokens.
    set(TEST_TARGET test-deltanet-save-all-steps-last-state)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-deltanet-save-all-steps-last-state.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # L1 — libllama-level observational test on save_per_step_ssm. Toggling
    # llama_spec_ckpt_init PER_STEP + save before a verify-style 5-row
    # decode must not change the per-row argmax sequence. Mirrors the
    # P0.A.3 bisect ladder run E result.
    set(TEST_TARGET test-dflash-save-per-step-ssm-observational)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-dflash-save-per-step-ssm-observational.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # P0.A.3 — Observational equivalence of residual extraction. With the
    # cb_eval hook armed, decoding the same prompt twice (extract disarmed
    # vs armed) must produce byte-identical argmax sequences. Binds the
    # ResidualExtractObservationallyEquivalent invariant (Allium +
    # CbEvalObservational.tla).
    set(TEST_TARGET test-dflash-extract-observational)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-dflash-extract-observational.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # PHASE_NSTREAM_KV_PERF T3.3 + T3.4 — unified-stream dispatch byte
    # identity. Multi-seq batch (two seqs, identical prompts) via one
    # llama_decode must match per-seq logits of two serial single-seq
    # decodes at the same starting state.
    set(TEST_TARGET test-multi-seq-decode-byte-identity)
    add_executable(${TEST_TARGET}
        spec/test-multi-seq-decode-byte-identity.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # K1' — DeltaNet save_all_steps intermediate-state equivalence. For
    # every k ∈ {0..4}, per_step_ssm[k] from an n=5 save_all run must
    # byte-equal final state from a fresh n=k+1 save=false run. Binds
    # the source data per_step_restore reads.
    set(TEST_TARGET test-deltanet-save-all-steps-intermediate)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-deltanet-save-all-steps-intermediate.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # L2 — libllama per_step_restore byte-identity. A fresh 3-token
    # decode + 1-token bonus must produce byte-identical logits to a
    # 5-token verify-batch decode under save_per_step_ssm + restore(2)
    # + 1-token bonus. Binds Suspect 4 (per_step_restore semantics).
    set(TEST_TARGET test-dflash-per-step-restore-byte-identity)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-dflash-per-step-restore-byte-identity.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # L3 — verify-batch vs autoregressive batch-shape invariance probe.
    # Single 5-token verify decode (no spec_ckpt, no DFlash pipeline)
    # compared row-by-row to spec-none 1-token autoregressive at the
    # same effective context. Test stays FAIL on HEAD as a regression
    # gate until the variance is fixed.
    set(TEST_TARGET test-dflash-multi-cycle-restore-drift)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-dflash-multi-cycle-restore-drift.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # L3' — verify-batch width sweep. For each verify_bs ∈ {1..8},
    # compare per-row argmax of one batch to spec-none autoregressive.
    # Localises the smallest n_tokens at which batch-shape variance
    # first appears.
    set(TEST_TARGET test-dflash-verify-batch-width-sweep)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-dflash-verify-batch-width-sweep.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # L4 — per-layer batch-shape divergence localiser. Capture
    # l_out-<il> for every Nth layer across two paths (path A = two
    # sequential 1-token decodes; path B = one 2-token batch).
    # First divergent layer points at which transformer block
    # introduces batch-shape variance.
    set(TEST_TARGET test-dflash-per-layer-batch-shape-diff)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-dflash-per-layer-batch-shape-diff.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # L5 — kernel-direct mul_mat(Q4_0 weight, F32 input) batch-shape
    # invariance test. Compares y_n1 row 0 to y_n2 col 1 with same
    # input vector. Tests whether MMQ kernel introduces batch-shape
    # variance.
    set(TEST_TARGET test-mulmat-batch-shape-invariance)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-mulmat-batch-shape-invariance.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # Cross-mmq_x dispatch byte-invariance — asserts that mul_mat output
    # for the SAME input column is byte-identical across every ne11 in
    # the dispatcher's tile sweep. Catches reduction-order divergence
    # between split-K (mmq_x ≤ 16) and non-split-K (mmq_x > 16) paths.
    set(TEST_TARGET test-mulmat-mmq_x-dispatch-invariance)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-mulmat-mmq_x-dispatch-invariance.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # TRACE-2 — intra-layer-3 named captures around full-attn block.
    set(TEST_TARGET test-trace-2-intra-layer-capture)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-trace-2-intra-layer-capture.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # Phase CY.B.1 — intra-layer-0 DeltaNet named captures to find the first
    # divergent tag between NP=1 (all_same_seq=true) and NP=4
    # (all_same_seq=false) per llama-delta-net.cpp:679/711-727.
    set(TEST_TARGET test-cy-layer0-capture)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-cy-layer0-capture.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # Phase CY.F.1 — production-dim MMQ shape-invariance: K=27648, N=5120.
    # Mirrors test-mmq-q4-0-ar16-shape-invariance but at Qwen 3.6 27B ffn dims.
    set(TEST_TARGET test-mmq-q4-0-ar16-shape-invariance-prod-dim)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-mmq-q4-0-ar16-shape-invariance-prod-dim.cpp)
    target_include_directories(${TEST_TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/ggml/src
        ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # Phase CY.F.6 — ggml_reduce(OP_ADD, n) shape-invariance unit test.
    # Drives the cross-device sum op at M ∈ {1,2,4,8} with same col-0 inputs.
    set(TEST_TARGET test-ggml-reduce-shape-invariance)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-ggml-reduce-shape-invariance.cpp)
    target_include_directories(${TEST_TARGET} PRIVATE
        ${CMAKE_SOURCE_DIR}/ggml/src
        ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # Phase CY.F.9 — umbrella seq_id sequential prefill test. Drives
    # llama_decode at seq_id ∈ {0..N-1} sequentially on a single context
    # with n_seq_max=N. Compares post-prefill logits bit-wise. Exposes any
    # slot-id > 0 addressing bug in the serial path.
    set(TEST_TARGET test-cy-seq-id-sequential)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-cy-seq-id-sequential.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # Phase CY.F.9b — batched multi-seq prefill vs serial baseline. Same
    # context, all N seqs batched into ONE llama_decode call. Per-seq
    # last-token logits are compared to the serial baseline. If differs,
    # the bug is in batched multi-seq graph build.
    set(TEST_TARGET test-cy-seq-id-batched)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-cy-seq-id-batched.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # Phase CY.F.17 — NP=2 multi-step decode race repro. CY.F.9b only checks
    # post-prefill logits and always PASSES. The production NP=2 failure
    # surfaces during 32-step decode. This test reproduces the multi-step
    # decode pattern WITHOUT the HTTP server.
    set(TEST_TARGET test-cy-np2-multi-step-decode)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-cy-np2-multi-step-decode.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # Phase CY.F.18 — layer-bisect probe. Captures per-layer residual at
    # decode step 1 across two NP=2 runs, finds first divergent layer for
    # slot 1 (race source).
    set(TEST_TARGET test-cy-f18-layer-bisect)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-cy-f18-layer-bisect.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # Phase CY.F.14 — find first-divergent layer between serial seq=0 and
    # batched seq=0 (multi-seq batch). Uses dflash extract-layers (authoritative
    # capture with dtype branching) to pin down which layer introduces the
    # batched-vs-serial drift.
    set(TEST_TARGET test-cy-batched-layer-first-div)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-cy-batched-layer-first-div.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # Phase CY.F.15 — drill into layer 0 to find first divergent intra-layer
    # tag (DeltaNet inputs/outputs, qkv projections, FFN) between serial seq=0
    # and batched seq=0 row. Uses cb_eval matched-name capture with F16→F32
    # dtype branching (matches libllama mechanism).
    set(TEST_TARGET test-cy-layer0-serial-vs-batched)
    add_executable(${TEST_TARGET}
        dflash-speculative/test-cy-layer0-serial-vs-batched.cpp)
    target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
    set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

    # TRACE-4 / 5 / 7 — focused CUDA probe for warp_reduce_sum mask-shape
    # non-associativity, warp_reduce_max invariance, and compact-then-reduce
    # fix candidate. CUDA target; sm_75 required.
    if (GGML_CUDA)
        set(TEST_TARGET test-trace-4-warp-reduce-mask-shape)
        add_executable(${TEST_TARGET}
            dflash-speculative/test-trace-4-warp-reduce-mask-shape.cu)
        set_target_properties(${TEST_TARGET} PROPERTIES CUDA_ARCHITECTURES "75")
        target_link_libraries(${TEST_TARGET} PRIVATE ${CMAKE_THREAD_LIBS_INIT})
        install(TARGETS ${TEST_TARGET} RUNTIME)
        add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
        set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")
    endif()
    llama_build_and_test(dflash-speculative/test-dflash-load-dense-qwen3.cpp)
    llama_build_and_test(dflash-speculative/test-dflash-load-source-layer-count.cpp)
    llama_build_and_test(dflash-speculative/test-dflash-load-block-size.cpp)
    # Property-based test: binds spec invariants over randomly
    # generated inputs. See dflash_properties.h for the framework.
    llama_build_and_test(dflash-speculative/test-dflash-fc-projection-shape.cpp)

    # DFlash CUDA kernel unit tests — built only when the sm_75 DFlash
    # kernel set is enabled. Each test exercises a kernel's fused output
    # against a scalar fp32 reference. SKIPs (exit 77) until the
    # corresponding kernel body lands.
    if (GGML_CUDA_DFLASH)
        # CUDAToolkit's imported targets created in ggml/src are scoped
        # to that sibling subdirectory; re-find here so CUDA::cudart and
        # CUDAToolkit_INCLUDE_DIRS are visible in tests/.
        find_package(CUDAToolkit REQUIRED)

        llama_build_and_test(dflash-speculative/test-dflash-combine-features.cpp)
        target_include_directories(test-dflash-combine-features PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(test-dflash-combine-features PRIVATE CUDA::cudart)

        llama_build_and_test(dflash-speculative/test-dflash-inject-fused.cpp)
        target_include_directories(test-dflash-inject-fused PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(test-dflash-inject-fused PRIVATE CUDA::cudart)

        # WMMA-mimicking scalar oracle self-test — validates the oracle
        # itself before any kernel binds against it. CPU-only test; needs
        # CUDA fp16 header for __half and conversion intrinsics.
        llama_build_and_test(dflash-speculative/test-wmma-mimicking-oracle.cpp)
        target_include_directories(test-wmma-mimicking-oracle PRIVATE
            ${CUDAToolkit_INCLUDE_DIRS})

        # Disabled 2026-05-14: pre-existing signature mismatch with
        # dflash_reference::drafter_forward_reference. Unrelated to Phase 2.
        # llama_build_and_test(dflash-speculative/test-dflash-drafter-forward.cpp)
        # target_include_directories(test-dflash-drafter-forward PRIVATE
        #     ${CMAKE_SOURCE_DIR}/ggml/src
        #     ${CUDAToolkit_INCLUDE_DIRS})
        # target_link_libraries(test-dflash-drafter-forward PRIVATE CUDA::cudart)

        llama_build_and_test(dflash-speculative/test-dflash-drafter-lm-head.cpp)
        target_include_directories(test-dflash-drafter-lm-head PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(test-dflash-drafter-lm-head PRIVATE CUDA::cudart)

        llama_build_and_test(dflash-speculative/test-dflash-argmax-match.cpp)
        target_include_directories(test-dflash-argmax-match PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(test-dflash-argmax-match PRIVATE CUDA::cudart)

        llama_build_and_test(dflash-speculative/test-dflash-drafter-load.cpp)
        target_include_directories(test-dflash-drafter-load PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CMAKE_SOURCE_DIR}/ggml/include
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(test-dflash-drafter-load PRIVATE CUDA::cudart ggml)

        # Disabled 2026-05-14: pre-existing signature mismatch. Unrelated to Phase 2.
        # llama_build_and_test(dflash-speculative/test-dflash-end-to-end.cpp)
        # target_include_directories(test-dflash-end-to-end PRIVATE
        #     ${CMAKE_SOURCE_DIR}/ggml/src
        #     ${CMAKE_SOURCE_DIR}/ggml/include
        #     ${CUDAToolkit_INCLUDE_DIRS})
        # target_link_libraries(test-dflash-end-to-end PRIVATE CUDA::cudart ggml)

        llama_build_and_test(dflash-speculative/test-dflash-closure.cpp)
        target_include_directories(test-dflash-closure PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CMAKE_SOURCE_DIR}/ggml/include
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(test-dflash-closure PRIVATE CUDA::cudart ggml)

        # DFlash multi-slot Phase 3: cb_eval per-seq demux gate. Decodes
        # a 2-seq batch through the qwen35 graph; verifies each seq_id's
        # extract buffer count and that seq_id=1's buffer is populated.
        llama_build_and_test(dflash-speculative/test-dflash-extract-multi-seq.cpp)

        # DFlash multi-slot Phase 4: byte-identical-binding gate.
        # Verifies that draft_batch(n_slots=2, [0,1]) produces the same
        # 8 candidates as two serial draft_batch(n_slots=1) calls (one
        # per seq_id). Pins the multi-slot dispatch semantic.
        llama_build_and_test(dflash-speculative/test-dflash-batch-vs-serial.cpp)

        # DFlash multi-slot Phase 6: full-pipeline np-invariance + throughput
        # measurement of record. Drives llama_dflash_draft_batch at
        # NP ∈ {1,2,4,8} with identical per-seq content; asserts slot-0
        # byte-identity across NP (orchestrator-level np-invariance
        # complementing T7's kernel-level probe) and reports per-NP
        # aggregate t/s.
        llama_build_and_test(dflash-speculative/test-dflash-np-multislot.cpp)

        # DFlash multi-slot Phase 5: orchestrator-level fan-out gate.
        # Verifies that common_speculative_draft_batched({spec0, spec1})
        # over a shared ctx (share-the-bind on the drafter) produces
        # candidates byte-identical to two serial common_speculative_draft
        # calls, AND that an asymmetric per-seq prefill yields per-slot
        # candidates that differ (seq_id routes through the all-DFlash
        # gate). Pins the server-orchestrator adapter contract.
        llama_build_and_test(dflash-speculative/test-dflash-spec-batched-fanout.cpp)

        # Empirical np-invariance probe — drafter_forward at N_slots ∈
        # {1,2,4,8} with identical slot content; slot 0 output must be
        # byte-identical across all N values.
        llama_build_and_test(dflash-speculative/test-dflash-np-invariance.cpp)
        target_include_directories(test-dflash-np-invariance PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(test-dflash-np-invariance PRIVATE CUDA::cudart)

        # T6: spec-ckpt flow witness for the 3 DeltaNet-rollback invariants
        # (DraftKVRollbackOnRejection, InjectedKVEvictedOnAnchorAdvance,
        # EffectiveSeqLensSubtractsRejected). Drives llama_spec_ckpt_*
        # save → verify → restore.
        llama_build_and_test(dflash-speculative/test-dflash-spec-ckpt-flow.cpp)

        # PHASE_MMQ_Q4_0_AR16 A.3: load_tiles_q4_0_ar16 SMEM dump vs CPU
        # dequant oracle. Single CUDA source (.cu), self-contained main().
        # Needs CUDA language enabled at tests/ scope.
        enable_language(CUDA)
        set(TEST_TARGET test-mmq-q4-0-ar16-load-tiles)
        add_executable(${TEST_TARGET}
            dflash-speculative/test-mmq-q4-0-ar16-load-tiles.cu)
        target_include_directories(${TEST_TARGET} PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CMAKE_SOURCE_DIR}/ggml/include
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(${TEST_TARGET} PRIVATE CUDA::cudart ggml)
        target_compile_definitions(${TEST_TARGET} PRIVATE
            GGML_USE_CUDA
            GGML_CUDA_FUSION=1
            GGML_CUDA_MIN_BATCH_OFFLOAD=32
            GGML_CUDA_DMMV_X=32
            GGML_CUDA_MMV_Y=1
            GGML_CUDA_PEER_MAX_BATCH_SIZE=128
            K_QUANTS_PER_ITERATION=2)
        set_target_properties(${TEST_TARGET} PROPERTIES CUDA_ARCHITECTURES 75)
        install(TARGETS ${TEST_TARGET} RUNTIME)
        add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
        set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

        # PHASE_MMQ_Q4_0_AR16 A.4: vec_dot_q4_0_ar16_q8_1_dp4a × scalar fp32
        # reference. Single CUDA source, self-contained main(). Closure
        # binding: cos ≥ 0.9999, NMSE ≤ 1e-4.
        set(TEST_TARGET test-mmq-q4-0-ar16-dp4a)
        add_executable(${TEST_TARGET}
            dflash-speculative/test-mmq-q4-0-ar16-dp4a.cu)
        target_include_directories(${TEST_TARGET} PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CMAKE_SOURCE_DIR}/ggml/include
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(${TEST_TARGET} PRIVATE CUDA::cudart ggml)
        target_compile_definitions(${TEST_TARGET} PRIVATE
            GGML_USE_CUDA
            GGML_CUDA_FUSION=1
            GGML_CUDA_MIN_BATCH_OFFLOAD=32
            GGML_CUDA_DMMV_X=32
            GGML_CUDA_MMV_Y=1
            GGML_CUDA_PEER_MAX_BATCH_SIZE=128
            K_QUANTS_PER_ITERATION=2)
        set_target_properties(${TEST_TARGET} PROPERTIES CUDA_ARCHITECTURES 75)
        install(TARGETS ${TEST_TARGET} RUNTIME)
        add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
        set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

        # PHASE_MMQ_Q4_0_AR16 A.5: vec_dot_q4_0_ar16_q8_1_mma (INT8 tensor-core)
        # × scalar fp32 reference. Closure binding: cos ≥ 0.9999, NMSE ≤ 1e-4
        # AND byte-identical to A.4 DP4A output on same inputs.
        set(TEST_TARGET test-mmq-q4-0-ar16-mma)
        add_executable(${TEST_TARGET}
            dflash-speculative/test-mmq-q4-0-ar16-mma.cu)
        target_include_directories(${TEST_TARGET} PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CMAKE_SOURCE_DIR}/ggml/include
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(${TEST_TARGET} PRIVATE CUDA::cudart ggml)
        target_compile_definitions(${TEST_TARGET} PRIVATE
            GGML_USE_CUDA
            GGML_CUDA_FUSION=1
            GGML_CUDA_MIN_BATCH_OFFLOAD=32
            GGML_CUDA_DMMV_X=32
            GGML_CUDA_MMV_Y=1
            GGML_CUDA_PEER_MAX_BATCH_SIZE=128
            K_QUANTS_PER_ITERATION=2)
        set_target_properties(${TEST_TARGET} PROPERTIES CUDA_ARCHITECTURES 75)
        install(TARGETS ${TEST_TARGET} RUNTIME)
        add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
        set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

        # PHASE_MMQ_Q4_0_AR16 A.10: closure binding. Plain C++ (no .cu), drives
        # the full ggml-cuda backend via ggml_backend_cuda_init, runs ggml_mul_mat
        # at M ∈ {1,4,8,16,32} with LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1, and
        # bit-compares dst column 0 across M's.
        llama_build_and_test(dflash-speculative/test-mmq-q4-0-ar16-shape-invariance.cpp)
        target_include_directories(test-mmq-q4-0-ar16-shape-invariance PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(test-mmq-q4-0-ar16-shape-invariance PRIVATE ggml)

        # PHASE_MMQ_Q4_0_AR16 Phase B: MMVQ closure binding. With env UNSET,
        # batch ≤ MMVQ_MAX_BATCH_SIZE = 8 routes to MMVQ. Test combines B.2
        # (cosine + NMSE vs CPU fp32 reference at M=1) and B.4 (row-0
        # byte-identical across M ∈ {1, 2, 4, 8}).
        llama_build_and_test(dflash-speculative/test-mmvq-q4-0-ar16-shape-invariance.cpp)
        target_include_directories(test-mmvq-q4-0-ar16-shape-invariance PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(test-mmvq-q4-0-ar16-shape-invariance PRIVATE ggml)

        # Production-geometry variant of MMVQ shape-invariance.
        llama_build_and_test(dflash-speculative/test-mmvq-q4-0-ar16-shape-invariance-prod-dim.cpp)
        target_include_directories(test-mmvq-q4-0-ar16-shape-invariance-prod-dim PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(test-mmvq-q4-0-ar16-shape-invariance-prod-dim PRIVATE ggml)

        # PHASE_MMQ_Q4_0_AR16 Phase C: cuBLAS pinned-algo closure binding.
        # Sweeps F16/BF16/F32 weight types at M ∈ {1, 4, 8, 16, 32} under
        # LLAMA_FATTN_SHAPE_INVARIANT_DISPATCH=1; binds row-0 byte-identical.
        llama_build_and_test(dflash-speculative/test-cublas-pinned-shape-invariant.cpp)
        target_include_directories(test-cublas-pinned-shape-invariant PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(test-cublas-pinned-shape-invariant PRIVATE ggml)
    endif()
endif()

# llama_target_and_test(test-opt.cpp) # SLOW

llama_target_and_test(test-model-load-cancel.cpp  LABEL "model")
llama_target_and_test(test-autorelease.cpp        LABEL "model")

# S4 — Property tests derived from specs/scheduler/batch_composition.allium
# and specs/kv-cache/n_stream_layer.allium. See cached-crunching-tiger.md.

# test-batch-composition-gates: PRE-T4 LEGACY REGRESSION test of the
# two-gate scheduler that USED to close Bug C. The contracts asserted
# here (PrefillSerialisationGate, DecodeHoldGate, MixedBatchProhibition)
# were superseded by the Tier 4 chunked-prefill admission rewrite of
# specs/scheduler/batch_composition.allium (2026-05-22). The test
# remains as a regression check that the legacy gate semantics are
# still implementable in stub form — its internal stub is self-
# contained and does NOT bind on the production code path.
set(TEST_TARGET test-batch-composition-gates)
add_executable(${TEST_TARGET} spec/test-batch-composition-gates.cpp)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

# test-chunked-prefill-admission: T4.1 property test of the Tier 4
# chunked-prefill admission contracts (TokenBudgetPerUbatch,
# DecodePriorityAdmission, ChunkedPrefillAdmission,
# PrefillCarryProgressesMonotonically) from
# specs/scheduler/batch_composition.allium. Stub mirrors the
# admission loop landing in T4.2/T4.4; end-to-end server binding is
# via scripts/validate-batch-composition-trace.py under T4.6.
set(TEST_TARGET test-chunked-prefill-admission)
add_executable(${TEST_TARGET} spec/test-chunked-prefill-admission.cpp)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

# test-runtime-hardening-mlockall: binding test for the --mlockall flag
# and common_apply_runtime_hardening() helper added 2026-05-27 as part
# of the PHASE_NP8_FLAKE code-level RT mitigations (C-1). Treats EPERM
# (unprivileged mlockall) as PASS — the test asserts the helper does
# not crash and is idempotent, not that the runtime has CAP_IPC_LOCK.
set(TEST_TARGET test-runtime-hardening-mlockall)
add_executable(${TEST_TARGET} spec/test-runtime-hardening-mlockall.cpp)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TEST_TARGET} PRIVATE common ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "main")

# test-n-stream-kv-layout RETIRED 2026-05-29: model-driven, non-hermetic,
# never run in CI (SKIP 77 without a model), and its CPU-only setup
# (n_gpu_layers=0) can no longer reach the GPU/split paged-4D layout it
# asserted (PHASE46 B.4 split-config check + T5.9 paged refresh broke its
# preconditions). The invariant it duplicated (KVTensorIsPaged4D /
# StreamPartition) is the gated source of truth in
# specs/kv-cache/n_stream_layer.allium + specs/mgpu-split/CrossCodepathConsistency.allium.
# See PHASE_HYBRID_CHECKPOINT.md §5.13 follow-up.

# P0.B.S4 property tests — derived from the five new Allium specs in
# specs/{graphs,kv-cache,dispatch,dflash,composition}/ (PHASE_NSTREAM_KV_PERF.md).

# test-cuda-graph-reuse: stub-style mirror of the CUDA graph cache state
# machine (cuda_graph_reuse.allium / CUDAGraphReuse.tla). PASS on HEAD.
set(TEST_TARGET test-cuda-graph-reuse)
add_executable(${TEST_TARGET} spec/test-cuda-graph-reuse.cpp)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")

# test-unified-stream-dispatch: stub-style mirror of the Tier 3
# dispatch contract (unified_stream_dispatch.allium /
# UnifiedStreamDispatch.tla). PASS on HEAD — documents the Tier 3
# contract algebraically and explicitly asserts that HEAD per-stream
# dispatch VIOLATES UnifiedUbatchInvariant (the binding RED).
set(TEST_TARGET test-unified-stream-dispatch)
add_executable(${TEST_TARGET} spec/test-unified-stream-dispatch.cpp)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")

# test-mtp-x-n-stream: stub-style mirror of the MTP fused × n_stream
# composition (mtp_fused_x_n_stream.allium / MTPxNStream.tla). PASS
# on HEAD — verifies the structural opt-out at multi-slot dispatch
# (server-context.cpp:3427-3430).
set(TEST_TARGET test-mtp-x-n-stream)
add_executable(${TEST_TARGET} spec/test-mtp-x-n-stream.cpp)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")

# test-dflash-server-cli: stub-style mirror of the --spec-type dflash
# server wiring (dflash_server_cli.allium). PASS on HEAD post-P0.A.4.
# End-to-end coverage at scripts/test-server-multi-slot-dflash.sh.
set(TEST_TARGET test-dflash-server-cli)
add_executable(${TEST_TARGET} spec/test-dflash-server-cli.cpp)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")

# PHASE_NSTREAM_KV_PERF T3.6.T1 — K-shift per-stream binding test.
# FAIL expected on T3.5 HEAD (build_k_shift asserts n_stream == 1).
# PASS after T3.6.I.c1 lifts the assert.
set(TEST_TARGET test-kv-shift-per-stream)
add_executable(${TEST_TARGET} spec/test-kv-shift-per-stream.cpp)
target_include_directories(${TEST_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src)
target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")
set_property(TEST ${TEST_TARGET} PROPERTY SKIP_RETURN_CODE 77)

# PHASE_NSTREAM_KV_PERF T3.6.T2 — defrag per-stream binding test.
# FAIL expected on T3.5 HEAD (build_defrag asserts n_stream == 1).
# PASS after T3.6.I.c2 lifts the assert.
set(TEST_TARGET test-kv-defrag-per-stream)
add_executable(${TEST_TARGET} spec/test-kv-defrag-per-stream.cpp)
target_include_directories(${TEST_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src)
target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")
set_property(TEST ${TEST_TARGET} PROPERTY SKIP_RETURN_CODE 77)

# PHASE_NSTREAM_KV_PERF T3.6.T3 — graph reuse under multi-seq + set_rows.
# FAIL expected on T3.5 HEAD (n_stream > 1 bailout sets miss_reason=6
# at src/llama.cpp:615). PASS after T3.6.I.b drops the bailout.
set(TEST_TARGET test-graph-reuse-set-rows)
add_executable(${TEST_TARGET} spec/test-graph-reuse-set-rows.cpp)
target_include_directories(${TEST_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_SOURCE_DIR}/ggml/src)
target_link_libraries(${TEST_TARGET} PRIVATE common llama ggml ${CMAKE_THREAD_LIBS_INIT})
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")
set_property(TEST ${TEST_TARGET} PROPERTY SKIP_RETURN_CODE 77)

# ============================================================
# PHASE_NSTREAM_KV_PERF T5.0 — paged KV property test stubs.
# ============================================================
# Eight RED-bound tests for the paged KV layout per
# /home/llm/yarn-agentic/PHASE_NSTREAM_KV_PERF.md §Tier 5.
# Each test FAILS on HEAD (LLAMA_PAGED_KV_LANDED undefined) and
# transitions PASS at the noted T5.x card close.
#
# To activate: define LLAMA_PAGED_KV_LANDED=1 as a build flag when
# the production paged KV implementation lands AND the test body
# (where stubbed) is filled in by the T5.x author.

set(TEST_TARGET test-kv-block-allocator)
add_executable(${TEST_TARGET}
    spec/test-kv-block-allocator.cpp
    ${CMAKE_SOURCE_DIR}/src/llama-paged-kv-allocator.cpp
    ${CMAKE_SOURCE_DIR}/src/llama-paged-kv-trace.cpp)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")

# T5.9.E cleanup (2026-05-23): four byte-identity stubs from T5.2/T5.3/T5.5/T5.6
# (test-paged-write-index-formula, test-paged-write-byte-identity-at-identity,
#  test-paged-byte-identity-trivial-mapping, test-paged-multi-seq-byte-identity)
# were deleted. Their contracts were superseded at T5.6 when the paged
# path moved from "byte-layout identity at trivial block_table" to
# "value-equality under identity block_table mapping". Coverage is
# now held by verify-production-determinism.sh (cross-NP byte-identity
# over the full pipeline) + test-paged-kshift-byte-identity (paged K-shift
# specific) + test-paged-defrag-preserves-contents (defrag composition).

set(TEST_TARGET test-paged-kshift-byte-identity)
add_executable(${TEST_TARGET} spec/test-paged-kshift-byte-identity.cpp)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TEST_TARGET} PRIVATE LLAMA_PAGED_KV_LANDED=1)
target_link_libraries(${TEST_TARGET} PRIVATE llama)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")

set(TEST_TARGET test-paged-defrag-preserves-contents)
add_executable(${TEST_TARGET}
    spec/test-paged-defrag-preserves-contents.cpp
    ${CMAKE_SOURCE_DIR}/src/llama-paged-kv-allocator.cpp
    ${CMAKE_SOURCE_DIR}/src/llama-paged-kv-trace.cpp)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TEST_TARGET} PRIVATE LLAMA_PAGED_KV_LANDED=1)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")

set(TEST_TARGET test-paged-allocator-determinism)
add_executable(${TEST_TARGET}
    spec/test-paged-allocator-determinism.cpp
    ${CMAKE_SOURCE_DIR}/src/llama-paged-kv-allocator.cpp
    ${CMAKE_SOURCE_DIR}/src/llama-paged-kv-trace.cpp)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")

# T5.9.A.4 — paged BACKING binding tests
set(TEST_TARGET test-pool-bounds-respected)
add_executable(${TEST_TARGET}
    spec/test-pool-bounds-respected.cpp
    ${CMAKE_SOURCE_DIR}/src/llama-paged-kv-allocator.cpp
    ${CMAKE_SOURCE_DIR}/src/llama-paged-kv-trace.cpp)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")

# PHASE_HYBRID_STATE_RESTORE — paged resync reclaim (single-stream conservation
# + identity). Models the seq_rm resync that was gated off for n_stream==1.
set(TEST_TARGET test-paged-resync-reclaim)
add_executable(${TEST_TARGET}
    spec/test-paged-resync-reclaim.cpp
    ${CMAKE_SOURCE_DIR}/src/llama-paged-kv-allocator.cpp
    ${CMAKE_SOURCE_DIR}/src/llama-paged-kv-trace.cpp)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")

set(TEST_TARGET test-paged-backing-feasibility)
add_executable(${TEST_TARGET}
    spec/test-paged-backing-feasibility.cpp
    ${CMAKE_SOURCE_DIR}/src/llama-paged-kv-allocator.cpp
    ${CMAKE_SOURCE_DIR}/src/llama-paged-kv-trace.cpp)
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")

# ---------------------------------------------------------------------------
# PHASE 46 — Multi-GPU CLIP tests.
#
# Two pure-spec tests (multi-backend-init, weight-split) PASS on any host;
# two harness-driven tests (encode-equivalence, encode-latency) SKIP with
# exit 77 when their JSON input is missing, and PASS/FAIL against the
# JSON produced by scripts/verify-multigpu-clip.sh during the maintenance
# window. See docs/phases/80-multimodal/PHASE46-MULTIGPU-CLIP-TENSOR-SPLIT.md
# §10 (B.5 sub-tests) and §11.5 (B.7 perf gate).
# ---------------------------------------------------------------------------

set(TEST_TARGET test-clip-multi-backend-init)
add_executable(${TEST_TARGET} spec/test-clip-multi-backend-init.cpp)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")

set(TEST_TARGET test-clip-weight-split)
add_executable(${TEST_TARGET} spec/test-clip-weight-split.cpp)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")

# Spec test for CrossCodepathConsistency.allium — structural invariants
# of the LM ⇔ CLIP cross-codepath contract over a shared mgpu_split_config.
set(TEST_TARGET test-mgpu-split-cross-codepath)
add_executable(${TEST_TARGET} spec/test-mgpu-split-cross-codepath.cpp)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")

set(TEST_TARGET test-clip-encode-equivalence)
add_executable(${TEST_TARGET} spec/test-clip-encode-equivalence.cpp)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")
set_property(TEST ${TEST_TARGET} PROPERTY SKIP_RETURN_CODE 77)

set(TEST_TARGET test-clip-encode-latency)
add_executable(${TEST_TARGET} spec/test-clip-encode-latency.cpp)
target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
install(TARGETS ${TEST_TARGET} RUNTIME)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
set_property(TEST ${TEST_TARGET} PROPERTY LABELS "spec")
set_property(TEST ${TEST_TARGET} PROPERTY SKIP_RETURN_CODE 77)

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

# audit F.2: capture loader / byte-identity harness (PLAN_DETERMINISM_AUDIT.md)
add_executable(test-capture-loader-smoke audit/test-capture-loader-smoke.cpp)
target_include_directories(test-capture-loader-smoke PRIVATE audit)
target_compile_features(test-capture-loader-smoke PRIVATE cxx_std_17)



# local diagnostic only (plan/0132 box 2 — Windows CUDA op-correctness localization)
add_executable(test-backend-ops test-backend-ops.cpp)
target_link_libraries(test-backend-ops PRIVATE common)
