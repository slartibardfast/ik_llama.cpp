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
    llama_build_and_test(mtp-verify-accept/test-accept-symbols.cpp)
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

        llama_build_and_test(dflash-speculative/test-dflash-drafter-forward.cpp)
        target_include_directories(test-dflash-drafter-forward PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(test-dflash-drafter-forward PRIVATE CUDA::cudart)

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

        llama_build_and_test(dflash-speculative/test-dflash-end-to-end.cpp)
        target_include_directories(test-dflash-end-to-end PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CMAKE_SOURCE_DIR}/ggml/include
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(test-dflash-end-to-end PRIVATE CUDA::cudart ggml)

        llama_build_and_test(dflash-speculative/test-dflash-closure.cpp)
        target_include_directories(test-dflash-closure PRIVATE
            ${CMAKE_SOURCE_DIR}/ggml/src
            ${CMAKE_SOURCE_DIR}/ggml/include
            ${CUDAToolkit_INCLUDE_DIRS})
        target_link_libraries(test-dflash-closure PRIVATE CUDA::cudart ggml)

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
    endif()
endif()

# llama_target_and_test(test-opt.cpp) # SLOW

llama_target_and_test(test-model-load-cancel.cpp  LABEL "model")
llama_target_and_test(test-autorelease.cpp        LABEL "model")


# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)


