# libmgpu — Megatron-style tensor-parallelism graph builders
#
# Layered above libggml; consumed by libllama and libmtmd.
# See yarn-agentic plan: /home/dconnolly/.claude/plans/examine-how-we-do-serene-liskov.md

add_library(mgpu
            include/mgpu/types.h
            include/mgpu/classify.h
            include/mgpu/tp.h
            src/classify.cpp
            src/tp.cpp
            )

target_include_directories(mgpu PUBLIC  include)
target_include_directories(mgpu PRIVATE ../ggml/include ../ggml/src)
target_link_libraries     (mgpu PUBLIC  ggml)
target_compile_features   (mgpu PUBLIC  cxx_std_17)

if (BUILD_SHARED_LIBS)
    set_target_properties     (mgpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(mgpu PRIVATE LLAMA_BUILD)
endif()
