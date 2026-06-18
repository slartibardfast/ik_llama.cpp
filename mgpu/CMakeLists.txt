# libmgpu — Megatron-style tensor-parallelism graph builders
#
# Layered above libggml; consumed by libllama and libmtmd.
# See yarn-agentic plan: /home/dconnolly/.claude/plans/examine-how-we-do-serene-liskov.md

# TODO: should not use this — match ggml/llama so the shared mgpu lib exports its
# symbols on Windows (otherwise the DLL has no exports and no import .lib, and
# libllama fails to link against mgpu).
if (WIN32)
    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

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
