cmake_minimum_required(VERSION 3.16)
project(mesa_repro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vulkan REQUIRED)

# Locate glslc for compiling the compute shader to SPIR-V at build time.
find_program(GLSLC_EXECUTABLE glslc HINTS ENV VULKAN_SDK PATH_SUFFIXES bin)
if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found; install shaderc or Vulkan SDK")
endif()

set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shader.comp)
set(SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/shader.spv)

add_custom_command(
    OUTPUT ${SHADER_SPV}
    COMMAND ${GLSLC_EXECUTABLE}
            --target-env=vulkan1.2
            -O
            -o ${SHADER_SPV}
            ${SHADER_SRC}
    DEPENDS ${SHADER_SRC}
    COMMENT "glslc shader.comp -> shader.spv"
    VERBATIM
)

add_custom_target(mesa_repro_shader ALL DEPENDS ${SHADER_SPV})

add_executable(repro repro.cpp)
add_dependencies(repro mesa_repro_shader)
target_link_libraries(repro PRIVATE Vulkan::Vulkan)
