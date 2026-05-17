set(TARGET test-capture-loader-smoke)
add_executable(${TARGET} test-capture-loader-smoke.cpp)
target_include_directories(${TARGET} PRIVATE .)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
