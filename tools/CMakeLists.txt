# PHASE46 — determinism tooling.
#
# trace-validate: standalone NDJSON validator (no llama linkage).

add_executable(trace-validate trace-validate.cpp)
install(TARGETS trace-validate RUNTIME)
