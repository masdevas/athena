find_package(LLVM CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

add_subdirectory(runtime-cpu)

add_library(backend-llvm INTERFACE)

target_link_libraries(backend-llvm INTERFACE athena-core runtime-llvm-cpu)