find_package(LLVM CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

add_subdirectory(runtime-cpu)
add_subdirectory(runtime-driver)

file(GLOB codegen_src "codegen/*.cpp")

add_athena_library(backend-llvm STATIC
        LLVMExecutor.cpp
        LLVMGenerator.cpp
        AthenaJIT.cpp
        LLVMTrivialAllocator.cpp
        utils.cpp
        ${codegen_src})

target_include_directories(backend-llvm PUBLIC ${LLVM_INCLUDE_DIRS})
target_compile_definitions(backend-llvm PUBLIC ${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs core irreader support transformutils passes codegen orcjit x86asmparser x86codegen)

target_link_libraries(backend-llvm PUBLIC athena-core runtime-llvm-cpu ${llvm_libs})
# TODO make JIT link against executable
target_link_libraries(backend-llvm PUBLIC MemoryLoader)
