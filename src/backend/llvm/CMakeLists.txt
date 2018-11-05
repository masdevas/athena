add_subdirectory(runtime-cpu)

add_library(backend-llvm INTERFACE)

target_link_libraries(backend-llvm INTERFACE athena-core runtime-llvm-cpu)