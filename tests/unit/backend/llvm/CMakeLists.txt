file(GLOB CORE_TEST_SRC "*.cpp")

add_athena_executable(runUnitLLVMBackend ${CORE_TEST_SRC})
target_link_libraries(runUnitLLVMBackend athena-core backend-llvm)
target_link_libraries(runUnitLLVMBackend ${GTEST_BOTH_LIBRARIES} Threads::Threads)
target_include_directories(runUnitLLVMBackend PUBLIC ${GTEST_INCLUDE_DIRS})

add_test(NAME UnitLLVMBackendTests COMMAND runUnitLLVMBackend)