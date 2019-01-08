file(GLOB CORE_TEST_SRC "*.cpp")

add_athena_executable(runUnitCore ${CORE_TEST_SRC})
target_link_libraries(runUnitCore athena-core)
target_link_libraries(runUnitCore ${GTEST_MAIN_LIBRARIES})
target_include_directories(runUnitCore PUBLIC ${GTEST_INCLUDE_DIRS})

add_test(NAME UnitCoreTests COMMAND runUnitCore)

target_compile_definitions(runUnitCore PUBLIC TEST_ENVIRONMENT)

if($ENV{USE_CCOV})
    target_link_libraries(runUnitCore gcov)
endif()