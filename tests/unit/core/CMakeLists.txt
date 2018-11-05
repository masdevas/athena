file(GLOB CORE_TEST_SRC "*.cpp")

add_executable(runUnitCore ${CORE_TEST_SRC})
target_link_libraries(runUnitCore athena-core)
target_link_libraries(runUnitCore ${GTEST_MAIN_LIBRARIES})
target_include_directories(runUnitCore PUBLIC ${GTEST_INCLUDE_DIRS})

add_test(NAME UnitCoreTests COMMAND runUnitCore)