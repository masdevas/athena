find_package(YAMLCPP)
find_package(Boost 1.50.0 COMPONENTS filesystem REQUIRED)

add_library(IntegrationTestFramework OBJECT framework.cpp)

configure_file(options.h.in options.h)

target_include_directories(IntegrationTestFramework PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(IntegrationTestFramework PUBLIC ${GTEST_INCLUDE_DIRS})
target_include_directories(IntegrationTestFramework PUBLIC ${YAMLCPP_INCLUDE_DIR})
target_include_directories(IntegrationTestFramework PUBLIC ${Boost_INCLUDE_DIRS})

