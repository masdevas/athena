cmake_minimum_required(VERSION 3.12)
project(athena)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/null.cpp "")
add_library(athena STATIC ${CMAKE_CURRENT_BINARY_DIR}/null.cpp)

add_subdirectory(src)

target_link_libraries(athena INTERFACE athena-core)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_subdirectory(tests)
    if ($ENV{USE_CCOV})
        include(CodeCoverage)
        target_compile_options(runTests PUBLIC -g -O0 --coverage -fprofile-arcs -ftest-coverage)
        target_compile_options(runIntegrationTests PUBLIC -g -O0 --coverage -fprofile-arcs -ftest-coverage)
    endif()
endif()
