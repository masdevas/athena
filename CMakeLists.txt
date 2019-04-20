cmake_minimum_required(VERSION 3.9)
project(athena)

set(ATHENA_MAJOR_VERSION "0")
set(ATHENA_MINOR_VERSION "1")
set(ATHENA_PATCH_VERSION "0")

option(DISABLE_TESTS "OFF")
option(ENABLE_CODE_COVERAGE "OFF")
set(USE_SANITIZERS "OFF" CACHE STRING "Specify Google Sanitizers")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)
# todo remove hacks http://lists.llvm.org/pipermail/llvm-dev/2016-May/099874.html
if (UNIX AND NOT APPLE)
set(CMAKE_CXX_FLAGS  "-Wl,--export-dynamic")
endif()

include(AthenaTarget)
include(AthenaEnableSanitizers)

enable_testing()
enable_sanitizers()

include_directories(${CMAKE_SOURCE_DIR}/include)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/null.cpp "")
add_athena_library(athena SHARED ${CMAKE_CURRENT_BINARY_DIR}/null.cpp)
add_definitions(-DATHENA_MAJOR_VERSION="${ATHENA_MAJOR_VERSION}")
add_definitions(-DATHENA_MINOR_VERSION="${ATHENA_MINOR_VERSION}")
add_definitions(-DATHENA_PATCH_VERSION="${ATHENA_PATCH_VERSION}")
if (DEFINED $ENV{ATHENA_CI_BUILD_NUMBER})
    add_definitions(-DATHENA_PATCH_VERSION="$ENV{ATHENA_CI_BUILD_NUMBER}")
endif()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Debug build contains assertions")
    add_definitions(-DDEBUG)
endif()

add_subdirectory(src)

target_link_libraries(athena PUBLIC athena-core athena-ops backend-llvm)


if (NOT DISABLE_TESTS)
    include(AthenaCoverageOptions)
    enable_coverage()
    add_subdirectory(tests)
endif()

