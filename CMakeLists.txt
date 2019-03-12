cmake_minimum_required(VERSION 3.12)
project(athena)

set(ATHENA_MAJOR_VERSION "0")
set(ATHENA_MINOR_VERSION "1")
set(ATHENA_PATCH_VERSION "0")

function(set_sanitizers)
    ## Flags ##
    set(SEQENCE_SANITIZERS_FLAG seq)
    set(PARALLEL_SANITIZERS_FLAG par)
    set(OFF_SANITIZERS_FLAG off)

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        if(DEFINED FSANITIZE)
            message(STATUS "Flag FSANITIZE ignored due to build type")
        endif()
        return()
    elseif(NOT DEFINED FSANITIZE)
        return()
    elseif(${FSANITIZE} STREQUAL ${OFF_SANITIZERS_FLAG})
        return()
    elseif(${FSANITIZE} STREQUAL ${SEQENCE_SANITIZERS_FLAG})
        set(SANITIZERS undefined,address,leak)
    elseif(${FSANITIZE} STREQUAL ${PARALLEL_SANITIZERS_FLAG})
        set(SANITIZERS undefined,thread)
    else()
        message(FATAL_ERROR "Sanitizers: WRONG FLAG")
    endif()
    message(STATUS "Sanitizers: ${SANITIZERS}")
    set(SANITIZERS ${SANITIZERS} PARENT_SCOPE)
endfunction()

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)
set_sanitizers()
if(SANITIZERS)
    set(CMAKE_CXX_FLAGS -fsanitize=${SANITIZERS})
else()
    message(STATUS "Sanitizers: off")
endif()

include(AthenaTarget)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/null.cpp "")
add_athena_library(athena SHARED ${CMAKE_CURRENT_BINARY_DIR}/null.cpp)
add_definitions(-DATHENA_MAJOR_VERSION="${ATHENA_MAJOR_VERSION}")
add_definitions(-DATHENA_MINOR_VERSION="${ATHENA_MINOR_VERSION}")
add_definitions(-DATHENA_PATCH_VERSION="${ATHENA_PATCH_VERSION}")
if (DEFINED $ENV{ATHENA_CI_BUILD_NUMBER})
    add_definitions(-DATHENA_PATCH_VERSION="$ENV{ATHENA_CI_BUILD_NUMBER}")
endif()

add_subdirectory(src)

target_link_libraries(athena PUBLIC athena-core athena-ops backend-llvm)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_subdirectory(tests)
    if ($ENV{USE_CCOV})
        include(CodeCoverage)
        target_compile_options(runUnitCore PUBLIC -g -O0 --coverage -fprofile-arcs -ftest-coverage)
    endif()
endif()

