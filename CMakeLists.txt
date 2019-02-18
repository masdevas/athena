cmake_minimum_required(VERSION 3.12)
project(athena)

function(set_sanitizers)
    ## Flags ##
    set(SEQENCE_SANITIZERS_FLAG seq)
    set(PARALLEL_SANITIZERS_FLAG par)
    set(OFF_SANITIZERS_FLAG off)

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
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
        message(FATAL_ERROR "-- Sanitizers: WRONG FLAG")
    endif()
    message("-- Sanitizers: ${SANITIZERS}")
    set(SANITIZERS ${SANITIZERS} PARENT_SCOPE)
endfunction()

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)
set_sanitizers()
if(SANITIZERS)
    set(CMAKE_CXX_FLAGS -fsanitize=${SANITIZERS})
else()
    message("-- Sanitizers: off")
endif()

include(AthenaTarget)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/null.cpp "")
add_athena_library(athena SHARED ${CMAKE_CURRENT_BINARY_DIR}/null.cpp)

add_subdirectory(src)

target_link_libraries(athena PUBLIC athena-core backend-llvm)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    add_subdirectory(tests)
    if ($ENV{USE_CCOV})
        include(CodeCoverage)
        target_compile_options(runUnitCore PUBLIC -g -O0 --coverage -fprofile-arcs -ftest-coverage)
    endif()
endif()

