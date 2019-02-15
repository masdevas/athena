include(AthenaRuntime)

function(add_athena_library target_name modifier)
    set(source_list ${ARGN})
    add_library(${target_name} ${modifier} ${source_list})

    if(NOT "${modifier}" STREQUAL "INTERFACE")
        athena_disable_rtti(${target_name})
        athena_disable_exceptions(${target_name})
    endif()
endfunction(add_athena_library)

function(add_athena_executable target_name)
    set(source_list ${ARGN})
    add_executable(${target_name} ${modifier} ${source_list})
    athena_disable_rtti(${target_name})
    athena_disable_exceptions(${target_name})
endfunction()
