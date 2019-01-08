include(AthenaRuntime)

function(add_athena_library target_name modifier)
    set(source_list ${ARGN})
    add_library(${target_name} ${modifier} ${source_list})

    if(NOT "${modifier}" STREQUAL "INTERFACE")
        athena_disable_rtti(${target_name})
        athena_disable_exceptions(${target_name})
    endif()

    if("${modifier}" STREQUAL "SHARED")
        if (WIN32)
            set_target_properties(${target_name} PROPERTIES
                    LINK_FLAGS "/WHOLEARCHIVE"
                    )
        elseif (APPLE)
            set_target_properties(${target_name} PROPERTIES
                    LINK_FLAGS "-Wl,-all_load"
                    )
        else ()
            set_target_properties(${target_name} PROPERTIES
                    LINK_FLAGS "-Wl,--whole-archive"
                    )
        endif ()
    endif()
endfunction(add_athena_library)

function(add_athena_executable target_name)
    set(source_list ${ARGN})
    add_executable(${target_name} ${modifier} ${source_list})
    athena_disable_rtti(${target_name})
    athena_disable_exceptions(${target_name})
endfunction()