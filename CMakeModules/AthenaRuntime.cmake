function(athena_disable_rtti target_name)
    get_target_property(CURRENT_COMPILE_OPTIONS ${target_name} COMPILE_OPTIONS)
    if (${CURRENT_COMPILE_OPTIONS} STREQUAL "CURRENT_COMPILE_OPTIONS-NOTFOUND")
        set(CURRENT_COMPILE_OPTIONS "")
    endif()
    if (WIN32)
        set(NEW_COMPILE_OPTIONS "/GR-")
    else()
        set(NEW_COMPILE_OPTIONS "-fno-rtti")
    endif()

    target_compile_options(${target_name} PUBLIC "${CURRENT_COMPILE_OPTIONS};${NEW_COMPILE_OPTIONS}")
endfunction(athena_disable_rtti)

function(athena_disable_exceptions target_name)
    get_target_property(CURRENT_COMPILE_OPTIONS ${target_name} COMPILE_OPTIONS)
    if (${CURRENT_COMPILE_OPTIONS} STREQUAL "CURRENT_COMPILE_OPTIONS-NOTFOUND")
        set(CURRENT_COMPILE_OPTIONS "")
    endif()
    if (WIN32)
        set(NEW_COMPILE_OPTIONS "/EHsc-")
    else()
        set(NEW_COMPILE_OPTIONS "-fno-exceptions")
    endif()

    target_compile_options(${target_name} PUBLIC "${CURRENT_COMPILE_OPTIONS};${NEW_COMPILE_OPTIONS}")
endfunction(athena_disable_exceptions)