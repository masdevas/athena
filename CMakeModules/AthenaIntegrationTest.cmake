function(add_athena_integration_test)
    cmake_parse_arguments(PARSED_ARGS "" "TARGET_NAME" "SRCS;LIBS" ${ARGN})
    add_executable(${PARSED_ARGS_TARGET_NAME} ${modifier} ${PARSED_ARGS_SRCS})
    target_link_libraries(${PARSED_ARGS_TARGET_NAME} ${PARSED_ARGS_LIBS})
    athena_disable_rtti(${PARSED_ARGS_TARGET_NAME})
    athena_disable_exceptions(${PARSED_ARGS_TARGET_NAME})
    configure_file(../integration_test_run.py ../integration_test_run.py COPYONLY)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/TestSuite_ci.conf)
        configure_file(TestSuite_ci.conf TestSuite_ci.xml COPYONLY)
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/TestSuite_dev.conf)
        configure_file(TestSuite_dev.conf TestSuite_dev.xml COPYONLY)
    endif()
    add_custom_command(OUTPUT run_test_script COMMAND python3 ../integration_test_run.py ${PARSED_ARGS_TARGET_NAME})
    add_custom_target(${PARSED_ARGS_TARGET_NAME}-run ALL DEPENDS ${PARSED_ARGS_TARGET_NAME} run_test_script)
endfunction(add_athena_integration_test)
