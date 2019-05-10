add_subdirectory(framework)

include(AthenaIntegrationTest)
add_subdirectory(JIT)
add_subdirectory(RuntimeDriver)

configure_file(${INTEG_RUN_SCRIPT_NAME} ${INTEG_RUN_SCRIPT_NAME} COPYONLY)
