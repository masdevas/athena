include(AthenaIntegrationTest)
add_subdirectory(JIT)
add_subdirectory(runtime-driver-test)

configure_file(${INTEG_RUN_SCRIPT_NAME} ${INTEG_RUN_SCRIPT_NAME} COPYONLY)
