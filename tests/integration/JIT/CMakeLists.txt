add_athena_integration_test(
        TARGET_NAME JITIntegrationTests
        SRCS main.cpp
        LIBS athena runtime-driver MemoryLoader
        CONF_STRS ATHENA_RT_LIB_dev=${PROJECT_BINARY_DIR}/src/backend/llvm/runtime-cpu/libruntime-llvm-cpu
)
