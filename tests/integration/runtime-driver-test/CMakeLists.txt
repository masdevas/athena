add_athena_integration_test(TARGET_NAME runtime-driver-test
                            SRCS main.cpp
                            LIBS runtime-driver
                            CONF_STRS PATH_TO_RUNTIME_CPU_dev=${PROJECT_BINARY_DIR}/src/backend/llvm/runtime-cpu/libruntime-llvm-cpu.so)