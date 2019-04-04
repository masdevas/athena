file(GLOB_RECURSE CORE_SRC "*.cpp")

add_athena_library(athena-core STATIC ${CORE_SRC})