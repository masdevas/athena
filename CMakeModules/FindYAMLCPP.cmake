function(__get_yaml_cpp)
    configure_file(${CMAKE_MODULE_PATH}/YAMLCPP_CMakeLists.txt.in ${CMAKE_BINARY_DIR}/third_party/yaml-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/third_party/yaml-download)
    if (result)
        message(FATAL_ERROR "CMake step for yamlcpp failed: ${result}")
    endif ()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/third_party/yaml-download)
    if (result)
        message(FATAL_ERROR "Build step for yaml-cpp failed: ${result}")
    endif ()

    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}"
            -DYAML_CPP_BUILD_TESTS=OFF
            -DYAML_CPP_BUILD_TOOLS=OFF
            -DYAML_CPP_BUILD_CONTRIB=OFF
            -DYAML_CPP_INSTALL=OFF
            ${CMAKE_BINARY_DIR}/third_party/yaml-src
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/third_party/yaml-build)
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/third_party/yaml-build)
    if (result)
        message(FATAL_ERROR "Library build step for yaml-cpp failed: ${result}")
    endif ()
endfunction()

find_library(yaml_lib yaml-cpp)
find_path(yaml_header yaml-cpp/yaml.h)

set(yaml_include_dir ${yaml_header} CACHE STRING "")

set(YAML_DOWNLOADED OFF CACHE BOOL "")

if (${yaml_lib} MATCHES "yaml_lib-NOTFOUND")
    if (NOT ${YAML_DOWNLOADED})
        __get_yaml_cpp()
        set(YAML_DOWNLOADED ON CACHE BOOL "" FORCE)
    endif ()
    find_library(yaml_lib yaml-cpp PATHS ${CMAKE_BINARY_DIR}/third_party/yaml-build)
    set(yaml_include_dir "${CMAKE_BINARY_DIR}/third_party/yaml-src/include" CACHE STRING "" FORCE)
endif ()

set(YAMLCPP_LIBRARY ${yaml_lib})
set(YAMLCPP_INCLUDE_DIR ${yaml_include_dir})