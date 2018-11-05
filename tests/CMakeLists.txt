function(get_gtest)
    include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
    ExternalProject_Add( googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG "master"

            UPDATE_COMMAND ""
            PATCH_COMMAND ""

            PREFIX ${PROJECT_SOURCE_DIR}/libs/googletest
            SOURCE_DIR "googletest"

            INSTALL_COMMAND ""
    )

    ExternalProject_Get_Property(googletest source_dir)
    set(GTEST_INCLUDE_DIR ${source_dir}/include)
    ExternalProject_Get_Property(googletest binary_dir)
    set(GTEST_LIBRARY_PATH ${binary_dir}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a)
    set(GTEST_LIBRARY gtest)
    add_library(${GTEST_LIBRARY} UNKNOWN IMPORTED)
endfunction(get_gtest)

enable_testing()
find_package(GTest)

if (${GTEST_FOUND})
else()
    get_gtest()
endif()