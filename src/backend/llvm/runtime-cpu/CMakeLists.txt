function(getboost)
    find_package(Boost COMPONENTS numeric)

    if(NOT BLAS_FOUND)
        if (NOT WIN32)
            execute_process(COMMAND ${PROJECT_SOURCE_DIR}/scripts/boost.sh 1.68.0 ${PROJECT_SOURCE_DIR}/libs)
            set(BOOST_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/libs/boost)
        endif()
    endif()

    target_include_directories(runtime-llvm-cpu PUBLIC ${BOOST_INCLUDE_DIRS})
endfunction(getboost)

function(getblis)
    include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
    find_program(MAKE_EXE NAMES gmake nmake make)
    externalproject_add(blis

            GIT_REPOSITORY "https://github.com/flame/blis.git"
            GIT_TAG "master"

            UPDATE_COMMAND ""
            PATCH_COMMAND ""

            PREFIX ${PROJECT_SOURCE_DIR}/libs/src/
            SOURCE_DIR "blis"
            INSTALL_DIR "blis"

            CONFIGURE_COMMAND ./configure --prefix=${PROJECT_SOURCE_DIR}/libs/src/blis/include/blis/ --enable-cblas auto

            BUILD_IN_SOURCE TRUE
            BUILD_COMMAND "make"

            INSTALL_COMMAND ${MAKE_EXE} install
            )
    externalproject_get_property(blis SOURCE_DIR)
    add_dependencies(athena blis)
    target_include_directories(athena PUBLIC ${PROJECT_SOURCE_DIR}/libs/src/blis/include/blis/include/blis)
    target_compile_definitions(athena PUBLIC -DFORCE_BLIS)
    set(BLAS_LIBRARIES ${PROJECT_SOURCE_DIR}/libs/src/blis/include/blis/lib/libblis.a
            PARENT_SCOPE)
endfunction(getblis)

option(FORCE_BLIS "Require library to use BLIS" OFF)
option(FORCE_OPENBLAS "Require library to use OpenBLAS" OFF)

#if (FORCE_BLIS)
#    message(STATUS "BLIS will be downloaded and compiled")
#    getblis()
#
#elseif (FORCE_UBLAS)
#    message(STATUS "Boost will be downloaded and compiled")
#    getboost()
#else ()
#    if (APPLE)
#        set(BLA_VENDOR Apple)
#        target_compile_definitions(athena PUBLIC -DAPPLE_ACCELERATE)
#        message(STATUS "Apple Accelerate Framework will be used")
#    else()
#        message(STATUS "No BLAS installation found. Boost will be downloaded and compiled")
#        getboost()
#    endif()
#endif ()

add_athena_library(runtime-llvm-cpu STATIC add.cpp)