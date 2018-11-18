function(getopenblas)
    include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
    find_program(MAKE_EXE NAMES gmake nmake make)
    externalproject_add(OpenBLAS
        GIT_REPOSITORY "https://github.com/xianyi/OpenBLAS.git"
        GIT_TAG "master"

        UPDATE_COMMAND ""
        PATCH_COMMAND ""

        PREFIX ${PROJECT_SOURCE_DIR}/libs
        SOURCE_DIR "openblas"
        INSTALL_DIR "openblas"

        CMAKE_GENERATOR "Ninja"

        CMAKE_ARGS -DNOFORTRAN=1 -DBUILD_WITHOUT_LAPACK=yes -DCMAKE_BUILD_TYPE=Release

        BUILD_IN_SOURCE TRUE

        INSTALL_COMMAND ${MAKE_EXE} PREFIX=${PROJECT_SOURCE_DIR}/libs/src/openblas install
    )
    externalproject_get_property(OpenBLAS SOURCE_DIR)
    add_dependencies(athena OpenBLAS)
    target_include_directories(athena PUBLIC
            ${PROJECT_SOURCE_DIR}/libs/src/openblas/include)
    target_compile_definitions(athena PUBLIC -DFORCE_OPENBLAS)
    set(BLAS_LIBRARIES ${PROJECT_SOURCE_DIR}/libs/src/openblas/lib/libopenblas.a
            PARENT_SCOPE)
endfunction(getopenblas)

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


if (FORCE_BLIS)
    message(STATUS "BLIS will be downloaded and compiled")
    getblis()

elseif (FORCE_OPENBLAS)
    message(STATUS "OpenBLAS will be downloaded and compiled")
    getopenblas()
else ()
    if (APPLE)
        set(BLA_VENDOR Apple)
        target_compile_definitions(athena PUBLIC -DAPPLE_ACCELERATE)
        message(STATUS "Apple Accelerate Framework will be used")
    endif ()
    find_package(BLAS)
    if (NOT BLAS_FOUND)
        message(STATUS "No BLAS installation found. OpenBLAS will be downloaded and compiled")
        getopenblas()
    endif()
endif ()

add_library(runtime-llvm-cpu INTERFACE)