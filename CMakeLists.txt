cmake_minimum_required(VERSION 3.12)
project(athena)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_library(athena INTERFACE)

add_subdirectory(src)
add_subdirectory(tests)

target_link_libraries(athena INTERFACE athena-core)
