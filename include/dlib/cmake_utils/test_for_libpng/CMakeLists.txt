
cmake_minimum_required(VERSION 2.8.12)
project(test_if_libpng_is_broken)

find_package(PNG)

include_directories(${PNG_INCLUDE_DIR})
add_executable(libpng_test libpng_test.cpp)
target_link_libraries(libpng_test ${PNG_LIBRARIES})


