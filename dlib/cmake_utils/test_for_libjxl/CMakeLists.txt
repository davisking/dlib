
cmake_minimum_required(VERSION 3.8.0)
project(test_if_libjxl_is_broken)

include_directories(${JXL_INCLUDE_DIR})
add_executable(libjxl_test libjxl_test.cpp)
target_link_libraries(libjxl_test ${JXL_LIBRARY})
