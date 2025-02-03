
cmake_minimum_required(VERSION 3.10.0)
project(test_if_libwebp_is_broken)

include_directories(${WEBP_INCLUDE_DIR})
add_executable(libwebp_test libwebp_test.cpp)
target_link_libraries(libwebp_test ${WEBP_LIBRARY})
