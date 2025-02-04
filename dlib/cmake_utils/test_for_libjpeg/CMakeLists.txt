
cmake_minimum_required(VERSION 3.10.0)
project(test_if_libjpeg_is_broken)

find_package(JPEG)

include_directories(${JPEG_INCLUDE_DIR})
add_executable(libjpeg_test libjpeg_test.cpp)
target_link_libraries(libjpeg_test ${JPEG_LIBRARY})


