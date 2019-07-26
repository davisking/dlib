cmake_minimum_required(VERSION 3.10)

project(dlibfuzzing)

add_subdirectory(../dlib dlib_build)

add_executable(fuzz_imageload imageload.cpp)
target_link_libraries(fuzz_imageload dlib::dlib)

add_executable(fuzz_xmlparse xmlparse.cpp)
target_link_libraries(fuzz_xmlparse dlib::dlib)

# does not work
#target_compile_options(fuzz_imageload PUBLIC -fsanitize=fuzzer)
string(APPEND CMAKE_CXX_FLAGS " -fsanitize=fuzzer")
