cmake_minimum_required(VERSION 3.17)

project(dlib_python_bindings)
set(DLIB_BUILD_PYTHON ON CACHE BOOL "Building python")
add_subdirectory(../../ dlib_build)