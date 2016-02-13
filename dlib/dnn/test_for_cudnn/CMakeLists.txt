
cmake_minimum_required(VERSION 2.8.4)
project(cudnn_test)
include(../../use_cpp_11.cmake)

include_directories(${cudnn_include})

find_package(CUDA 7.0 REQUIRED)
set(CUDA_HOST_COMPILATION_CPP ON)
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-D__STRICT_ANSI__")
add_definitions(-DDLIB_USE_CUDA)

include(find_cudnn.txt)

cuda_add_library(cudnn_test STATIC ../cudnn_dlibapi.cpp ${cudnn} )
