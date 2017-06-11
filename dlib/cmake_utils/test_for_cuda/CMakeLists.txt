
cmake_minimum_required(VERSION 2.8.12)
project(cuda_test)

include_directories(../../dnn)
add_definitions(-DDLIB_USE_CUDA)

find_package(CUDA 7.5 REQUIRED)
set(CUDA_HOST_COMPILATION_CPP ON)
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-D__STRICT_ANSI__;-D_MWAITXINTRIN_H_INCLUDED;-D_FORCE_INLINES")

cuda_add_library(cuda_test STATIC cuda_test.cu )
