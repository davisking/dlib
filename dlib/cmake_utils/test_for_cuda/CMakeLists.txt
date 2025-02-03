
cmake_minimum_required(VERSION 3.10.0)
project(cuda_test)

include_directories(../../cuda)
add_definitions(-DDLIB_USE_CUDA)

# Override the FindCUDA.cmake setting to avoid duplication of host flags if using a toolchain:
option(CUDA_PROPAGATE_HOST_FLAGS "Propage C/CXX_FLAGS and friends to the host compiler via -Xcompile" OFF)
find_package(CUDA 7.5 REQUIRED)
set(CUDA_HOST_COMPILATION_CPP ON)
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_50;-std=c++14;-D__STRICT_ANSI__;-D_MWAITXINTRIN_H_INCLUDED;-D_FORCE_INLINES")

cuda_add_library(cuda_test STATIC cuda_test.cu )
