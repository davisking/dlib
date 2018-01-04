
cmake_minimum_required(VERSION 2.8.12)
project(cudnn_test)
include(../use_cpp_11.cmake)

# Override the FindCUDA.cmake setting to avoid duplication of host flags if using a toolchain:
option(CUDA_PROPAGATE_HOST_FLAGS "Propage C/CXX_FLAGS and friends to the host compiler via -Xcompile" OFF)
find_package(CUDA 7.5 REQUIRED)
set(CUDA_HOST_COMPILATION_CPP ON)
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-D__STRICT_ANSI__")
add_definitions(-DDLIB_USE_CUDA)

include(find_cudnn.txt)

if (cudnn_include AND cudnn)
   include_directories(${cudnn_include})
   cuda_add_library(cudnn_test STATIC ../../dnn/cudnn_dlibapi.cpp ${cudnn} )
   enable_cpp11_for_target(cudnn_test)
endif()
