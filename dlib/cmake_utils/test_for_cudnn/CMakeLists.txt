
cmake_minimum_required(VERSION 3.10.0)
project(cudnn_test)

# Override the FindCUDA.cmake setting to avoid duplication of host flags if using a toolchain:
option(CUDA_PROPAGATE_HOST_FLAGS "Propage C/CXX_FLAGS and friends to the host compiler via -Xcompile" OFF)
find_package(CUDA 7.5 REQUIRED)
set(CUDA_HOST_COMPILATION_CPP ON)
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_50;-std=c++14;-D__STRICT_ANSI__")
add_definitions(-DDLIB_USE_CUDA)

include(find_cudnn.txt)

if (cudnn_include AND cudnn)
   include_directories(${cudnn_include})
   cuda_add_library(cudnn_test STATIC ../../cuda/cudnn_dlibapi.cpp ${cudnn} )
   target_compile_features(cudnn_test PUBLIC cxx_std_14)
endif()
