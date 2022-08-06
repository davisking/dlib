# This script checks if your compiler and host processor can generate and then run programs with AVX instructions.

cmake_minimum_required(VERSION 3.8.0)

# Don't rerun this script if its already been executed.
if (DEFINED AVX_IS_AVAILABLE_ON_HOST)
   return()
endif()

# Set to false unless we find out otherwise in the code below.
set(AVX_IS_AVAILABLE_ON_HOST 0)

try_compile(test_for_avx_worked ${PROJECT_BINARY_DIR}/avx_test_build ${CMAKE_CURRENT_LIST_DIR}/test_for_avx 
	avx_test)

if(test_for_avx_worked)
	message (STATUS "AVX instructions can be executed by the host processor.")
	set(AVX_IS_AVAILABLE_ON_HOST 1)
endif()
