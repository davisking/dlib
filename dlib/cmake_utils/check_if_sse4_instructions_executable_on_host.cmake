# This script checks if your compiler and host processor can generate and then run programs with SSE4 instructions.

cmake_minimum_required(VERSION 3.8.0)

# Don't rerun this script if its already been executed.
if (DEFINED SSE4_IS_AVAILABLE_ON_HOST)
   return()
endif()

# Set to false unless we find out otherwise in the code below.
set(SSE4_IS_AVAILABLE_ON_HOST 0)

try_compile(test_for_sse4_worked ${PROJECT_BINARY_DIR}/sse4_test_build ${CMAKE_CURRENT_LIST_DIR}/test_for_sse4
	sse4_test)

if(test_for_sse4_worked)
	message (STATUS "SSE4 instructions can be executed by the host processor.")
	set(SSE4_IS_AVAILABLE_ON_HOST 1)
endif()
