# This script checks if your compiler and host processor can generate and then run programs with RDRND instructions.

cmake_minimum_required(VERSION 2.8.12)

# Don't rerun this script if its already been executed.
if (DEFINED RDRND_IS_AVAILABLE_ON_HOST)
   return()
endif()

# Set to false unless we find out otherwise in the code below.
set(RDRND_IS_AVAILABLE_ON_HOST 0)

try_compile(test_for_rdrnd_worked ${PROJECT_BINARY_DIR}/rdrnd_test_build ${CMAKE_CURRENT_LIST_DIR}/test_for_rdrnd 
	rdrnd_test)

if(test_for_rdrnd_worked)
	message (STATUS "RDRND instructions can be executed by the host processor.")
	set(RDRND_IS_AVAILABLE_ON_HOST 1)
endif()
