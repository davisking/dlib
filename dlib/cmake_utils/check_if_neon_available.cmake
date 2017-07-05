# This script checks if __ARM_NEON__ is defined for your compiler

cmake_minimum_required(VERSION 2.8.12)

# Don't rerun this script if its already been executed.
if (DEFINED COMPILER_ON_ARM_NEON)
   return()
endif()

# Set to false unless we find out otherwise in the code below.
set(COMPILER_ON_ARM_NEON 0)

# test if __ARM_NEON__ is defined
try_compile(test_for_neon_worked ${PROJECT_BINARY_DIR}/neon_test_build ${CMAKE_CURRENT_LIST_DIR}/test_for_neon 
	neon_test)

message ("ARM NEON TEST OK: ${test_for_neon_worked}")

if(test_for_neon_worked)
	message ("__ARM_NEON__ defined.")
	set(COMPILER_ON_ARM_NEON 1)
endif()
