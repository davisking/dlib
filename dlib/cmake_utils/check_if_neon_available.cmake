# This script checks if __ARM_NEON__ is defined for your compiler

cmake_minimum_required(VERSION 3.8.0)

# Don't rerun this script if its already been executed.
if (DEFINED ARM_NEON_IS_AVAILABLE)
   return()
endif()

# Set to false unless we find out otherwise in the code below.
set(ARM_NEON_IS_AVAILABLE 0)

# test if __ARM_NEON__ is defined
try_compile(test_for_neon_worked ${PROJECT_BINARY_DIR}/neon_test_build ${CMAKE_CURRENT_LIST_DIR}/test_for_neon 
   neon_test)

if(test_for_neon_worked)
   message (STATUS "__ARM_NEON__ defined.")
   set(ARM_NEON_IS_AVAILABLE 1)
endif()
