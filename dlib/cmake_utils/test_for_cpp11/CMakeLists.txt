
cmake_minimum_required(VERSION 2.8.12)
project(cpp11_test)

# Try to enable C++11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   message(STATUS "C++11 activated.")
   ADD_DEFINITIONS("-std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
   message(STATUS "C++0x activated.")
   ADD_DEFINITIONS("-std=c++0x")
endif()

add_library(cpp11_test STATIC cpp11_test.cpp )
