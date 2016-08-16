#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#

cmake_minimum_required(VERSION 2.8.4)

set (target_name test_all_source_builds)
PROJECT(${target_name})
include(../../cmake_utils/use_cpp_11.cmake)
ADD_LIBRARY(${target_name} STATIC ../../all/source.cpp) 


