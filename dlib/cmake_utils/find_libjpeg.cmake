#This script just runs CMake's built in JPEG finding tool.  But it also checks that the
#copy of libjpeg that cmake finds actually builds and links.

cmake_minimum_required(VERSION 2.8.12)

# Don't rerun this script if its already been executed.
if (DEFINED JPEG_FOUND)
   return()
endif()

find_package(JPEG QUIET)

if(JPEG_FOUND) 
   try_compile(test_for_libjpeg_worked 
      ${PROJECT_BINARY_DIR}/test_for_libjpeg_build  
      ${CMAKE_CURRENT_LIST_DIR}/test_for_libjpeg
      test_if_libjpeg_is_broken)

   message (STATUS "Found system copy of libjpeg: ${JPEG_LIBRARY}")
   if(NOT test_for_libjpeg_worked)
      set(JPEG_FOUND 0)
      message (STATUS "System copy of libjpeg is broken.  Will build our own libjpeg and use that instead.")
   endif()
endif()


