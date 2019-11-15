#This script just runs CMake's built in JPEG finding tool.  But it also checks that the
#copy of libjpeg that cmake finds actually builds and links.

cmake_minimum_required(VERSION 2.8.12)

if (BUILDING_PYTHON_IN_MSVC)
   # Never use any system copy of libjpeg when building python in visual studio
   set(JPEG_FOUND 0)
   return()
endif()

# Don't rerun this script if its already been executed.
if (DEFINED JPEG_FOUND)
   return()
endif()

find_package(JPEG QUIET)

if(JPEG_FOUND)
   set(JPEG_TEST_CMAKE_FLAGS 
      "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
      "-DCMAKE_INCLUDE_PATH=${CMAKE_INCLUDE_PATH}"
      "-DCMAKE_LIBRARY_PATH=${CMAKE_LIBRARY_PATH}")

   try_compile(test_for_libjpeg_worked 
      ${PROJECT_BINARY_DIR}/test_for_libjpeg_build  
      ${CMAKE_CURRENT_LIST_DIR}/test_for_libjpeg
      test_if_libjpeg_is_broken
      CMAKE_FLAGS "${JPEG_TEST_CMAKE_FLAGS}")

   message (STATUS "Found system copy of libjpeg: ${JPEG_LIBRARY}")
   if(NOT test_for_libjpeg_worked)
      set(JPEG_FOUND 0)
      message (STATUS "System copy of libjpeg is broken or too old.  Will build our own libjpeg and use that instead.")
   endif()
endif()


