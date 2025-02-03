#This script just runs CMake's built in PNG finding tool.  But it also checks that the
#copy of libpng that cmake finds actually builds and links.

cmake_minimum_required(VERSION 3.10.0)

if (BUILDING_PYTHON_IN_MSVC)
   # Never use any system copy of libpng when building python in visual studio
   set(PNG_FOUND 0)
   return()
endif()

# Don't rerun this script if its already been executed.
if (DEFINED PNG_FOUND)
   return()
endif()

find_package(PNG QUIET)

if(PNG_FOUND)
   set(PNG_TEST_CMAKE_FLAGS 
      "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
      "-DCMAKE_INCLUDE_PATH=${CMAKE_INCLUDE_PATH}"
      "-DCMAKE_LIBRARY_PATH=${CMAKE_LIBRARY_PATH}")

   try_compile(test_for_libpng_worked 
      ${PROJECT_BINARY_DIR}/test_for_libpng_build  
      ${CMAKE_CURRENT_LIST_DIR}/test_for_libpng
      test_if_libpng_is_broken
      CMAKE_FLAGS "${PNG_TEST_CMAKE_FLAGS}")

   message (STATUS "Found system copy of libpng: ${PNG_LIBRARIES}")
   if(NOT test_for_libpng_worked)
      set(PNG_FOUND 0)
      message (STATUS "System copy of libpng is broken.  Will build our own libpng and use that instead.")
   endif()
endif()

