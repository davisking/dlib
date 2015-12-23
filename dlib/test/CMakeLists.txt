#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#

cmake_minimum_required(VERSION 2.8.4)

# create a variable called target_name and set it to the string "dtest"
set (target_name dtest)
PROJECT(${target_name})

include(../cmake)

# This variable contains a list of all the tests we are building
# into the regression test suite.
set (tests
   )

# Tests that require C++11 support
if (COMPILER_CAN_DO_CPP_11)
   set(tests ${tests}
      dnn.cpp
      cublas.cpp
      )
endif()


# add all the cpp files we want to compile to this list.  This tells
# cmake that they are part of our target (which is the executable named dtest)
ADD_EXECUTABLE(${target_name} main.cpp tester.cpp ${tests})

# Turn on all warnings when using gcc.
if (CMAKE_COMPILER_IS_GNUCXX)
   add_definitions("-W -Wall")
endif()


TARGET_LINK_LIBRARIES(${target_name} dlib )


