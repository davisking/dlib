# This script checks if your compiler has C++11 support and enables it if it does.
# Also, it sets the COMPILER_CAN_DO_CPP_11 variable to 1 if it was successful.


cmake_minimum_required(VERSION 2.8.4)

# Don't rerun this script if its already been executed.
if (DEFINED COMPILER_CAN_DO_CPP_11)
   return()
endif()

if (POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

# Set to false unless we find out otherwise in the code below.
set(COMPILER_CAN_DO_CPP_11 0)

# Determine the path to dlib.
string(REGEX REPLACE "use_cpp_11.cmake$" "" dlib_path ${CMAKE_CURRENT_LIST_FILE})
include(${dlib_path}/add_global_compiler_switch.cmake)


# Now turn on the appropriate compiler switch to enable C++11 if you have a
# C++11 compiler.  In CMake 3.1 there is a simple flag you can set, but earlier
# verions of CMake are not so convenient.
if (CMAKE_VERSION VERSION_LESS "3.1.2")
   if(CMAKE_COMPILER_IS_GNUCXX)
      execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
      if (GCC_VERSION VERSION_GREATER 4.8 OR GCC_VERSION VERSION_EQUAL 4.8)
         message(STATUS "C++11 activated.")
         add_global_compiler_switch("-std=gnu++11")
         set(COMPILER_CAN_DO_CPP_11 1)
      endif()
   elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      execute_process( COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE clang_full_version_string )
      string (REGEX REPLACE ".*clang version ([0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION ${clang_full_version_string})
      if (CLANG_VERSION VERSION_GREATER 3.3)
         message(STATUS "C++11 activated.")
         add_global_compiler_switch("-std=c++11")
         set(COMPILER_CAN_DO_CPP_11 1)
      endif()
   else()
      # Since we don't know what compiler this is ust try to build a c++11 project and see if it compiles.
      message(STATUS "Building a C++11 test project to see if your compiler supports C++11")
      try_compile(test_for_cpp11_worked ${PROJECT_BINARY_DIR}/cpp11_test_build 
         ${dlib_path}/dnn/test_for_cpp11 cpp11_test)
      if (test_for_cpp11_worked)
         message(STATUS "C++11 activated.")
         set(COMPILER_CAN_DO_CPP_11 1)
      else()
         message(STATUS "*** Your compiler failed to build a C++11 project, so dlib won't use C++11 features.***")
      endif()
   endif()
else()
   # Set a flag if the compiler you are using is capable of providing C++11 features.
   if (";${CMAKE_CXX_COMPILE_FEATURES};" MATCHES ";cxx_rvalue_references;" AND
       ";${CMAKE_CXX_COMPILE_FEATURES};" MATCHES ";cxx_variadic_templates;" AND
       ";${CMAKE_CXX_COMPILE_FEATURES};" MATCHES ";cxx_lambdas;" AND
       ";${CMAKE_CXX_COMPILE_FEATURES};" MATCHES ";cxx_defaulted_move_initializers;" AND
       ";${CMAKE_CXX_COMPILE_FEATURES};" MATCHES ";cxx_delegating_constructors;" AND
       ";${CMAKE_CXX_COMPILE_FEATURES};" MATCHES ";cxx_thread_local;" AND
       ";${CMAKE_CXX_COMPILE_FEATURES};" MATCHES ";cxx_constexpr;" AND
       ";${CMAKE_CXX_COMPILE_FEATURES};" MATCHES ";cxx_decltype_incomplete_return_types;" AND
       ";${CMAKE_CXX_COMPILE_FEATURES};" MATCHES ";cxx_auto_type;")

      set(COMPILER_CAN_DO_CPP_11 1)
      # Set which standard to use unless someone has already set it to something
      # newer.
      if (NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 11)
         set(CMAKE_CXX_STANDARD 11)
         message(STATUS "C++11 activated.")
      endif()
   endif()
endif()

