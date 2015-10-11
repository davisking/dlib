# This script checks if your compiler has C++11 support and enables it if it does.
# Also, it sets the COMPILER_CAN_DO_CPP_11 variable to 1 if it was successful.


cmake_minimum_required(VERSION 2.8.4)

# Don't rerun this script if its already been executed.
if (COMPILER_CAN_DO_CPP_11)
   return()
endif()

# Determine the path to dlib.
string(REGEX REPLACE "use_cpp_11.cmake$" "" dlib_path ${CMAKE_CURRENT_LIST_FILE})
include(${dlib_path}/add_global_compiler_switch.cmake)


# Now turn on the appropriate compiler switch to enable C++11 if you have a
# C++11 compiler.  In CMake 3.1 there is a simple flag you can set, but earlier
# verions of CMake are not so convenient.
if (CMAKE_VERSION VERSION_LESS "3.1")
   if(CMAKE_COMPILER_IS_GNUCXX)
      execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
      if (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7)
         message(STATUS "C++11 activated.")
         add_global_compiler_switch("-std=gnu++11")
         set(COMPILER_CAN_DO_CPP_11 1)
      elseif(GCC_VERSION VERSION_GREATER 4.3 OR GCC_VERSION VERSION_EQUAL 4.3)
         message(STATUS "C++0x activated.")
         add_global_compiler_switch("-std=gnu++0x")
         set(COMPILER_CAN_DO_CPP_11 1)
      endif()
   endif()
else()
   # Set a flag if the compiler you are using is capable of providing C++11 features.
   if (";${CMAKE_CXX_COMPILE_FEATURES};" MATCHES ";cxx_rvalue_references;" AND
       ";${CMAKE_CXX_COMPILE_FEATURES};" MATCHES ";cxx_variadic_templates;" AND
       ";${CMAKE_CXX_COMPILE_FEATURES};" MATCHES ";cxx_lambdas;" AND
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

