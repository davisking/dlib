# This script checks if your compiler has C++11 support and enables it if it does.
# Also, it sets the COMPILER_CAN_DO_CPP_11 variable to 1 if it was successful.


cmake_minimum_required(VERSION 2.8.12)

# Don't rerun this script if its already been executed.
if (DEFINED COMPILER_CAN_DO_CPP_11)
   return()
endif()

if (POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

# Set to false unless we find out otherwise in the code below.
set(COMPILER_CAN_DO_CPP_11 0)

include(${CMAKE_CURRENT_LIST_DIR}/add_global_compiler_switch.cmake)

if(MSVC AND MSVC_VERSION VERSION_LESS 1900)
   message(FATAL_ERROR "C++11 is required to use dlib, but the version of Visual Studio you are using is too old and doesn't support C++11.  You need Visual Studio 2015 or newer. ")
endif()

macro(test_compiler_for_cpp11)
   message(STATUS "Building a C++11 test project to see if your compiler supports C++11")
   try_compile(test_for_cpp11_worked ${PROJECT_BINARY_DIR}/cpp11_test_build 
      ${CMAKE_CURRENT_LIST_DIR}/test_for_cpp11 cpp11_test)
   if (test_for_cpp11_worked)
      message(STATUS "C++11 activated.")
      set(COMPILER_CAN_DO_CPP_11 1)
   else()
      set(COMPILER_CAN_DO_CPP_11 0)
      message(STATUS "********** Your compiler failed to build a C++11 project.  C++11 is required to use all parts of dlib! **********")
   endif()
endmacro()

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
      # Since we don't know what compiler this is just try to build a c++11 project and see if it compiles.
      test_compiler_for_cpp11()
   endif()
elseif( MSVC AND CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 3.3) 
   # Clang can compile all Dlib's code at Windows platform. Tested with Clang 5
   message(STATUS "C++11 activated.")
   add_global_compiler_switch("-Xclang -fcxx-exceptions")
   set(COMPILER_CAN_DO_CPP_11 1)
elseif(MSVC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.0.24215.1 ) 
   message(STATUS "NOTE: Visual Studio didn't have good enough C++11 support until Visual Studio 2015 update 3 (v19.0.24215.1)")
   message(STATUS "So we aren't enabling things that require full C++11 support (e.g. the deep learning tools).")
   message(STATUS "Also, be aware that Visual Studio's version naming is confusing, in particular, there are multiple versions of 'update 3'")
   message(STATUS "So if you are getting this message you need to update to the newer version of Visual Studio to use full C++11.")
   set(USING_OLD_VISUAL_STUDIO_COMPILER 1)
else()  

   # Set a flag if the compiler you are using is capable of providing C++11 features.
   get_property(cxx_features GLOBAL PROPERTY CMAKE_CXX_KNOWN_FEATURES)
   if (";${cxx_features};" MATCHES ";cxx_rvalue_references;" AND
       ";${cxx_features};" MATCHES ";cxx_variadic_templates;" AND
       ";${cxx_features};" MATCHES ";cxx_lambdas;" AND
       ";${cxx_features};" MATCHES ";cxx_defaulted_move_initializers;" AND
       ";${cxx_features};" MATCHES ";cxx_delegating_constructors;" AND
       ";${cxx_features};" MATCHES ";cxx_thread_local;" AND
       ";${cxx_features};" MATCHES ";cxx_constexpr;" AND
       ";${cxx_features};" MATCHES ";cxx_decltype_incomplete_return_types;" AND
       ";${cxx_features};" MATCHES ";cxx_auto_type;")

      set(COMPILER_CAN_DO_CPP_11 1)
      # Set which standard to use unless someone has already set it to something
      # newer.
      if (NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 11)
         set(CMAKE_CXX_STANDARD 11)
         set(CMAKE_CXX_STANDARD_REQUIRED YES)
         if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
            # Sometimes clang will lie and report that it supports C++11 when
            # really it doesn't support thread_local.  So check for that.
            test_compiler_for_cpp11()
            add_global_compiler_switch("-std=c++11")
         else()
            message(STATUS "C++11 activated.")
         endif()
      endif()
   endif()
endif()

# Always enable whatever partial C++11 support we have, even if it isn't full
# support, and just hope for the best.
if (NOT COMPILER_CAN_DO_CPP_11)
   include(CheckCXXCompilerFlag)
   CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
   CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
   if(COMPILER_SUPPORTS_CXX11)
      message(STATUS "C++11 activated (compiler doesn't have full C++11 support).")
      add_global_compiler_switch("-std=c++11")
   elseif(COMPILER_SUPPORTS_CXX0X)
      message(STATUS "C++0x activated (compiler doesn't have full C++11 support).")
      add_global_compiler_switch("-std=c++0x")
   endif()
endif()

