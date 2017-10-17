# This script creates a function, enable_cpp11_for_target(), which checks if your
# compiler has C++11 support and enables it if it does.


cmake_minimum_required(VERSION 2.8.12)

if (POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()


set(_where_is_cmake_utils_dir ${CMAKE_CURRENT_LIST_DIR})

function(enable_cpp11_for_target target_name)


# Set to false unless we find out otherwise in the code below.
set(COMPILER_CAN_DO_CPP_11 0)



macro(test_compiler_for_cpp11)
   message(STATUS "Building a C++11 test project to see if your compiler supports C++11")
   try_compile(test_for_cpp11_worked ${PROJECT_BINARY_DIR}/cpp11_test_build 
      ${_where_is_cmake_utils_dir}/test_for_cpp11 cpp11_test)
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
         target_compile_options(${target_name} PUBLIC "-std=gnu++11")
         set(COMPILER_CAN_DO_CPP_11 1)
      endif()
   elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      execute_process( COMMAND ${CMAKE_CXX_COMPILER} --version OUTPUT_VARIABLE clang_full_version_string )
      string (REGEX REPLACE ".*clang version ([0-9]+\\.[0-9]+).*" "\\1" CLANG_VERSION ${clang_full_version_string})
      if (CLANG_VERSION VERSION_GREATER 3.3)
         message(STATUS "C++11 activated.")
         target_compile_options(${target_name} PUBLIC "-std=c++11")
         set(COMPILER_CAN_DO_CPP_11 1)
      endif()
   else()
      # Since we don't know what compiler this is just try to build a c++11 project and see if it compiles.
      test_compiler_for_cpp11()
   endif()
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
      # Tell cmake that we need C++11 for dlib
      target_compile_features(${target_name} 
         PUBLIC 
            cxx_rvalue_references
            cxx_variadic_templates
            cxx_lambdas
            cxx_defaulted_move_initializers
            cxx_delegating_constructors
            cxx_thread_local
            cxx_constexpr
            # cxx_decltype_incomplete_return_types  # purposfully commented out because cmake errors out on this when using visual studio and cmake 3.8.0
            cxx_auto_type
         )

      if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
         # Sometimes clang will lie and report that it supports C++11 when
         # really it doesn't support thread_local.  So check for that.
         test_compiler_for_cpp11()
      else()
         message(STATUS "C++11 activated.")
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
      target_compile_options(${target_name} PUBLIC "-std=c++11")
   elseif(COMPILER_SUPPORTS_CXX0X)
      message(STATUS "C++0x activated (compiler doesn't have full C++11 support).")
      target_compile_options(${target_name} PUBLIC "-std=c++0x")
   endif()
endif()

endfunction()

