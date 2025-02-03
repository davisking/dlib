
# Including this cmake script into your cmake project will cause visual studio
# to build your project against the static C runtime.

cmake_minimum_required(VERSION 3.10.0)

if (MSVC OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
   option (DLIB_FORCE_MSVC_STATIC_RUNTIME "use static runtime" ON)
   if (DLIB_FORCE_MSVC_STATIC_RUNTIME)
      foreach(flag_var
         CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
         CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
         if(${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
         endif()
      endforeach(flag_var)
   endif ()
endif()

