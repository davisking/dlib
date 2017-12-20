
cmake_minimum_required(VERSION 2.8.12)

if (POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

set(USING_OLD_VISUAL_STUDIO_COMPILER 0)
if(MSVC AND MSVC_VERSION VERSION_LESS 1900)
   message(FATAL_ERROR "C++11 is required to use dlib, but the version of Visual Studio you are using is too old and doesn't support C++11.  You need Visual Studio 2015 or newer. ")
elseif(MSVC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.0.24210.0 ) 
   message(STATUS "NOTE: Visual Studio didn't have good enough C++11 support until Visual Studio 2015 update 3 (v19.0.24210.0)")
   message(STATUS "So we aren't enabling things that require full C++11 support (e.g. the deep learning tools).")
   message(STATUS "Also, be aware that Visual Studio's version naming is confusing, in particular, there are multiple versions of 'update 3'")
   message(STATUS "So if you are getting this message you need to update to the newer version of Visual Studio to use full C++11.")
   set(USING_OLD_VISUAL_STUDIO_COMPILER 1)
elseif(MSVC AND (MSVC_VERSION EQUAL 1911 OR MSVC_VERSION EQUAL 1910))
   message(STATUS "******************************************************************************************")
   message(STATUS "Your version of Visual Studio has incomplete C++11 support and is unable to compile the ")
   message(STATUS "DNN examples. So we are disabling the deep learning tools.  If you want to use the DNN ")
   message(STATUS "tools in dlib then update your copy of Visual Studio.")
   message(STATUS "******************************************************************************************")
   set(USING_OLD_VISUAL_STUDIO_COMPILER 1)
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
   execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
   if (GCC_VERSION VERSION_LESS 4.8)
      message(FATAL_ERROR "C++11 is required to use dlib, but the version of GCC you are using is too old and doesn't support C++11.  You need GCC 4.8 or newer. ")
   endif()
endif()


# push USING_OLD_VISUAL_STUDIO_COMPILER to the parent so we can use it in the
# examples CMakeLists.txt file.
get_directory_property(has_parent PARENT_DIRECTORY)
if(has_parent)
   set(USING_OLD_VISUAL_STUDIO_COMPILER ${USING_OLD_VISUAL_STUDIO_COMPILER} PARENT_SCOPE)
endif()



set(gcc_like_compilers GNU Clang  Intel)
set(intel_archs x86_64 i386 i686 AMD64 amd64 x86)


# Setup some options to allow a user to enable SSE and AVX instruction use.  
if ((";${gcc_like_compilers};" MATCHES ";${CMAKE_CXX_COMPILER_ID};")  AND
   (";${intel_archs};"        MATCHES ";${CMAKE_SYSTEM_PROCESSOR};") AND NOT USE_AUTO_VECTOR)
   option(USE_SSE2_INSTRUCTIONS "Compile your program with SSE2 instructions" OFF)
   option(USE_SSE4_INSTRUCTIONS "Compile your program with SSE4 instructions" OFF)
   option(USE_AVX_INSTRUCTIONS  "Compile your program with AVX instructions"  OFF)
   if(USE_AVX_INSTRUCTIONS)
      list(APPEND active_compile_opts -mavx)
      message(STATUS "Enabling AVX instructions")
   elseif (USE_SSE4_INSTRUCTIONS)
      list(APPEND active_compile_opts -msse4)
      message(STATUS "Enabling SSE4 instructions")
   elseif(USE_SSE2_INSTRUCTIONS)
      list(APPEND active_compile_opts -msse2)
      message(STATUS "Enabling SSE2 instructions")
   endif()
elseif (MSVC OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC") # else if using Visual Studio 
   # Use SSE2 by default when using Visual Studio.
   option(USE_SSE2_INSTRUCTIONS "Compile your program with SSE2 instructions" ON)
   option(USE_SSE4_INSTRUCTIONS "Compile your program with SSE4 instructions" OFF)
   option(USE_AVX_INSTRUCTIONS  "Compile your program with AVX instructions"  OFF)

   include(CheckTypeSize)
   check_type_size( "void*" SIZE_OF_VOID_PTR)
   if(USE_AVX_INSTRUCTIONS)
      list(APPEND active_compile_opts /arch:AVX)
      message(STATUS "Enabling AVX instructions")
   elseif (USE_SSE4_INSTRUCTIONS)
      # Visual studio doesn't have an /arch:SSE2 flag when building in 64 bit modes.
      # So only give it when we are doing a 32 bit build.
      if (SIZE_OF_VOID_PTR EQUAL 4)
         list(APPEND active_compile_opts /arch:SSE2)
      endif()
      message(STATUS "Enabling SSE4 instructions")
      list(APPEND active_preprocessor_switches "-DDLIB_HAVE_SSE2")
      list(APPEND active_preprocessor_switches "-DDLIB_HAVE_SSE3")
      list(APPEND active_preprocessor_switches "-DDLIB_HAVE_SSE41")
   elseif(USE_SSE2_INSTRUCTIONS)
      # Visual studio doesn't have an /arch:SSE2 flag when building in 64 bit modes.
      # So only give it when we are doing a 32 bit build.
      if (SIZE_OF_VOID_PTR EQUAL 4)
         list(APPEND active_compile_opts /arch:SSE2)
      endif()
      message(STATUS "Enabling SSE2 instructions")
      list(APPEND active_preprocessor_switches "-DDLIB_HAVE_SSE2")
   endif()

elseif((";${gcc_like_compilers};" MATCHES ";${CMAKE_CXX_COMPILER_ID};")  AND
        ("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "^arm"))
   option(USE_NEON_INSTRUCTIONS "Compile your program with ARM-NEON instructions" OFF)
   if(USE_NEON_INSTRUCTIONS)
      list(APPEND active_compile_opts -mfpu=neon)
      message(STATUS "Enabling ARM-NEON instructions")
   endif()
endif()




if (CMAKE_COMPILER_IS_GNUCXX)
   # By default, g++ won't warn or error if you forget to return a value in a
   # function which requires you to do so.  This option makes it give a warning
   # for doing this.
   list(APPEND active_compile_opts "-Wreturn-type")
endif()

if ("Clang" MATCHES ${CMAKE_CXX_COMPILER_ID})
   # Increase clang's default tempalte recurision depth so the dnn examples don't error out.
   list(APPEND active_compile_opts "-ftemplate-depth=500")
endif()

if (MSVC)
   # By default Visual Studio does not support .obj files with more than 65k sections.
   # However, code generated by file_to_code_ex and code using DNN module can have
   # them.  So this flag enables > 65k sections, but produces .obj files
   # that will not be readable by VS 2005.
   list(APPEND active_compile_opts "/bigobj")

   if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 3.3) 
      # Clang can compile all Dlib's code at Windows platform. Tested with Clang 5
      list(APPEND active_compile_opts "-Xclang -fcxx-exceptions")
   endif()
endif()


