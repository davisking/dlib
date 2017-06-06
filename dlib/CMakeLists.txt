#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#


cmake_minimum_required(VERSION 2.8.12)
project(dlib)

# Adhere to GNU filesystem layout conventions
include(GNUInstallDirs)

# default to a Release build (except if CMAKE_BUILD_TYPE is set)
include(cmake_utils/release_build_by_default)
include(cmake_utils/use_cpp_11.cmake)


set(CPACK_PACKAGE_VERSION_MAJOR "19")
set(CPACK_PACKAGE_VERSION_MINOR "4")
set(CPACK_PACKAGE_VERSION_PATCH "99")
set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})
# Set DLIB_VERSION in the including CMake file so they can use it to do whatever they want. 
get_directory_property(has_parent PARENT_DIRECTORY)
if(has_parent)
   set(DLIB_VERSION ${VERSION} PARENT_SCOPE)
endif()

# Set only because there are old target_link_libraries() statements in the
# FindCUDA.cmake file that comes with CMake that error out if the new behavior
# is used.
cmake_policy(SET CMP0023 OLD)


macro (enable_preprocessor_switch option_name)
   list(APPEND active_preprocessor_switches "-D${option_name}")
endmacro()

macro (disable_preprocessor_switch option_name)
   if (active_preprocessor_switches)
      list(REMOVE_ITEM active_preprocessor_switches "-D${option_name}")
   endif()
endmacro()

macro (toggle_preprocessor_switch option_name)
   if (${option_name})
      enable_preprocessor_switch(${option_name})
   else()
      disable_preprocessor_switch(${option_name})
   endif()
endmacro()



# Suppress superfluous randlib warnings about libdlib.a having no symbols on MacOSX.
if (APPLE)
    set(CMAKE_C_ARCHIVE_CREATE   "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> Scr <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_C_ARCHIVE_FINISH   "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")
    set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> -no_warning_for_no_symbols -c <TARGET>")
endif()

# Don't try to call add_library(dlib) and setup dlib's stuff if it has already
# been done by some other part of the current cmake project.  We do this
# because it avoids getting warnings/errors about cmake policy CMP0002.  This
# happens when a project tries to call add_subdirectory() on dlib more than
# once.  This most often happens when the top level of a project depends on two
# or more other things which both depend on dlib. 
if (NOT TARGET dlib)

   set (DLIB_ISO_CPP_ONLY_STR 
   "Enable this if you don't want to compile any non-ISO C++ code (i.e. you don't use any of the API Wrappers)" )
   set (DLIB_NO_GUI_SUPPORT_STR 
   "Enable this if you don't want to compile any of the dlib GUI code" )
   set (DLIB_ENABLE_STACK_TRACE_STR 
   "Enable this if you want to turn on the DLIB_STACK_TRACE macros" )
   set (DLIB_USE_BLAS_STR
   "Disable this if you don't want to use a BLAS library" )
   set (DLIB_USE_LAPACK_STR
   "Disable this if you don't want to use a LAPACK library" )
   set (DLIB_USE_CUDA_STR
   "Disable this if you don't want to use NVIDIA CUDA" )
   set (DLIB_PNG_SUPPORT_STR
   "Disable this if you don't want to link against libpng" )
   set (DLIB_GIF_SUPPORT_STR
   "Disable this if you don't want to link against libgif" )
   set (DLIB_JPEG_SUPPORT_STR
   "Disable this if you don't want to link against libjpeg" )
   set (DLIB_LINK_WITH_SQLITE3_STR
   "Disable this if you don't want to link against sqlite3" )
   #set (DLIB_USE_FFTW_STR "Disable this if you don't want to link against fftw" )
   set (DLIB_USE_MKL_FFT_STR
   "Disable this is you don't want to use the MKL DFTI FFT implementation" )
   set (DLIB_ENABLE_ASSERTS_STR
   "Enable this if you want to turn on the DLIB_ASSERT macro" )


   option(DLIB_ENABLE_ASSERTS ${DLIB_ENABLE_ASSERTS_STR} OFF)
   option(DLIB_ISO_CPP_ONLY ${DLIB_ISO_CPP_ONLY_STR} OFF)
   toggle_preprocessor_switch(DLIB_ISO_CPP_ONLY)
   option(DLIB_NO_GUI_SUPPORT ${DLIB_NO_GUI_SUPPORT_STR} OFF)
   toggle_preprocessor_switch(DLIB_NO_GUI_SUPPORT)
   option(DLIB_ENABLE_STACK_TRACE ${DLIB_ENABLE_STACK_TRACE_STR} OFF)
   toggle_preprocessor_switch(DLIB_ENABLE_STACK_TRACE)

   if(DLIB_ENABLE_ASSERTS)
      # Set these variables so they are set in the config.h.in file when dlib
      # is installed.
      set (DLIB_DISABLE_ASSERTS false)
      set (ENABLE_ASSERTS true) 
      enable_preprocessor_switch(ENABLE_ASSERTS)
      disable_preprocessor_switch(DLIB_DISABLE_ASSERTS)
   else()
      # Set these variables so they are set in the config.h.in file when dlib
      # is installed.
      set (DLIB_DISABLE_ASSERTS true)
      set (ENABLE_ASSERTS false) 
      disable_preprocessor_switch(ENABLE_ASSERTS)
      # Never force the asserts off when doing an in project build.  The only
      # time this matters is when using visual studio.  The visual studio IDE
      # has a drop down that lets the user select either release or debug
      # builds.  The DLIB_ASSERT macro is setup to enable/disable automatically
      # based on this drop down (via preprocessor magic).  However, if
      # DLIB_DISABLE_ASSERTS is defined it permanently disables asserts no
      # matter what, which would defeat the visual studio drop down.  So here
      # we make a point to not do that kind of severe disabling when in a
      # project build.  It should also be pointed out that DLIB_DISABLE_ASSERTS
      # is only needed when building and installing dlib as a separately
      # installed library.  It doesn't matter when doing an in project build. 
      if (NOT DLIB_IN_PROJECT_BUILD)
         enable_preprocessor_switch(DLIB_DISABLE_ASSERTS)
      endif()
   endif()

   if (DLIB_ISO_CPP_ONLY)
      option(DLIB_JPEG_SUPPORT ${DLIB_JPEG_SUPPORT_STR} OFF)
      option(DLIB_LINK_WITH_SQLITE3 ${DLIB_LINK_WITH_SQLITE3_STR} OFF)
      option(DLIB_USE_BLAS ${DLIB_USE_BLAS_STR} OFF)
      option(DLIB_USE_LAPACK ${DLIB_USE_LAPACK_STR} OFF)
      option(DLIB_USE_CUDA ${DLIB_USE_CUDA_STR} OFF)
      option(DLIB_PNG_SUPPORT ${DLIB_PNG_SUPPORT_STR} OFF)
      option(DLIB_GIF_SUPPORT ${DLIB_GIF_SUPPORT_STR} OFF)
      #option(DLIB_USE_FFTW ${DLIB_USE_FFTW_STR} OFF)
      option(DLIB_USE_MKL_FFT ${DLIB_USE_MKL_FFT_STR} OFF)
   else()
      option(DLIB_JPEG_SUPPORT ${DLIB_JPEG_SUPPORT_STR} ON)
      option(DLIB_LINK_WITH_SQLITE3 ${DLIB_LINK_WITH_SQLITE3_STR} ON)
      option(DLIB_USE_BLAS ${DLIB_USE_BLAS_STR} ON)
      option(DLIB_USE_LAPACK ${DLIB_USE_LAPACK_STR} ON)
      option(DLIB_USE_CUDA ${DLIB_USE_CUDA_STR} ON)
      option(DLIB_PNG_SUPPORT ${DLIB_PNG_SUPPORT_STR} ON)
      option(DLIB_GIF_SUPPORT ${DLIB_GIF_SUPPORT_STR} ON)
      #option(DLIB_USE_FFTW ${DLIB_USE_FFTW_STR} ON)
      option(DLIB_USE_MKL_FFT ${DLIB_USE_MKL_FFT_STR} ON)
   endif()
   toggle_preprocessor_switch(DLIB_JPEG_SUPPORT)
   toggle_preprocessor_switch(DLIB_USE_BLAS)
   toggle_preprocessor_switch(DLIB_USE_LAPACK)
   toggle_preprocessor_switch(DLIB_USE_CUDA)
   toggle_preprocessor_switch(DLIB_PNG_SUPPORT) 
   toggle_preprocessor_switch(DLIB_GIF_SUPPORT) 
   #toggle_preprocessor_switch(DLIB_USE_FFTW)
   toggle_preprocessor_switch(DLIB_USE_MKL_FFT)


   set(source_files 
         base64/base64_kernel_1.cpp
         bigint/bigint_kernel_1.cpp
         bigint/bigint_kernel_2.cpp
         bit_stream/bit_stream_kernel_1.cpp
         entropy_decoder/entropy_decoder_kernel_1.cpp
         entropy_decoder/entropy_decoder_kernel_2.cpp
         entropy_encoder/entropy_encoder_kernel_1.cpp
         entropy_encoder/entropy_encoder_kernel_2.cpp
         md5/md5_kernel_1.cpp
         tokenizer/tokenizer_kernel_1.cpp
         unicode/unicode.cpp
         data_io/image_dataset_metadata.cpp
         data_io/mnist.cpp
         dnn/cpu_dlib.cpp
         dnn/tensor_tools.cpp
   )


   set(dlib_needed_libraries)
   set(dlib_needed_includes)

   if (DLIB_ISO_CPP_ONLY)
       add_library(dlib STATIC ${source_files} )
       if (UNIX AND NOT DLIB_IN_PROJECT_BUILD)
           add_library(dlib_shared SHARED ${source_files} )
           add_dependencies(dlib_shared dlib)
       endif()
   else()

      set(source_files ${source_files}
         sockets/sockets_kernel_1.cpp
         bsp/bsp.cpp
         dir_nav/dir_nav_kernel_1.cpp
         dir_nav/dir_nav_kernel_2.cpp
         dir_nav/dir_nav_extensions.cpp
         linker/linker_kernel_1.cpp
         logger/extra_logger_headers.cpp
         logger/logger_kernel_1.cpp
         logger/logger_config_file.cpp
         misc_api/misc_api_kernel_1.cpp
         misc_api/misc_api_kernel_2.cpp
         sockets/sockets_extensions.cpp
         sockets/sockets_kernel_2.cpp
         sockstreambuf/sockstreambuf.cpp
         sockstreambuf/sockstreambuf_unbuffered.cpp
         server/server_kernel.cpp
         server/server_iostream.cpp
         server/server_http.cpp
         threads/multithreaded_object_extension.cpp
         threads/threaded_object_extension.cpp
         threads/threads_kernel_1.cpp
         threads/threads_kernel_2.cpp
         threads/threads_kernel_shared.cpp
         threads/thread_pool_extension.cpp
         threads/async.cpp
         timer/timer.cpp
         stack_trace.cpp
         )

      if(UNIX)
         set(CMAKE_THREAD_PREFER_PTHREAD ON)
         find_package(Threads REQUIRED)
         set(dlib_needed_libraries ${dlib_needed_libraries} ${CMAKE_THREAD_LIBS_INIT})
      endif()

      # we want to link to the right stuff depending on our platform.  
      if (WIN32 AND NOT CYGWIN) ###############################################################################
         if (DLIB_NO_GUI_SUPPORT)
            set (dlib_needed_libraries ws2_32 winmm)
         else()
            set (dlib_needed_libraries ws2_32 winmm comctl32 gdi32 imm32)
         endif()
      elseif(APPLE) ############################################################################
         set(CMAKE_MACOSX_RPATH 1)
         if (NOT DLIB_NO_GUI_SUPPORT)
            find_package(X11 QUIET)
            if (X11_FOUND)
               # If both X11 and anaconda are installed, it's possible for the
               # anaconda path to appear before /opt/X11, so we remove anaconda.
               foreach (ITR ${X11_INCLUDE_DIR})
                  if ("${ITR}" MATCHES "(.*)(ana|mini)conda(.*)")
                     list (REMOVE_ITEM X11_INCLUDE_DIR ${ITR})
                  endif ()
               endforeach(ITR)
               include_directories(${X11_INCLUDE_DIR})
               set (dlib_needed_libraries ${dlib_needed_libraries} ${X11_LIBRARIES})
            else()
               find_library(xlib X11)
               # Make sure X11 is in the include path.  Note that we look for
               # Xlocale.h rather than Xlib.h because it avoids finding a partial
               # copy of the X11 headers on systems with anaconda installed.
               find_path(xlib_path Xlocale.h
                  PATHS 
                  /Developer/SDKs/MacOSX10.4u.sdk/usr/X11R6/include
                  /opt/local/include
                  PATH_SUFFIXES X11
                  )
               if (xlib AND xlib_path)
                  get_filename_component(x11_path ${xlib_path} PATH CACHE)
                  include_directories(${x11_path})
                  set(dlib_needed_libraries ${dlib_needed_libraries} ${xlib} )
                  set(X11_FOUND 1)
               endif()
            endif()
            if (NOT X11_FOUND)
               message(" *****************************************************************************")
               message(" *** DLIB GUI SUPPORT DISABLED BECAUSE X11 DEVELOPMENT LIBRARIES NOT FOUND ***")
               message(" *** Make sure XQuartz is installed if you want GUI support.               ***")
               message(" *** You can download XQuartz from: http://xquartz.macosforge.org/landing/ ***")
               message(" *****************************************************************************")
               set(DLIB_NO_GUI_SUPPORT ON CACHE STRING ${DLIB_NO_GUI_SUPPORT_STR} FORCE )
               enable_preprocessor_switch(DLIB_NO_GUI_SUPPORT)
            endif()
         endif()

         mark_as_advanced(pthreadlib xlib xlib_path x11_path)
      else () ##################################################################################
         # link to the nsl library if it exists.  this is something you need sometimes 
         find_library(nsllib nsl)
         if (nsllib)
            set (dlib_needed_libraries ${dlib_needed_libraries} ${nsllib})
         endif ()

         # link to the socket library if it exists.  this is something you need on solaris
         find_library(socketlib socket)
         if (socketlib)
            set (dlib_needed_libraries ${dlib_needed_libraries} ${socketlib})
         endif ()

         if (NOT DLIB_NO_GUI_SUPPORT)
            include(FindX11)
            if (X11_FOUND)
               include_directories(${X11_INCLUDE_DIR})
               set (dlib_needed_libraries ${dlib_needed_libraries} ${X11_LIBRARIES})
            else()
               message(" *****************************************************************************")
               message(" *** DLIB GUI SUPPORT DISABLED BECAUSE X11 DEVELOPMENT LIBRARIES NOT FOUND ***")
               message(" *** Make sure libx11-dev is installed if you want GUI support.            ***")
               message(" *** On Ubuntu run: sudo apt-get install libx11-dev                        ***")
               message(" *****************************************************************************")
               set(DLIB_NO_GUI_SUPPORT ON CACHE STRING ${DLIB_NO_GUI_SUPPORT_STR} FORCE )
               enable_preprocessor_switch(DLIB_NO_GUI_SUPPORT)
            endif()
         endif()

         mark_as_advanced(nsllib pthreadlib socketlib)
      endif () ##################################################################################

      if (NOT DLIB_NO_GUI_SUPPORT)
         set(source_files ${source_files}
            gui_widgets/fonts.cpp
            gui_widgets/widgets.cpp
            gui_widgets/drawable.cpp
            gui_widgets/canvas_drawing.cpp
            gui_widgets/style.cpp
            gui_widgets/base_widgets.cpp
            gui_core/gui_core_kernel_1.cpp
            gui_core/gui_core_kernel_2.cpp
            )
      endif()

      INCLUDE (CheckFunctionExists)

      if (DLIB_GIF_SUPPORT)
         find_package(GIF QUIET)
         if (GIF_FOUND)
            set (dlib_needed_includes ${dlib_needed_includes} ${GIF_INCLUDE_DIR})
            set (dlib_needed_libraries ${dlib_needed_libraries} ${GIF_LIBRARY})
         else()
            set(DLIB_GIF_SUPPORT OFF CACHE STRING ${DLIB_GIF_SUPPORT_STR} FORCE )
            toggle_preprocessor_switch(DLIB_GIF_SUPPORT)
         endif()
      endif()

      if (DLIB_PNG_SUPPORT)
         # try to find libpng 
         find_package(PNG QUIET)
         # Make sure there isn't something wrong with the version of LIBPNG
         # installed on this system.  
         if (PNG_FOUND)
            set(CMAKE_REQUIRED_LIBRARIES ${PNG_LIBRARY})
            CHECK_FUNCTION_EXISTS(png_create_read_struct LIBPNG_IS_GOOD)
         endif()
         if (PNG_FOUND AND LIBPNG_IS_GOOD)
            include_directories(${PNG_INCLUDE_DIR})
            set (dlib_needed_libraries ${dlib_needed_libraries} ${PNG_LIBRARY})
            set(REQUIRES_LIBS " libpng")
         else()
            # If we can't find libpng then statically compile it in.
            include_directories(external/libpng external/zlib)
            set(source_files ${source_files}
               external/libpng/png.c
               external/libpng/pngerror.c
               external/libpng/pngget.c
               external/libpng/pngmem.c
               external/libpng/pngpread.c
               external/libpng/pngread.c
               external/libpng/pngrio.c
               external/libpng/pngrtran.c
               external/libpng/pngrutil.c
               external/libpng/pngset.c
               external/libpng/pngtrans.c
               external/libpng/pngwio.c
               external/libpng/pngwrite.c
               external/libpng/pngwtran.c
               external/libpng/pngwutil.c
               external/zlib/adler32.c
               external/zlib/compress.c
               external/zlib/crc32.c
               external/zlib/deflate.c
               external/zlib/gzclose.c
               external/zlib/gzlib.c
               external/zlib/gzread.c
               external/zlib/gzwrite.c
               external/zlib/infback.c
               external/zlib/inffast.c
               external/zlib/inflate.c
               external/zlib/inftrees.c
               external/zlib/trees.c
               external/zlib/uncompr.c
               external/zlib/zutil.c
               )
            if (NEON)
               enable_language(ASM)
               set(source_files ${source_files}
                  external/libpng/arm/arm_init.c
                  external/libpng/arm/filter_neon_intrinsics.c
                  external/libpng/arm/filter_neon.S
                  )
               set_source_files_properties(external/libpng/arm/filter_neon.S PROPERTIES COMPILE_FLAGS "${CMAKE_ASM_FLAGS} ${CMAKE_CXX_FLAGS} -x assembler-with-cpp")
            endif()
            set(REQUIRES_LIBS "")
         endif()
         set(source_files ${source_files}
            image_loader/png_loader.cpp
            image_saver/save_png.cpp
            )
      endif()

      if (DLIB_JPEG_SUPPORT)
         # try to find libjpeg 
         find_package(JPEG QUIET)
         # Make sure there isn't something wrong with the version of libjpeg 
         # installed on this system.  Also don't use the installed libjpeg
         # if this is an APPLE system because apparently it's broken (as of 2015/01/01).
         if (JPEG_FOUND)
            set(CMAKE_REQUIRED_LIBRARIES ${JPEG_LIBRARY})
            CHECK_FUNCTION_EXISTS(jpeg_read_header LIBJPEG_IS_GOOD)
         endif()
         if (JPEG_FOUND AND LIBJPEG_IS_GOOD AND NOT APPLE)
            include_directories(${JPEG_INCLUDE_DIR})
            set (dlib_needed_libraries ${dlib_needed_libraries} ${JPEG_LIBRARY})
         else()
            # If we can't find libjpeg then statically compile it in.
            add_definitions(-DDLIB_JPEG_STATIC)
            set(source_files ${source_files}
                  external/libjpeg/jcomapi.cpp
                  external/libjpeg/jdapimin.cpp
                  external/libjpeg/jdapistd.cpp
                  external/libjpeg/jdatasrc.cpp
                  external/libjpeg/jdcoefct.cpp
                  external/libjpeg/jdcolor.cpp
                  external/libjpeg/jddctmgr.cpp
                  external/libjpeg/jdhuff.cpp
                  external/libjpeg/jdinput.cpp
                  external/libjpeg/jdmainct.cpp
                  external/libjpeg/jdmarker.cpp
                  external/libjpeg/jdmaster.cpp
                  external/libjpeg/jdmerge.cpp
                  external/libjpeg/jdphuff.cpp
                  external/libjpeg/jdpostct.cpp
                  external/libjpeg/jdsample.cpp
                  external/libjpeg/jerror.cpp
                  external/libjpeg/jidctflt.cpp
                  external/libjpeg/jidctfst.cpp
                  external/libjpeg/jidctint.cpp
                  external/libjpeg/jidctred.cpp
                  external/libjpeg/jmemmgr.cpp
                  external/libjpeg/jmemnobs.cpp
                  external/libjpeg/jquant1.cpp
                  external/libjpeg/jquant2.cpp
                  external/libjpeg/jutils.cpp  
                  external/libjpeg/jcapimin.cpp
                  external/libjpeg/jdatadst.cpp
                  external/libjpeg/jcparam.cpp
                  external/libjpeg/jcapistd.cpp
                  external/libjpeg/jcmarker.cpp
                  external/libjpeg/jcinit.cpp
                  external/libjpeg/jcmaster.cpp
                  external/libjpeg/jcdctmgr.cpp
                  external/libjpeg/jccoefct.cpp  
                  external/libjpeg/jccolor.cpp  
                  external/libjpeg/jchuff.cpp  
                  external/libjpeg/jcmainct.cpp  
                  external/libjpeg/jcphuff.cpp  
                  external/libjpeg/jcprepct.cpp  
                  external/libjpeg/jcsample.cpp
                  external/libjpeg/jfdctint.cpp
                  external/libjpeg/jfdctflt.cpp
                  external/libjpeg/jfdctfst.cpp
                  )
         endif()
         set(source_files ${source_files}
            image_loader/jpeg_loader.cpp
            image_saver/save_jpeg.cpp
            )
      endif()


      if (DLIB_USE_BLAS OR DLIB_USE_LAPACK OR DLIB_USE_MKL_FFT)
          # Try to find BLAS, LAPACK and MKL
         include(cmake_utils/cmake_find_blas.txt)

         if (DLIB_USE_BLAS)
            if (blas_found)
               set (dlib_needed_libraries ${dlib_needed_libraries} ${blas_libraries})
            else()
               set(DLIB_USE_BLAS OFF CACHE STRING ${DLIB_USE_BLAS_STR} FORCE )
               toggle_preprocessor_switch(DLIB_USE_BLAS)
            endif()
         endif()

         if (DLIB_USE_LAPACK)
            if (lapack_found)
               set (dlib_needed_libraries ${dlib_needed_libraries} ${lapack_libraries})
            else()
               set(DLIB_USE_LAPACK OFF CACHE STRING ${DLIB_USE_LAPACK_STR} FORCE )
               toggle_preprocessor_switch(DLIB_USE_LAPACK)
            endif()
         endif()

         if (DLIB_USE_MKL_FFT)
            if (found_intel_mkl AND found_intel_mkl_headers)
               set (dlib_needed_includes ${dlib_needed_includes} ${mkl_include_dir})
               set (dlib_needed_libraries ${dlib_needed_libraries} ${mkl_libraries})
            else()
               set(DLIB_USE_MKL_FFT OFF CACHE STRING ${DLIB_USE_MKL_FFT_STR} FORCE )
               toggle_preprocessor_switch(DLIB_USE_MKL_FFT)
            endif()
         endif()
      endif()


      if (DLIB_USE_CUDA)
         find_package(CUDA 7.5)

         if (CUDA_FOUND AND MSVC AND NOT CUDA_CUBLAS_LIBRARIES AND "${CMAKE_SIZEOF_VOID_P}" EQUAL "4")
            message(WARNING "You have CUDA installed, but we can't use it unless you put visual studio in 64bit mode.")
            set(CUDA_FOUND 0)
         endif()

         if (CUDA_FOUND AND COMPILER_CAN_DO_CPP_11)

            # There is some bug in cmake that causes it to mess up the
            # -std=c++11 option if you let it propagate it to nvcc in some
            # cases.  So instead we disable this and manually include
            # things from CMAKE_CXX_FLAGS in the CUDA_NVCC_FLAGS list below.
            if (APPLE)
               set(CUDA_PROPAGATE_HOST_FLAGS OFF)
               # Grab all the -D flags from CMAKE_CXX_FLAGS so we can pass them
               # to nvcc.
               string(REGEX MATCHALL "-D[^ ]*" FLAGS_FOR_NVCC ${CMAKE_CXX_FLAGS})
            endif()


            set(CUDA_HOST_COMPILATION_CPP ON)
            # Note that we add __STRICT_ANSI__ to avoid freaking out nvcc with gcc specific
            # magic in the standard C++ header files (since nvcc uses gcc headers on
            # linux).
            list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-D__STRICT_ANSI__;-D_MWAITXINTRIN_H_INCLUDED;-D_FORCE_INLINES;${FLAGS_FOR_NVCC}")
            list(APPEND CUDA_NVCC_FLAGS ${active_preprocessor_switches})
            if (NOT MSVC)
               list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
            endif()
            if (CMAKE_POSITION_INDEPENDENT_CODE)
               # sometimes this setting isn't propagated to NVCC, which then causes the
               # compile to fail.  So make sure it's propagated.
               if (NOT MSVC) # Visual studio doesn't have -fPIC so don't do it in that case.
                  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
               endif()
            endif()

            include(cmake_utils/test_for_cudnn/find_cudnn.txt)

            if (cudnn AND NOT DEFINED cuda_test_compile_worked AND NOT DEFINED cudnn_test_compile_worked)
               # make sure cuda is really working by doing a test compile
               message(STATUS "Building a CUDA test project to see if your compiler is compatible with CUDA...")

               set(CUDA_TEST_CMAKE_FLAGS "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
                                         "-DCMAKE_INCLUDE_PATH=${CMAKE_INCLUDE_PATH}"
                                         "-DCMAKE_LIBRARY_PATH=${CMAKE_LIBRARY_PATH}")

               if (NOT MSVC) # see https://github.com/davisking/dlib/issues/363
                  list(APPEND CUDA_TEST_CMAKE_FLAGS "-DCUDA_HOST_COMPILER=${CUDA_HOST_COMPILER}")
               endif()

               try_compile(cuda_test_compile_worked ${PROJECT_BINARY_DIR}/cuda_test_build 
                           ${PROJECT_SOURCE_DIR}/cmake_utils/test_for_cuda cuda_test
                           CMAKE_FLAGS ${CUDA_TEST_CMAKE_FLAGS}
                           )
               if (NOT cuda_test_compile_worked)
                  message(STATUS "*** CUDA was found but your compiler failed to compile a simple CUDA program so dlib isn't going to use CUDA. ***")
               else()
                  message(STATUS "Checking if you have the right version of cuDNN installed.")
                  try_compile(cudnn_test_compile_worked ${PROJECT_BINARY_DIR}/cudnn_test_build 
                              ${PROJECT_SOURCE_DIR}/cmake_utils/test_for_cudnn cudnn_test
                              CMAKE_FLAGS ${CUDA_TEST_CMAKE_FLAGS}
                                          )
                  if (cudnn_test_compile_worked)
                     message(STATUS "Found cuDNN: " ${cudnn})
                  else()
                     message(STATUS "*** Found cuDNN, but it looks like the wrong version so dlib will not use it. ***")
                  endif()
               endif()
            endif()

            # Find where cuSOLVER is since the FindCUDA cmake package doesn't
            # bother to look for it.
            get_filename_component(cuda_blas_path "${CUDA_CUBLAS_LIBRARIES}" DIRECTORY)
            find_library(cusolver cusolver HINTS ${cuda_blas_path})
            mark_as_advanced(cusolver)
            # Also find OpenMP since cuSOLVER needs it.  Importantly, we only
            # look for one to link to if our use of BLAS, specifically the
            # Intel MKL, hasn't already decided what to use.  This is because
            # it makes the MKL bug out if you link to another openmp lib other
            # than Intel's when you use the MKL.
            if (NOT openmp_libarires AND NOT MSVC)
               find_package(OpenMP)
               if (OPENMP_FOUND)
                  set(openmp_libarires ${OpenMP_CXX_FLAGS}) 
               else()
                  message(STATUS "*** Didn't find OpenMP, which is required to use CUDA. ***")
                  set(CUDA_FOUND 0)
               endif()
            endif()
         endif()

         if (CUDA_FOUND AND cudnn AND COMPILER_CAN_DO_CPP_11 AND cuda_test_compile_worked AND cudnn_test_compile_worked AND cudnn_include)
            set(source_files ${source_files} 
               dnn/cuda_dlib.cu 
               dnn/cudnn_dlibapi.cpp
               dnn/cublas_dlibapi.cpp
               dnn/cusolver_dlibapi.cu
               dnn/curand_dlibapi.cpp
               dnn/cuda_data_ptr.cpp
               dnn/gpu_data.cpp
               )
            set(dlib_needed_libraries ${dlib_needed_libraries} 
                                      ${CUDA_CUBLAS_LIBRARIES} 
                                      ${cudnn}
                                      ${CUDA_curand_LIBRARY}
                                      ${cusolver}
                                      ${openmp_libarires}
                                      )
            include_directories(${cudnn_include})
         else()
            set(DLIB_USE_CUDA OFF CACHE STRING ${DLIB_USE_BLAS_STR} FORCE )
            toggle_preprocessor_switch(DLIB_USE_CUDA)
            if (COMPILER_CAN_DO_CPP_11)
               if (NOT cudnn OR NOT cudnn_include OR NOT cudnn_test_compile_worked)
                  message(STATUS "*** cuDNN V5.0 OR GREATER NOT FOUND.  DLIB WILL NOT USE CUDA. ***")
                  message(STATUS "*** If you have cuDNN then set CMAKE_PREFIX_PATH to include cuDNN's folder.")
               endif()
            else ()
               message(STATUS "*** Dlib CUDA support requires C++11 but your compiler doesn't support it. ***")
            endif()
         endif()
      endif()
      

      if (DLIB_LINK_WITH_SQLITE3)
         find_library(sqlite sqlite3)
         # make sure sqlite3.h is in the include path
         find_path(sqlite_path sqlite3.h)
         if (sqlite AND sqlite_path)
            get_filename_component(sqlite_path2 ${sqlite_path} PATH CACHE)
            set(dlib_needed_includes ${dlib_needed_includes} ${sqlite_path2})
            set(dlib_needed_libraries ${dlib_needed_libraries} ${sqlite} )
         else()
            set(DLIB_LINK_WITH_SQLITE3 OFF CACHE STRING ${DLIB_LINK_WITH_SQLITE3_STR} FORCE )
         endif()
         mark_as_advanced(sqlite sqlite_path sqlite_path2)
      endif()



      if (DLIB_USE_FFTW)
         find_library(fftw fftw3)
         # make sure fftw3.h is in the include path
         find_path(fftw_path fftw3.h)
         if (fftw AND fftw_path)
            set(dlib_needed_includes ${dlib_needed_includes} ${fftw_path})
            set(dlib_needed_libraries ${dlib_needed_libraries} ${fftw} )
         else()
            set(DLIB_USE_FFTW OFF CACHE STRING ${DLIB_USE_FFTW_STR} FORCE )
            toggle_preprocessor_switch(DLIB_USE_FFTW)
         endif()
         mark_as_advanced(fftw fftw_path)
      endif()



      # Tell CMake to build dlib via add_library()/cuda_add_library()
      if (DLIB_USE_CUDA)
         # The old cuda_add_library() command doesn't support CMake's newer dependency
         # stuff, so we have to set the include path manually still, which we do here.
         include_directories(${dlib_needed_includes})
         cuda_add_library(dlib STATIC ${source_files} )
      else()
         add_library(dlib STATIC ${source_files} )
      endif()
      if (UNIX AND NOT DLIB_IN_PROJECT_BUILD)
         if (DLIB_USE_CUDA)
            cuda_add_library(dlib_shared SHARED ${source_files} )
            add_dependencies(dlib_shared dlib)
         else()
            add_library(dlib_shared SHARED ${source_files} )
            add_dependencies(dlib_shared dlib)
         endif()
      endif()

   endif ()  ##### end of if NOT DLIB_ISO_CPP_ONLY ##########################################################


   target_include_directories(dlib
                              INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                              INTERFACE $<INSTALL_INTERFACE:include>
                              PUBLIC ${dlib_needed_includes}
                              )
   target_link_libraries(dlib PRIVATE ${dlib_needed_libraries})
   if (DLIB_IN_PROJECT_BUILD)
      target_compile_options(dlib PUBLIC ${active_preprocessor_switches})
   else()
      target_compile_options(dlib PRIVATE ${active_preprocessor_switches})
   endif()
   if (UNIX AND NOT DLIB_IN_PROJECT_BUILD)
      target_include_directories(dlib_shared
                                 INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                                 INTERFACE $<INSTALL_INTERFACE:include>
                                 PUBLIC ${dlib_needed_includes}
                                 )
      target_link_libraries(dlib_shared PRIVATE ${dlib_needed_libraries})
      target_compile_options(dlib_shared PRIVATE ${active_preprocessor_switches})
   endif()


   # Allow the unit tests to ask us to compile the all/source.cpp file just to make sure it compiles.
   if (DLIB_TEST_COMPILE_ALL_SOURCE_CPP)
      ADD_LIBRARY(dlib_all_source_cpp STATIC all/source.cpp) 
      target_link_libraries(dlib_all_source_cpp dlib)
      target_compile_options(dlib_all_source_cpp PUBLIC ${active_preprocessor_switches})
   endif()

   # Install the library
   if (NOT DLIB_IN_PROJECT_BUILD)
       cmake_minimum_required(VERSION 2.8.8)
       if(UNIX)
           set_target_properties(dlib_shared PROPERTIES
                                        OUTPUT_NAME dlib 
                                        VERSION ${VERSION})
           install(TARGETS dlib dlib_shared
                   EXPORT dlib 
                   RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Windows (including cygwin) considers .dll to be runtime artifacts
                   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                   ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
       else()
           install(TARGETS dlib
                   EXPORT dlib 
                   RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Windows considers .dll to be runtime artifacts
                   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                   ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
       endif()

       install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dlib
               FILES_MATCHING PATTERN "*.h" PATTERN "*.cmake"
               REGEX "${CMAKE_CURRENT_BINARY_DIR}" EXCLUDE)


       configure_file(${PROJECT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)
       # overwrite config.h with the configured one
       install(FILES ${CMAKE_CURRENT_BINARY_DIR}/config.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dlib)

       configure_file(${PROJECT_SOURCE_DIR}/revision.h.in ${CMAKE_CURRENT_BINARY_DIR}/revision.h)
       install(FILES ${CMAKE_CURRENT_BINARY_DIR}/revision.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dlib)

       ## Config.cmake generation and installation

       set(ConfigPackageLocation "${CMAKE_INSTALL_LIBDIR}/cmake/dlib")
       install(EXPORT dlib
            NAMESPACE dlib::
            DESTINATION ${ConfigPackageLocation})

       configure_file(cmake_utils/dlibConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/config/dlibConfig.cmake" @ONLY)

       include(CMakePackageConfigHelpers)
       write_basic_package_version_file(
           "${CMAKE_CURRENT_BINARY_DIR}/config/dlibConfigVersion.cmake"
           VERSION ${VERSION}
           COMPATIBILITY AnyNewerVersion
           )

       install(FILES 
                    "${CMAKE_CURRENT_BINARY_DIR}/config/dlibConfig.cmake" 
                    "${CMAKE_CURRENT_BINARY_DIR}/config/dlibConfigVersion.cmake" 
               DESTINATION ${ConfigPackageLocation})

       ## dlib-1.pc generation and installation

       configure_file("cmake_utils/dlib.pc.in" "dlib-1.pc" @ONLY)
       install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dlib-1.pc"
           DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

   endif()

endif()

add_library(dlib::dlib ALIAS dlib)
