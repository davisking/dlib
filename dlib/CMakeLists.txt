#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#


cmake_minimum_required(VERSION 2.8.4)
project(dlib)

# default to a Release build (except if CMAKE_BUILD_TYPE is set)
include(release_build_by_default)
include(use_cpp_11.cmake)


set(CPACK_PACKAGE_VERSION_MAJOR "19")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})
# Set DLIB_VERSION in the including CMake file so they can use it to do whatever they want. 
get_directory_property(has_parent PARENT_DIRECTORY)
if(has_parent)
   set(DLIB_VERSION ${VERSION} PARENT_SCOPE)
endif()

set(CMAKE_LEGACY_CYGWIN_WIN32 0) # Remove when CMake >= 2.8.4 is required
# Suppress cmake warnings about changes in new versions.
if(COMMAND cmake_policy) 
   cmake_policy(SET CMP0003 NEW) 
   if (POLICY CMP0054)
      cmake_policy(SET CMP0054 NEW)
   endif()
endif()

include(add_global_compiler_switch.cmake)


# Make sure ENABLE_ASSERTS is defined for debug builds
if (NOT CMAKE_CXX_FLAGS_DEBUG MATCHES "-DENABLE_ASSERTS")
   set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DENABLE_ASSERTS" 
      CACHE STRING "Flags used by the compiler during C++ debug builds." 
      FORCE)
endif ()
macro (toggle_preprocessor_switch option_name)
   if (${option_name})
      add_global_define(${option_name})
   else()
      remove_global_define(${option_name})
   endif()
endmacro()

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
   set (DLIB_ENABLE_ASSERTS_STR 
   "Enable this if you want to turn on the DLIB_ASSERT macro" )
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

   option(DLIB_ISO_CPP_ONLY ${DLIB_ISO_CPP_ONLY_STR} OFF)
   toggle_preprocessor_switch(DLIB_ISO_CPP_ONLY)
   option(DLIB_NO_GUI_SUPPORT ${DLIB_NO_GUI_SUPPORT_STR} OFF)
   toggle_preprocessor_switch(DLIB_NO_GUI_SUPPORT)
   option(DLIB_ENABLE_STACK_TRACE ${DLIB_ENABLE_STACK_TRACE_STR} OFF)
   toggle_preprocessor_switch(DLIB_ENABLE_STACK_TRACE)

   option(DLIB_ENABLE_ASSERTS ${DLIB_ENABLE_ASSERTS_STR} OFF)
   if(DLIB_ENABLE_ASSERTS)
      # Set these variables so they are set in the config.h.in file when dlib
      # is installed.
      set (DLIB_DISABLE_ASSERTS false)
      set (ENABLE_ASSERTS true) 
      add_global_define(ENABLE_ASSERTS)
   else()
      # Set these variables so they are set in the config.h.in file when dlib
      # is installed.
      set (DLIB_DISABLE_ASSERTS true)
      set (ENABLE_ASSERTS false) 
      remove_global_define(ENABLE_ASSERTS)
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
   else()
      option(DLIB_JPEG_SUPPORT ${DLIB_JPEG_SUPPORT_STR} ON)
      option(DLIB_LINK_WITH_SQLITE3 ${DLIB_LINK_WITH_SQLITE3_STR} ON)
      option(DLIB_USE_BLAS ${DLIB_USE_BLAS_STR} ON)
      option(DLIB_USE_LAPACK ${DLIB_USE_LAPACK_STR} ON)
      option(DLIB_USE_CUDA ${DLIB_USE_CUDA_STR} ON)
      option(DLIB_PNG_SUPPORT ${DLIB_PNG_SUPPORT_STR} ON)
      option(DLIB_GIF_SUPPORT ${DLIB_GIF_SUPPORT_STR} ON)
      #option(DLIB_USE_FFTW ${DLIB_USE_FFTW_STR} ON)
   endif()
   toggle_preprocessor_switch(DLIB_JPEG_SUPPORT)
   toggle_preprocessor_switch(DLIB_USE_BLAS)
   toggle_preprocessor_switch(DLIB_USE_LAPACK)
   toggle_preprocessor_switch(DLIB_USE_CUDA)
   toggle_preprocessor_switch(DLIB_PNG_SUPPORT) 
   toggle_preprocessor_switch(DLIB_GIF_SUPPORT) 
   #toggle_preprocessor_switch(DLIB_USE_FFTW)


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
         data_io/mnist.cpp)

   if (COMPILER_CAN_DO_CPP_11)
      set(source_files ${source_files}
         dnn/cpu_dlib.cpp
         dnn/tensor_tools.cpp
         )
   endif()

   if (DLIB_ISO_CPP_ONLY)
       add_library(dlib STATIC ${source_files} )
       if (UNIX AND NOT DLIB_IN_PROJECT_BUILD)
           add_library(dlib_shared SHARED ${source_files} )
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
         timer/timer.cpp
         stack_trace.cpp
         )

      set(dlib_needed_libraries)
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
         if (NOT DLIB_NO_GUI_SUPPORT)
            find_package(X11 QUIET)
            if (X11_FOUND)
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
               add_global_define(DLIB_NO_GUI_SUPPORT)
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
               add_global_define(DLIB_NO_GUI_SUPPORT)
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
            include_directories(${GIF_INCLUDE_DIR})
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


      if (DLIB_USE_BLAS OR DLIB_USE_LAPACK)
         # Try to find BLAS and LAPACK 
         include(cmake_find_blas.txt)

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
      endif()


      if (DLIB_USE_CUDA)
         find_package(CUDA 7.5)


         if (CUDA_FOUND AND COMPILER_CAN_DO_CPP_11)

            set(CUDA_HOST_COMPILATION_CPP ON)
            # Note that we add __STRICT_ANSI__ to avoid freaking out nvcc with gcc specific
            # magic in the standard C++ header files (since nvcc uses gcc headers on
            # linux).
            list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-D__STRICT_ANSI__")

            include(dnn/test_for_cudnn/find_cudnn.txt)

            if (cudnn AND NOT DEFINED cuda_test_compile_worked AND NOT DEFINED cudnn_test_compile_worked)
               # make sure cuda is really working by doing a test compile
               message(STATUS "Building a CUDA test project to see if your compiler is compatible with CUDA...")
               try_compile(cuda_test_compile_worked ${PROJECT_BINARY_DIR}/cuda_test_build 
                           ${PROJECT_SOURCE_DIR}/dnn/test_for_cuda cuda_test
                           CMAKE_FLAGS "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
                                       "-DCMAKE_INCLUDE_PATH=${CMAKE_INCLUDE_PATH}"
                                       "-DCMAKE_LIBRARY_PATH=${CMAKE_LIBRARY_PATH}"
                           )
               if (NOT cuda_test_compile_worked)
                  message(STATUS "*** CUDA was found but your compiler failed to compile a simple CUDA program so dlib isn't going to use CUDA. ***")
               else()
                  message(STATUS "Checking if you have the right version of cuDNN installed.")
                  try_compile(cudnn_test_compile_worked ${PROJECT_BINARY_DIR}/cudnn_test_build 
                              ${PROJECT_SOURCE_DIR}/dnn/test_for_cudnn cudnn_test
                              CMAKE_FLAGS "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"
                                          "-DCMAKE_INCLUDE_PATH=${CMAKE_INCLUDE_PATH}"
                                          "-DCMAKE_LIBRARY_PATH=${CMAKE_LIBRARY_PATH}"
                                          )
                  if (cudnn_test_compile_worked)
                     message(STATUS "Found cuDNN: " ${cudnn})
                  else()
                     message(STATUS "*** Found cuDNN, but it looks like the wrong version so dlib will not use it. ***")
                  endif()
               endif()
            endif()
         endif()


         if (CUDA_FOUND AND cudnn AND cudnn_include AND COMPILER_CAN_DO_CPP_11 AND cuda_test_compile_worked AND cudnn_test_compile_worked)
            set(source_files ${source_files} 
               dnn/cuda_dlib.cu 
               dnn/cudnn_dlibapi.cpp
               dnn/cublas_dlibapi.cpp
               dnn/curand_dlibapi.cpp
               dnn/gpu_data.cpp
               )
            set(dlib_needed_libraries ${dlib_needed_libraries} 
                                      ${CUDA_CUBLAS_LIBRARIES} 
                                      ${cudnn}
                                      ${CUDA_curand_LIBRARY}
                                      )
            include_directories(${cudnn_include})
         else()
            set(DLIB_USE_CUDA OFF CACHE STRING ${DLIB_USE_BLAS_STR} FORCE )
            toggle_preprocessor_switch(DLIB_USE_CUDA)
            if (NOT cudnn OR NOT cudnn_include OR NOT cudnn_test_compile_worked)
               message(STATUS "*** cuDNN V5.0 OR GREATER NOT FOUND.  DLIB WILL NOT USE CUDA. ***")
               message(STATUS "*** If you have cuDNN then set CMAKE_PREFIX_PATH to include cuDNN's folder.")
            endif()
            if (NOT COMPILER_CAN_DO_CPP_11)
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
            include_directories(${sqlite_path2})
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
            include_directories(${fftw_path})
            set(dlib_needed_libraries ${dlib_needed_libraries} ${fftw} )
         else()
            set(DLIB_USE_FFTW OFF CACHE STRING ${DLIB_USE_FFTW_STR} FORCE )
            toggle_preprocessor_switch(DLIB_USE_FFTW)
         endif()
         mark_as_advanced(fftw fftw_path)
      endif()



      # Tell CMake to build dlib via add_library()/cuda_add_library()
      if (DLIB_USE_CUDA)
         cuda_add_library(dlib STATIC ${source_files} )
      else()
         add_library(dlib STATIC ${source_files} )
      endif()
      target_link_libraries(dlib ${dlib_needed_libraries} )
      if (UNIX AND NOT DLIB_IN_PROJECT_BUILD)
         if (DLIB_USE_CUDA)
            cuda_add_library(dlib_shared SHARED ${source_files} )
         else()
            add_library(dlib_shared SHARED ${source_files} )
         endif()
         target_link_libraries(dlib_shared ${dlib_needed_libraries} )
      endif()

   endif ()  ##### end of if NOT DLIB_ISO_CPP_ONLY ##########################################################

   # Install the library
   if (NOT DLIB_IN_PROJECT_BUILD)
       set (LIB_INSTALL_DIR lib CACHE STRING "Install location of libraries (e.g. lib32 or lib64 for multilib installations)")
       cmake_minimum_required(VERSION 2.8.8)
       if(UNIX)
           set_target_properties(dlib_shared PROPERTIES
                                        OUTPUT_NAME dlib 
                                        VERSION ${VERSION})
           install(TARGETS dlib dlib_shared
                   EXPORT dlib 
                   RUNTIME DESTINATION bin # Windows (including cygwin) considers .dll to be runtime artifacts
                   LIBRARY DESTINATION "${LIB_INSTALL_DIR}"
                   ARCHIVE DESTINATION "${LIB_INSTALL_DIR}")
       else()
           install(TARGETS dlib
                   EXPORT dlib 
                   RUNTIME DESTINATION bin # Windows considers .dll to be runtime artifacts
                   LIBRARY DESTINATION lib
                   ARCHIVE DESTINATION lib)
       endif()

       install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION include/dlib
               FILES_MATCHING PATTERN "*.h"
               REGEX "${CMAKE_CURRENT_BINARY_DIR}" EXCLUDE)


       configure_file(${PROJECT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)
       # overwrite config.h with the configured one
       install(FILES ${CMAKE_CURRENT_BINARY_DIR}/config.h DESTINATION include/dlib)

       configure_file(${PROJECT_SOURCE_DIR}/revision.h.in ${CMAKE_CURRENT_BINARY_DIR}/revision.h)
       install(FILES ${CMAKE_CURRENT_BINARY_DIR}/revision.h DESTINATION include/dlib)

       install(FILES "LICENSE.txt" DESTINATION share/doc/dlib)

       ## Config.cmake generation and installation

       set(ConfigPackageLocation "${LIB_INSTALL_DIR}/cmake/dlib")
       install(EXPORT dlib
            NAMESPACE dlib::
            DESTINATION ${ConfigPackageLocation})

       set(CONF_INSTALL_PATH "\${dlib_CMAKE_DIR}/../../../")
       configure_file(dlibConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/config/dlibConfig.cmake" @ONLY)

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

       configure_file("dlib.pc.in" "dlib-1.pc" @ONLY)
       install(FILES "${CMAKE_CURRENT_BINARY_DIR}/dlib-1.pc"
           DESTINATION "${LIB_INSTALL_DIR}/pkgconfig")

   endif()

endif()
