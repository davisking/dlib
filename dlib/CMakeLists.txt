cmake_minimum_required(VERSION 3.17)

### General stuff ###########################################################################################
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
### General stuff ###########################################################################################

### Policies ################################################################################################
if (POLICY CMP0048)
   cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
if(POLICY CMP0077)
   cmake_policy(SET CMP0077 NEW)
endif(POLICY CMP0077)
if (POLICY CMP0075)
   cmake_policy(SET CMP0075 NEW)
endif()
### Policies ################################################################################################

project(dlib 
   VERSION        19.24.99
   DESCRIPTION    "Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems."
   HOMEPAGE_URL   http://dlib.net/
   LANGUAGES      C CXX) #CUDA

### Dependencies ############################################################################################ 
include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(cmake_utils/check_if_neon_available.cmake)
include(cmake_utils/check_if_avx_instructions_executable_on_host.cmake)
include(cmake_utils/check_if_sse4_instructions_executable_on_host.cmake)
find_package(Threads REQUIRED)
find_package(X11)
#find_package(CUDAToolkit)
find_package(GIF)
find_package(PNG)
find_package(JPEG)
find_package(SQLite3)
include(cmake_utils/find_libwebp.cmake)
include(cmake_utils/find_blas.cmake)
# include(cmake_utils/find_cudnn.cmake)
### Dependencies ############################################################################################

### Options #################################################################################################
option(DLIB_ENABLE_ASSERTS       
   "Enable this if you want to turn on the DLIB_ASSERT macro")
option(DLIB_ENABLE_STACK_TRACE   
   "Enable this if you want to turn on the DLIB_STACK_TRACE macros")
option(DLIB_ISO_CPP_ONLY         
   "Enable this if you don't want to compile any non-ISO C++ code (i.e. you don't use any of the API Wrappers)")
cmake_dependent_option(USE_SSE4_INSTRUCTIONS 
   "Compile your program with SSE4 instructions" ON "SSE4_IS_AVAILABLE_ON_HOST" OFF)
cmake_dependent_option(USE_AVX_INSTRUCTIONS  
   "Compile your program with AVX instructions" ON "AVX_IS_AVAILABLE_ON_HOST" OFF)
cmake_dependent_option(USE_NEON_INSTRUCTIONS 
   "Compile your program with ARM-NEON instructions" ON "ARM_NEON_IS_AVAILABLE" OFF)
cmake_dependent_option(DLIB_NO_GUI_SUPPORT
   "Enable this if you don't want to compile any of the dlib GUI code" OFF "DLIB_ISO_CPP_ONLY;X11_FOUND" ON)
cmake_dependent_option(DLIB_PNG_SUPPORT         
   "Disable this if you don't want to link against libpng"  ON "NOT DLIB_ISO_CPP_ONLY" OFF)
cmake_dependent_option(DLIB_JPEG_SUPPORT        
   "Disable this if you don't want to link against libjpeg" ON "NOT DLIB_ISO_CPP_ONLY" OFF)
cmake_dependent_option(DLIB_WEBP_SUPPORT        
   "Disable this if you don't want to link against libwebp" ON "NOT DLIB_ISO_CPP_ONLY;WEBP_FOUND" OFF)
cmake_dependent_option(DLIB_GIF_SUPPORT         
   "Disable this if you don't want to link against libgif"  ON "NOT DLIB_ISO_CPP_ONLY;GIF_FOUND" OFF)
cmake_dependent_option(DLIB_LINK_WITH_SQLITE3   
   "Disable this if you don't want to link against sqlite3" ON "NOT DLIB_ISO_CPP_ONLY;SQLite3_FOUND" OFF)
cmake_dependent_option(DLIB_USE_MKL_FFT         
   "Disable this is you don't want to use the MKL DFTI FFT implementation" ON "NOT DLIB_ISO_CPP_ONLY;INTEL_MKL_FOUND" OFF)
cmake_dependent_option(DLIB_USE_MKL_SEQUENTIAL  
   "Enable this if you have MKL installed and want to use the sequential version instead of the multi-core version." OFF "NOT DLIB_ISO_CPP_ONLY;INTEL_MKL_FOUND" OFF)
cmake_dependent_option(DLIB_USE_MKL_WITH_TBB   
   "Enable this if you have MKL installed and want to use the tbb version instead of the openmp version." OFF "NOT DLIB_ISO_CPP_ONLY;INTEL_MKL_FOUND" OFF)
cmake_dependent_option(DLIB_USE_BLAS            
   "Disable this if you don't want to use a BLAS library"   ON "NOT DLIB_ISO_CPP_ONLY;BLAS_FOUND" OFF)
cmake_dependent_option(DLIB_USE_LAPACK          
   "Disable this if you don't want to use a LAPACK library" ON "NOT DLIB_ISO_CPP_ONLY;LAPACK_FOUND" OFF)
cmake_dependent_option(DLIB_USE_CUDA            
   "Disable this if you don't want to use NVIDIA CUDA"      ON "NOT DLIB_ISO_CPP_ONLY;CUDAToolkit_FOUND;CUDNN_FOUND" OFF)

if (DLIB_USE_CUDA)
   set(DLIB_USE_CUDA_COMPUTE_CAPABILITIES 50 CACHE STRING ${DLIB_USE_CUDA_COMPUTE_CAPABILITIES_STR})
endif()

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()
### Options #################################################################################################

### Libraries and executables ###############################################################################
add_library(dlib
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
   test_for_odr_violations.cpp)

if (DLIB_ISO_CPP_ONLY)
   target_compile_definitions(dlib PUBLIC DLIB_ISO_CPP_ONLY)
else()
   target_sources(dlib PRIVATE
      sockets/sockets_kernel_1.cpp
      bsp/bsp.cpp
      dir_nav/dir_nav_kernel_1.cpp
      dir_nav/dir_nav_kernel_2.cpp
      dir_nav/dir_nav_extensions.cpp
      gui_widgets/fonts.cpp
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
      cuda/cpu_dlib.cpp
      cuda/tensor_tools.cpp
      data_io/image_dataset_metadata.cpp
      data_io/mnist.cpp
      data_io/cifar.cpp
      global_optimization/global_function_search.cpp
      filtering/kalman_filter.cpp
      svm/auto.cpp)

   target_link_libraries(dlib PUBLIC Threads::Threads)
endif()

if (DLIB_NO_GUI_SUPPORT)
   target_compile_definitions(dlib PUBLIC DLIB_NO_GUI_SUPPORT)
else()
   target_sources(dlib PRIVATE
      gui_widgets/widgets.cpp
      gui_widgets/drawable.cpp
      gui_widgets/canvas_drawing.cpp
      gui_widgets/style.cpp
      gui_widgets/base_widgets.cpp
      gui_core/gui_core_kernel_1.cpp
      gui_core/gui_core_kernel_2.cpp)

   target_link_libraries(dlib PUBLIC X11::X11)
endif()

if (DLIB_PNG_SUPPORT)
   target_sources(dlib PRIVATE
      image_loader/png_loader.cpp
      image_saver/save_png.cpp)

   target_compile_definitions(dlib PUBLIC DLIB_PNG_SUPPORT)

   if (PNG_FOUND)
      target_link_libraries(dlib PUBLIC PNG::PNG)

   else()
      target_include_directories(dlib PRIVATE external/libpng external/zlib)
      target_sources(dlib PRIVATE
         external/libpng/arm/arm_init.c
         external/libpng/arm/filter_neon_intrinsics.c
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
         external/zlib/zutil.c)
      
      if (USE_NEON_INSTRUCTIONS)
         target_sources(dlib PRIVATE
            external/libpng/arm/arm_init.c
            external/libpng/arm/filter_neon_intrinsics.c
            external/libpng/arm/filter_neon.S)
      endif()
   endif()
endif()

if (DLIB_JPEG_SUPPORT)
   target_sources(dlib PRIVATE 
      image_loader/jpeg_loader.cpp
      image_saver/save_jpeg.cpp)

   target_compile_definitions(dlib PUBLIC DLIB_JPEG_SUPPORT)

   if (JPEG_FOUND)
      target_link_libraries(dlib PUBLIC JPEG::JPEG)

   else()
      target_compile_definitions(DLIB_JPEG_STATIC)
      target_sources(dlib PRIVATE
         external/libjpeg/jaricom.c
         external/libjpeg/jcapimin.c
         external/libjpeg/jcapistd.c
         external/libjpeg/jcarith.c
         external/libjpeg/jccoefct.c
         external/libjpeg/jccolor.c
         external/libjpeg/jcdctmgr.c
         external/libjpeg/jchuff.c
         external/libjpeg/jcinit.c
         external/libjpeg/jcmainct.c
         external/libjpeg/jcmarker.c
         external/libjpeg/jcmaster.c
         external/libjpeg/jcomapi.c
         external/libjpeg/jcparam.c
         external/libjpeg/jcprepct.c
         external/libjpeg/jcsample.c
         external/libjpeg/jdapimin.c
         external/libjpeg/jdapistd.c
         external/libjpeg/jdarith.c
         external/libjpeg/jdatadst.c
         external/libjpeg/jdatasrc.c
         external/libjpeg/jdcoefct.c
         external/libjpeg/jdcolor.c
         external/libjpeg/jddctmgr.c
         external/libjpeg/jdhuff.c
         external/libjpeg/jdinput.c
         external/libjpeg/jdmainct.c
         external/libjpeg/jdmarker.c
         external/libjpeg/jdmaster.c
         external/libjpeg/jdmerge.c
         external/libjpeg/jdpostct.c
         external/libjpeg/jdsample.c
         external/libjpeg/jerror.c
         external/libjpeg/jfdctflt.c
         external/libjpeg/jfdctfst.c
         external/libjpeg/jfdctint.c
         external/libjpeg/jidctflt.c
         external/libjpeg/jidctfst.c
         external/libjpeg/jidctint.c
         external/libjpeg/jmemmgr.c
         external/libjpeg/jmemnobs.c
         external/libjpeg/jquant1.c
         external/libjpeg/jquant2.c
         external/libjpeg/jutils.c)
   endif()
endif()

if (DLIB_WEBP_SUPPORT)
   target_sources(dlib PRIVATE
      image_loader/webp_loader.cpp
      image_saver/save_webp.cpp)
   target_link_libraries(dlib PUBLIC ${WEBP_LIBRARIES})
   target_include_directories(dlib PRIVATE ${WEBP_INCLUDE_DIRS})
   target_compile_definitions(dlib PUBLIC DLIB_WEBP_SUPPORT)
endif()

if (DLIB_GIF_SUPPORT)
   target_compile_definitions(dlib PUBLIC DLIB_GIF_SUPPORT)
   target_link_libraries(dlib PUBLIC GIF::GIF)
endif()

if (DLIB_LINK_WITH_SQLITE3)
   target_compile_definitions(dlib PUBLIC DLIB_LINK_WITH_SQLITE3)
   target_link_libraries(dlib PUBLIC SQLite::SQLite3)
endif()

if (DLIB_USE_BLAS)
   target_compile_definitions(dlib PUBLIC DLIB_USE_BLAS)
   target_link_libraries(dlib PUBLIC ${BLAS_LIBRARIES})
endif()

if (DLIB_USE_LAPACK)
   target_compile_definitions(dlib PUBLIC DLIB_USE_LAPACK)
   target_link_libraries(dlib PUBLIC ${LAPACK_LIBRARIES})
endif()

if (DLIB_USE_MKL_FFT)
   target_compile_definitions(dlib PUBLIC DLIB_USE_MKL_FFT)
   target_include_directories(dlib PUBLIC ${MKL_INCLUDE_DIR})
   target_link_libraries(dlib PUBLIC ${MKL_LIBRARIES})
endif()

if (DLIB_USE_BLAS OR DLIB_USE_LAPACK OR DLIB_USE_MKL_FFT)
   if (DLIB_USE_MKL_WITH_TBB)
      target_compile_definitions(dlib PUBLIC DLIB_USE_MKL_WITH_TBB)
   endif()

   if (DLIB_USE_MKL_SEQUENTIAL)
      target_compile_definitions(dlib PUBLIC DLIB_USE_MKL_SEQUENTIAL)
   endif()
endif()

# if (DLIB_USE_CUDA)
#    set(CUDA_HOST_COMPILATION_CPP ON)
   
#    # list(APPEND CUDA_NVCC_FLAGS "-D__STRICT_ANSI__;-D_MWAITXINTRIN_H_INCLUDED;-D_FORCE_INLINES;${FLAGS_FOR_NVCC}")
#    # list(APPEND CUDA_NVCC_FLAGS ${active_preprocessor_switches})
#    # if (NOT DLIB_IN_PROJECT_BUILD)
#    #    LIST(APPEND CUDA_NVCC_FLAGS -DDLIB__CMAKE_GENERATED_A_CONFIG_H_FILE)
#    # endif()

#    target_sources(dlib PRIVATE 
#       cuda/cuda_dlib.cu 
#       cuda/cudnn_dlibapi.cpp
#       cuda/cublas_dlibapi.cpp
#       cuda/cusolver_dlibapi.cu
#       cuda/curand_dlibapi.cpp
#       cuda/cuda_data_ptr.cpp
#       cuda/gpu_data.cpp)

#    target_link_libraries(dlib PUBLIC CUDA::cudart CUDA::curand CUDA::cusolver CUDA::cublas)
#    target_link_libraries(dlib PUBLIC ${CUDNN_LIBRARIES})
#    target_include_directories(dlib PRIVATE ${CUDNN_INCLUDE_DIRS})
#    target_compile_definitions(dlib PRIVATE ${CUDNN_DEFINITIONS})
#    target_compile_definitions(dlib PRIVATE DLIB_USE_CUDA)
#    set_target_properties(dlib PROPERTIES CUDA_ARCHITECTURES ${DLIB_USE_CUDA_COMPUTE_CAPABILITIES})
# endif()

# Used by config.h.in
if (DLIB_ENABLE_ASSERTS)
   target_compile_definitions(dlib PUBLIC ENABLE_ASSERTS)
else()
   target_compile_definitions(dlib PUBLIC DLIB_DISABLE_ASSERTS)
endif()

if (DLIB_ENABLE_STACK_TRACE)
   target_compile_definitions(dlib PUBLIC DLIB_ENABLE_STACK_TRACE)
endif()

target_include_directories(dlib PUBLIC
        INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        INTERFACE $<INSTALL_INTERFACE:include>)

set(DLIB_CHECK_FOR_VERSION_MISMATCH
        DLIB_VERSION_MISMATCH_CHECK__EXPECTED_VERSION_${PROJECT_VERSION_MAJOR}_${PROJECT_VERSION_MINOR}_${PROJECT_VERSION_PATCH})
target_compile_definitions(dlib PUBLIC DLIB_CHECK_FOR_VERSION_MISMATCH=${DLIB_CHECK_FOR_VERSION_MISMATCH})

# target_compile_features(dlib PUBLIC cxx_std_14)
set_target_properties(dlib PROPERTIES 
   VERSION ${PROJECT_VERSION}
   CXX_STANDARD 14
   CXX_STANDARD_REQUIRED ON
   POSITION_INDEPENDENT_CODE ON)
   # CXX_VISIBILITY_PRESET  "hidden"
   # CUDA_VISIBILITY_PRESET "hidden")

add_library(dlib::dlib ALIAS dlib)

### Libraries and executables ###############################################################################

### Installation ############################################################################################

get_directory_property(has_parent PARENT_DIRECTORY)

if (NOT has_parent)
   include(GNUInstallDirs)

   install(TARGETS dlib
      EXPORT dlib
      FILE_SET HEADERS
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Windows considers .dll to be runtime artifacts
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

   install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dlib
      FILES_MATCHING
      PATTERN "*.h"
      PATTERN "*.cmake"
      PATTERN "*_tutorial.txt"
      PATTERN "cassert"
      PATTERN "cstring"
      PATTERN "fstream"
      PATTERN "iomanip"
      PATTERN "iosfwd"
      PATTERN "iostream"
      PATTERN "istream"
      PATTERN "locale"
      PATTERN "ostream"
      PATTERN "sstream"
      REGEX "${CMAKE_CURRENT_BINARY_DIR}" EXCLUDE)

   configure_file(${PROJECT_SOURCE_DIR}/config.h.in   ${CMAKE_CURRENT_BINARY_DIR}/config.h)
   configure_file(${PROJECT_SOURCE_DIR}/revision.h.in ${CMAKE_CURRENT_BINARY_DIR}/revision.h)
   configure_file(cmake_utils/dlib.pc.in              dlib-1.pc @ONLY)

   configure_package_config_file(cmake_utils/dlibConfig.cmake.in
           ${CMAKE_CURRENT_BINARY_DIR}/config/FooConfig.cmake
           INSTALL_DESTINATION ${LIB_INSTALL_DIR}/dlib/cmake)

   install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/config.h
      ${CMAKE_CURRENT_BINARY_DIR}/revision.h
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dlib)

   install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dlib-1.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

   set(ConfigPackageLocation "${CMAKE_INSTALL_LIBDIR}/cmake/dlib")
   install(EXPORT dlib
      FILE dlib.cmake
      NAMESPACE dlib::
      DESTINATION ${ConfigPackageLocation})

   write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/config/dlibConfigVersion.cmake"
      VERSION ${VERSION}
      COMPATIBILITY AnyNewerVersion)

   install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/config/dlibConfig.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/config/dlibConfigVersion.cmake
      DESTINATION ${ConfigPackageLocation})

   include(CPack)
endif()
### Installation ############################################################################################