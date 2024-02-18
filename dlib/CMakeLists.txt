cmake_minimum_required(VERSION 3.17.0)

set(CPACK_PACKAGE_NAME "dlib")
set(CPACK_PACKAGE_VERSION_MAJOR "19")
set(CPACK_PACKAGE_VERSION_MINOR "24")
set(CPACK_PACKAGE_VERSION_PATCH "99")
set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})
set(DLIB_VERSION ${VERSION})
project(dlib VERSION ${VERSION} LANGUAGES C CXX)
message(STATUS "Using CMake version:      ${CMAKE_VERSION}")
message(STATUS "Compiling dlib version:   ${VERSION}")

# Generator expression helpers
set(IS_CLANG      "$<CXX_COMPILER_ID:Clang>")
set(IS_GCC        "$<CXX_COMPILER_ID:GNU>")
set(IS_MSVC       "$<CXX_COMPILER_ID:MSVC>")
set(IS_CLANG_OLD  "$<AND:${IS_CLANG},$<VERSION_LESS:$<CXX_COMPILER_VERSION>,7.0.0>>")
set(IS_WIN32      "$<BOOL:${WIN32}>")
set(IS_NOT_WIN32  "$<NOT:${IS_WIN32}>")

# Dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CMakeDependentOption)
find_package(PkgConfig)
find_package(Threads)
find_package(AVX)
find_package(SSE)
find_package(NEON)
find_package(GIF)
find_package(JPEG)
find_package(PNG)
find_package(WEBP)
find_package(X11)
find_package(SQLite3)
find_package(CUDAToolkit)
find_package(CUDNN)
find_package(MKL)
find_package(CBLAS)
if (PKG_CONFIG_FOUND)
   pkg_check_modules(FFMPEG IMPORTED_TARGET libavdevice libavfilter libavformat libavcodec libavutil libswresample libswscale)
endif()

# Options
option(DLIB_ISO_CPP_ONLY         "Build using standard c++ only"  OFF)
option(DLIB_ENABLE_ASSERTS       "Enable Debug assert statements" OFF)
option(DLIB_ENABLE_STACK_TRACE   "Enabled Debug stack trace"      OFF)
option(DLIB_JPEG_SUPPORT         "Enable JPEG support"            ON)
option(DLIB_PNG_SUPPORT          "Enable PNG support"             ON)
cmake_dependent_option(DLIB_USE_BLAS            "Enable BLAS support"      ON  "MKL_FOUND OR BLAS_FOUND"       OFF)
cmake_dependent_option(DLIB_USE_LAPACK          "Enable LAPACK support"    ON  "MKL_FOUND OR LAPACK_FOUND"     OFF)
cmake_dependent_option(DLIB_USE_AVX             "Enable AVX SIMD"          ON  "AVX_FOUND"                     OFF)
cmake_dependent_option(DLIB_USE_SSE             "Enable SSE SIMD"          ON  "SSE_FOUND"                     OFF)
cmake_dependent_option(DLIB_USE_NEON            "Enable NEON SIMD"         ON  "NEON_FOUND"                    OFF)
cmake_dependent_option(DLIB_GIF_SUPPORT         "Enable GIF support"       ON  "GIF_FOUND"                     OFF)
cmake_dependent_option(DLIB_WEBP_SUPPORT        "Enable WEBP support"      ON  "WEBP_FOUND"                    OFF)
cmake_dependent_option(DLIB_NO_GUI_SUPPORT      "Disable GUI support"      OFF "X11_FOUND OR WIN32"            ON)
cmake_dependent_option(DLIB_LINK_WITH_SQLITE3   "Enable Sqlite3 support"   ON  "SQLite3_FOUND"                 OFF)
cmake_dependent_option(DLIB_USE_FFMPEG          "Enable FFMPEG support"    ON  "FFMPEG_FOUND"                  OFF)
cmake_dependent_option(DLIB_USE_CUDA            "Enable CUDA support"      ON  "CUDAToolkit_FOUND;CUDNN_FOUND" OFF)
cmake_dependent_option(DLIB_USE_MKL_SEQUENTIAL  "Enable MKL sequential"    ON  "MKL_FOUND"                     OFF)
cmake_dependent_option(DLIB_USE_MKL_WITH_TBB    "Enable MKL tbb"           ON  "MKL_FOUND"                     OFF)
cmake_dependent_option(DLIB_USE_MKL_WITH_GNU    "Enable MKL GNU OMP"       ON  "MKL_FOUND"                     OFF)
include(cmake_utils/release_build_by_default)

# dlib
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
   test_for_odr_violations.cpp
   fft/fft.cpp)

add_library(dlib::dlib ALIAS dlib)
target_include_directories(dlib
   INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
   INTERFACE $<INSTALL_INTERFACE:include>)
target_compile_options(dlib PUBLIC $<${IS_CLANG_OLD}:-ftemplate-depth=500>)
target_compile_options(dlib PUBLIC $<${IS_MSVC}:/bigobj>)
target_compile_options(dlib PUBLIC $<${IS_GCC}:-Wreturn-type>)
target_compile_definitions(dlib PUBLIC DLIB_VERSION=${DLIB_VERSION})
set_target_properties(dlib PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (NOT DLIB_ISO_CPP_ONLY)
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

   if (DLIB_NO_GUI_SUPPORT)
      target_compile_definitions(dlib PUBLIC DLIB_NO_GUI_SUPPORT)
   else()
      message(STATUS "Enabled GUI support")
      target_sources(dlib PRIVATE
         gui_core/gui_core_kernel_1.cpp
         gui_core/gui_core_kernel_2.cpp
         gui_widgets/base_widgets.cpp
         gui_widgets/widgets.cpp
         gui_widgets/drawable.cpp
         gui_widgets/canvas_drawing.cpp
         gui_widgets/style.cpp)
      target_link_libraries(dlib PUBLIC $<${IS_NOT_WIN32}:X11::X11>)
   endif()

   if(DLIB_ENABLE_ASSERTS)
      message(STATUS "Enabled DLIB_ASSERT statements")
      target_compile_definitions(dlib PUBLIC DLIB_ENABLE_ASSERTS)
      target_compile_definitions(dlib PUBLIC ENABLE_ASSERTS)
   else()
      target_compile_definitions(dlib PUBLIC DLIB_DISABLE_ASSERTS)
   endif()

   if (DLIB_ENABLE_STACK_TRACE)
      message(STATUS "Enabled stack trace support")
      target_compile_definitions(dlib PUBLIC DLIB_ENABLE_STACK_TRACE)
   endif()

   if (DLIB_USE_AVX)
      message(STATUS "Enabled AVX intrinsics")
      target_compile_options(dlib PUBLIC $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${AVX_CFLAGS}>)
      target_compile_definitions(dlib PUBLIC $<${HAVE_AVX2}:DLIB_HAVE_AVX2>)
      target_compile_definitions(dlib PUBLIC $<${HAVE_AVX}:DLIB_HAVE_AVX>)
   endif()

   if (DLIB_USE_SSE)
      message(STATUS "Enabled SSE intrinsics")
      target_compile_options(dlib PUBLIC $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${SSE_CFLAGS}>)
      target_compile_definitions(dlib PUBLIC $<${HAVE_SSE42}:DLIB_HAVE_SSE42>)
      target_compile_definitions(dlib PUBLIC $<${HAVE_SSE41}:DLIB_HAVE_SSE41>)
      target_compile_definitions(dlib PUBLIC $<${HAVE_SSE3}:DLIB_HAVE_SSE3>)
      target_compile_definitions(dlib PUBLIC $<${HAVE_SSE2}:DLIB_HAVE_SSE2>)
   endif()

   if (DLIB_USE_NEON)
      message(STATUS "Enabled NEON intrinsics")
      target_compile_features(dlib PUBLIC $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${NEON_CFLAGS}>)
      target_compile_definitions(dlib PUBLIC DLIB_HAVE_NEON)
   endif()

   if (DLIB_JPEG_SUPPORT)
      message(STATUS "Enabled JPEG wrappers")
      target_sources(dlib PRIVATE
         image_loader/jpeg_loader.cpp
         image_saver/save_jpeg.cpp)
      target_compile_definitions(dlib PUBLIC DLIB_JPEG_SUPPORT)

      if (JPEG_FOUND)
         target_link_libraries(dlib PUBLIC JPEG::JPEG)
      else()
         message(STATUS "    Using internal copy of libjpeg. Consider installing libjpeg")
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
         target_compile_definitions(dlib PRIVATE DLIB_JPEG_STATIC)
      endif()
   endif()

   if (DLIB_PNG_SUPPORT)
      message(STATUS "Enabled PNG wrappers")
      target_sources(dlib PRIVATE
         image_loader/png_loader.cpp
         image_saver/save_png.cpp)
      target_compile_definitions(dlib PUBLIC DLIB_PNG_SUPPORT)

      if (PNG_FOUND)
         target_link_libraries(dlib PUBLIC PNG::PNG)
      else()
         message(STATUS "    Using internal copy of libpng. Consider installing libpng")
         target_sources(dlib PRIVATE
            external/libpng/arm/arm_init.c
            external/libpng/arm/filter_neon_intrinsics.c
            external/libpng/arm/palette_neon_intrinsics.c
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
         target_include_directories(dlib PRIVATE external/zlib external/libpng)
      endif()
   endif()

   if (DLIB_GIF_SUPPORT)
      message(STATUS "Enabled GIF wrappers")
      target_compile_definitions(dlib PUBLIC DLIB_GIF_SUPPORT)
      target_link_libraries(dlib PUBLIC GIF::GIF)
   endif()

   if (DLIB_WEBP_SUPPORT)
      message(STATUS "Enabled WEBP wrappers")
      target_sources(dlib PRIVATE
         image_loader/webp_loader.cpp
         image_saver/save_webp.cpp)
      target_compile_definitions(dlib PUBLIC DLIB_WEBP_SUPPORT)
      target_link_libraries(dlib PUBLIC WEBP::WEBP)
   endif()

   if (DLIB_LINK_WITH_SQLITE3)
      message(STATUS "Enabled SQLITE3 wrappers")
      target_link_libraries(dlib PUBLIC SQLite::SQLite3)
      target_compile_definitions(dlib PUBLIC DLIB_LINK_WITH_SQLITE3)
   endif()

   if (DLIB_USE_FFMPEG)
      message(STATUS "Enabled FFMPEG (libavformat) wrappers")
      target_link_libraries(dlib PUBLIC PkgConfig::FFMPEG)
      target_compile_definitions(dlib PUBLIC DLIB_USE_FFMPEG)
   endif()

   if(DLIB_USE_MKL_WITH_TBB)
      message(STATUS "Enabled MKL TBB")
      target_compile_definitions(dlib PUBLIC DLIB_USE_MKL_WITH_TBB DLIB_USE_MKL_FFT)
      target_link_libraries(dlib PUBLIC mkl::tbb)
   elseif(DLIB_USE_MKL_WITH_GNU)
      message(STATUS "Enabled MKL GNU OMP")
      target_compile_definitions(dlib PUBLIC DLIB_USE_MKL_WITH_GNU DLIB_USE_MKL_FFT)
      target_link_libraries(dlib PUBLIC mkl::gnu)
   elseif (DLIB_USE_MKL_SEQUENTIAL)
      message(STATUS "Enabled MKL sequential")
      target_compile_definitions(dlib PUBLIC DLIB_USE_MKL_SEQUENTIAL DLIB_USE_MKL_FFT)
      target_link_libraries(dlib PUBLIC mkl::sequential)
   endif()

   if (DLIB_USE_BLAS)
      target_compile_definitions(dlib PUBLIC DLIB_USE_BLAS)
      if (NOT DLIB_USE_MKL_SEQUENTIAL AND NOT DLIB_USE_MKL_WITH_TBB AND NOT DLIB_USE_MKL_WITH_GNU)
         message(STATUS "Enabled BLAS support - using BLAS library")
         target_link_libraries(dlib PUBLIC BLAS::BLAS)
      else()
         message(STATUS "Enabled BLAS support - Using MKL library")
      endif()
   endif()

   if (DLIB_USE_LAPACK)
      target_compile_definitions(dlib PUBLIC DLIB_USE_LAPACK)
      if (NOT DLIB_USE_MKL_SEQUENTIAL AND NOT DLIB_USE_MKL_WITH_TBB AND NOT DLIB_USE_MKL_WITH_GNU)
         message(STATUS "Enabled LAPACK support - using LAPACK library")
         target_link_libraries(dlib PUBLIC LAPACK::LAPACK)
      else()
         message(STATUS "Enabled LAPACK support - using MKL library")
      endif()
   endif()

   if (DLIB_USE_CUDA)
      message(STATUS "Enabled CUDA support")
      set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
      enable_language(CUDA)
      target_sources(dlib PRIVATE
         cuda/cuda_dlib.cu 
         cuda/cudnn_dlibapi.cpp
         cuda/cublas_dlibapi.cpp
         cuda/cusolver_dlibapi.cu
         cuda/curand_dlibapi.cpp
         cuda/cuda_data_ptr.cpp
         cuda/gpu_data.cpp)
      target_link_libraries(dlib PUBLIC CUDA::cudart CUDA::cublas CUDA::curand CUDA::cusolver CUDNN::CUDNN)
      target_compile_definitions(dlib PUBLIC DLIB_USE_CUDA)
      set_target_properties(dlib PROPERTIES CUDA_ARCHITECTURES "50;72")
   endif()
endif()

# Install
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS dlib
   EXPORT dlib 
   RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Windows considers .dll to be runtime artifacts
   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
   ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

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


configure_file(${PROJECT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)
configure_file(${PROJECT_SOURCE_DIR}/revision.h.in ${CMAKE_CURRENT_BINARY_DIR}/revision.h)
configure_file(${PROJECT_SOURCE_DIR}/cmake_utils/dlibConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/config/dlibConfig.cmake @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/config.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dlib)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/revision.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dlib)

## Config.cmake generation and installation
set(ConfigPackageLocation "${CMAKE_INSTALL_LIBDIR}/cmake/dlib")
install(EXPORT dlib
   NAMESPACE dlib::
   DESTINATION ${ConfigPackageLocation})

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