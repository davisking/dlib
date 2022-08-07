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

### Summary and versioning ##################################################################################
project(dlib 
   VERSION        19.24.99
   DESCRIPTION    "Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems."
   HOMEPAGE_URL   http://dlib.net/
   LANGUAGES      C CXX) #CUDA

set(DLIB_CHECK_FOR_VERSION_MISMATCH
        DLIB_VERSION_MISMATCH_CHECK__EXPECTED_VERSION_${PROJECT_VERSION_MAJOR}_${PROJECT_VERSION_MINOR}_${PROJECT_VERSION_PATCH})
### Summary and versioning ##################################################################################

### Dependencies ############################################################################################ 
include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(FetchContent)
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
find_package(OpenCV QUIET)
include(cmake_utils/find_libwebp.cmake)
include(cmake_utils/find_blas.cmake)
# include(cmake_utils/find_cudnn.cmake)

# If you update this to 2.3.0 or later (latest is 2.10.0), it doesn't recognise PYBIND11_DESCR. TODO: fix
# TODO: use nanobind instead of pybind. According to author, it's better.
FetchContent_Declare(
     pybind11
     GIT_REPOSITORY https://github.com/pybind/pybind11.git
     GIT_TAG v2.2.3
)
FetchContent_MakeAvailable(pybind11)
### Dependencies ############################################################################################

### Options #################################################################################################
option(DLIB_ENABLE_ASSERTS       
   "Enable this if you want to turn on the DLIB_ASSERT macro")
option(DLIB_ENABLE_STACK_TRACE   
   "Enable this if you want to turn on the DLIB_STACK_TRACE macros")
option(DLIB_ISO_CPP_ONLY         
   "Enable this if you don't want to compile any non-ISO C++ code (i.e. you don't use any of the API Wrappers)")
option(DLIB_BUILD_TESTS
   "Build unit tests")
option(DLIB_BUILD_EXAMPLES
   "Build dlib examples")
option(DLIB_BUILD_TOOLS
   "Build dlib tools")
option(DLIB_BUILD_PYTHON
   "Build python bindings")
cmake_dependent_option(USE_SSE4_INSTRUCTIONS 
   "Compile your program with SSE4 instructions" ON "SSE4_IS_AVAILABLE_ON_HOST" OFF)
cmake_dependent_option(USE_AVX_INSTRUCTIONS  
   "Compile your program with AVX instructions" ON "AVX_IS_AVAILABLE_ON_HOST" OFF)
cmake_dependent_option(USE_NEON_INSTRUCTIONS 
   "Compile your program with ARM-NEON instructions" ON "ARM_NEON_IS_AVAILABLE" OFF)
cmake_dependent_option(DLIB_NO_GUI_SUPPORT
   "Enable this if you don't want to compile any of the dlib GUI code" OFF "NOT DLIB_ISO_CPP_ONLY;X11_FOUND" ON)
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
### Options #################################################################################################

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

if (DLIB_BUILD_PYTHON)
   pybind11_add_module(_dlib_pybind11
      ../tools/python/src/dlib.cpp
      ../tools/python/src/matrix.cpp
      ../tools/python/src/vector.cpp
      ../tools/python/src/svm_c_trainer.cpp
      ../tools/python/src/svm_rank_trainer.cpp
      ../tools/python/src/decision_functions.cpp
      ../tools/python/src/other.cpp
      ../tools/python/src/basic.cpp
      ../tools/python/src/cca.cpp
      ../tools/python/src/sequence_segmenter.cpp
      ../tools/python/src/svm_struct.cpp
      ../tools/python/src/image.cpp
      ../tools/python/src/image2.cpp
      ../tools/python/src/image3.cpp
      ../tools/python/src/image4.cpp
      ../tools/python/src/rectangles.cpp
      ../tools/python/src/object_detection.cpp
      ../tools/python/src/shape_predictor.cpp
      ../tools/python/src/correlation_tracker.cpp
      ../tools/python/src/face_recognition.cpp
      ../tools/python/src/cnn_face_detector.cpp
      ../tools/python/src/global_optimization.cpp
      ../tools/python/src/image_dataset_metadata.cpp
      ../tools/python/src/numpy_returns.cpp
      ../tools/python/src/line.cpp)

   if (NOT DLIB_NO_GUI_SUPPORT)
      target_sources(_dlib_pybind11 PRIVATE
         ../tools/python/src/gui.cpp)
   endif()

   target_link_libraries(_dlib_pybind11 PRIVATE dlib::dlib)
endif()

if (DLIB_BUILD_TESTS)
   add_executable(dtest test/main.cpp test/tester.cpp test/tester.h
      test/example.cpp
      test/active_learning.cpp
      test/any.cpp
      test/any_function.cpp
      test/array2d.cpp
      test/array.cpp
      test/assignment_learning.cpp
      test/base64.cpp
      test/bayes_nets.cpp
      test/bigint.cpp
      test/binary_search_tree_kernel_1a.cpp
      test/binary_search_tree_kernel_2a.cpp
      test/binary_search_tree_mm1.cpp
      test/binary_search_tree_mm2.cpp
      test/bridge.cpp
      test/bsp.cpp
      test/byte_orderer.cpp
      test/cca.cpp
      test/clustering.cpp
      test/cmd_line_parser.cpp
      test/cmd_line_parser_wchar_t.cpp
      test/compress_stream.cpp
      test/conditioning_class_c.cpp
      test/conditioning_class.cpp
      test/config_reader.cpp
      test/correlation_tracker.cpp
      test/crc32.cpp
      test/create_iris_datafile.cpp
      test/data_io.cpp
      test/directed_graph.cpp
      test/discriminant_pca.cpp
      test/disjoint_subsets.cpp
      test/disjoint_subsets_sized.cpp
      test/ekm_and_lisf.cpp
      test/empirical_kernel_map.cpp
      test/entropy_coder.cpp
      test/entropy_encoder_model.cpp
      test/example_args.cpp
      test/face.cpp
      test/fft.cpp
      test/fhog.cpp
      test/filtering.cpp
      test/find_max_factor_graph_nmplp.cpp
      test/find_max_factor_graph_viterbi.cpp
      test/geometry.cpp
      test/graph.cpp
      test/graph_cuts.cpp
      test/graph_labeler.cpp
      test/hash.cpp
      test/hash_map.cpp
      test/hash_set.cpp
      test/hash_table.cpp
      test/hog_image.cpp
      test/image.cpp
      test/invoke.cpp
      test/iosockstream.cpp
      test/is_same_object.cpp
      test/isotonic_regression.cpp
      test/kcentroid.cpp
      test/kernel_matrix.cpp
      test/kmeans.cpp
      test/learning_to_track.cpp
      test/least_squares.cpp
      test/linear_manifold_regularizer.cpp
      test/lspi.cpp
      test/lz77_buffer.cpp
      test/map.cpp
      test/matrix2.cpp
      test/matrix3.cpp
      test/matrix4.cpp
      test/matrix_chol.cpp
      test/matrix.cpp
      test/matrix_eig.cpp
      test/matrix_lu.cpp
      test/matrix_qr.cpp
      test/max_cost_assignment.cpp
      test/max_sum_submatrix.cpp
      test/md5.cpp
      test/member_function_pointer.cpp
      test/metaprogramming.cpp
      test/mpc.cpp
      test/multithreaded_object.cpp
      test/numerical_integration.cpp
      test/object_detector.cpp
      test/oca.cpp
      test/one_vs_all_trainer.cpp
      test/one_vs_one_trainer.cpp
      test/optimization.cpp
      test/optimization_test_functions.cpp
      test/global_optimization.cpp
      test/opt_qp_solver.cpp
      test/parallel_for.cpp
      test/parse.cpp
      test/pipe.cpp
      test/pixel.cpp
      test/probabilistic.cpp
      test/pyramid_down.cpp
      test/queue.cpp
      test/rand.cpp
      test/ranking.cpp
      test/read_write_mutex.cpp
      test/reference_counter.cpp
      test/rls.cpp
      test/random_forest.cpp
      test/sammon.cpp
      test/scan_image.cpp
      test/sequence.cpp
      test/sequence_labeler.cpp
      test/sequence_segmenter.cpp
      test/serialize.cpp
      test/set.cpp
      test/sldf.cpp
      test/sliding_buffer.cpp
      test/sockets2.cpp
      test/sockets.cpp
      test/sockstreambuf.cpp
      test/sparse_vector.cpp
      test/stack.cpp
      test/static_map.cpp
      test/static_set.cpp
      test/statistics.cpp
      test/std_vector_c.cpp
      test/string.cpp
      test/svm_c_linear.cpp
      test/svm_c_linear_dcd.cpp
      test/svm.cpp
      test/svm_multiclass_linear.cpp
      test/svm_struct.cpp
      test/svr_linear_trainer.cpp
      test/symmetric_matrix_cache.cpp
      test/thread_pool.cpp
      test/threads.cpp
      test/timer.cpp
      test/tokenizer.cpp
      test/trust_region.cpp
      test/tuple.cpp
      test/type_safe_union.cpp
      test/vectorstream.cpp
      test/dnn.cpp
      test/cublas.cpp
      test/find_optimal_parameters.cpp
      test/elastic_net.cpp)

   if (DLIB_USE_BLAS)
      target_sources(dtest PRIVATE
         test/blas_bindings/blas_bindings_dot.cpp
         test/blas_bindings/blas_bindings_gemm.cpp
         test/blas_bindings/blas_bindings_gemv.cpp
         test/blas_bindings/blas_bindings_ger.cpp
         test/blas_bindings/blas_bindings_scal_axpy.cpp
         test/blas_bindings/vector.cpp)

      target_compile_definitions(dtest PRIVATE DLIB_TEST_BLAS_BINDINGS)
   endif()

   # Treat warnings as errors.
   # Unused functions are ok though.
   if (CMAKE_COMPILER_IS_GNUCXX)
      target_compile_options(dtest PRIVATE -W -Werror -Wall -Wextra -Wpedantic -fdiagnostics-color=always)
      target_compile_options(dtest PRIVATE -Wno-unused-function -Wno-strict-overflow -Wno-maybe-uninitialized)
   elseif (MSVC)
      target_compile_options(dtest PRIVATE /WX)
   else() # basically Clang
      target_compile_options(dtest PRIVATE -W -Werror -Wno-comment)
   endif()

   target_compile_features(dtest PUBLIC cxx_std_14)
   target_link_libraries(dtest PRIVATE dlib::dlib)
endif()

if (DLIB_BUILD_EXAMPLES)
   macro(add_example name)
      add_executable(${name} ../examples/${name}.cpp)
      target_link_libraries(${name} PRIVATE dlib::dlib)
      target_compile_features(${name} PUBLIC cxx_std_14)
   endmacro()

   add_example(bayes_net_ex)
   add_example(bayes_net_from_disk_ex)
   add_example(bridge_ex)
   add_example(bsp_ex)
   add_example(compress_stream_ex)
   add_example(config_reader_ex)
   add_example(custom_trainer_ex)
   add_example(dnn_metric_learning_ex)
   add_example(dnn_introduction_ex)
   add_example(dnn_introduction2_ex)
   add_example(dnn_introduction3_ex)
   add_example(dnn_inception_ex)
   add_example(dnn_mmod_train_find_cars_ex)
   add_example(dnn_imagenet_train_ex)
   add_example(dnn_semantic_segmentation_train_ex)
   add_example(dnn_instance_segmentation_train_ex)
   add_example(dnn_metric_learning_on_images_ex)
   add_example(dir_nav_ex)
   add_example(empirical_kernel_map_ex)
   add_example(file_to_code_ex)
   add_example(graph_labeling_ex)
   add_example(integrate_function_adapt_simp_ex)
   add_example(iosockstream_ex)
   add_example(kcentroid_ex)
   add_example(kkmeans_ex)
   add_example(krls_ex)
   add_example(krls_filter_ex)
   add_example(krr_classification_ex)
   add_example(krr_regression_ex)
   add_example(learning_to_track_ex)
   add_example(least_squares_ex)
   add_example(linear_manifold_regularizer_ex)
   add_example(logger_custom_output_ex)
   add_example(logger_ex)
   add_example(logger_ex_2)
   add_example(matrix_ex)
   add_example(matrix_expressions_ex)
   add_example(max_cost_assignment_ex)
   add_example(member_function_pointer_ex)
   add_example(mlp_ex)
   add_example(model_selection_ex)
   add_example(multiclass_classification_ex)
   add_example(multithreaded_object_ex)
   add_example(optimization_ex)
   add_example(parallel_for_ex)
   add_example(pipe_ex)
   add_example(pipe_ex_2)
   add_example(quantum_computing_ex)
   add_example(queue_ex)
   add_example(rank_features_ex)
   add_example(running_stats_ex)
   add_example(rvm_ex)
   add_example(rvm_regression_ex)
   add_example(sequence_labeler_ex)
   add_example(sequence_segmenter_ex)
   add_example(server_http_ex)
   add_example(server_iostream_ex)
   add_example(sockets_ex)
   add_example(sockstreambuf_ex)
   add_example(std_allocator_ex)
   add_example(svm_c_ex)
   add_example(svm_ex)
   add_example(svm_pegasos_ex)
   add_example(svm_rank_ex)
   add_example(svm_sparse_ex)
   add_example(svm_struct_ex)
   add_example(svr_ex)
   add_example(thread_function_ex)
   add_example(thread_pool_ex)
   add_example(threaded_object_ex)
   add_example(threads_ex)
   add_example(timer_ex)
   add_example(train_shape_predictor_ex)
   add_example(using_custom_kernels_ex)
   add_example(xml_parser_ex)

   if (NOT DLIB_NO_GUI_SUPPORT)
      add_example(bayes_net_gui_ex)
      add_example(dnn_face_recognition_ex)
      add_example(dnn_mmod_ex)
      add_example(dnn_mmod_face_detection_ex)
      add_example(dnn_mmod_dog_hipsterizer)
      add_example(dnn_imagenet_ex)
      add_example(dnn_mmod_find_cars_ex)
      add_example(dnn_mmod_find_cars2_ex)
      add_example(dnn_semantic_segmentation_ex)
      add_example(dnn_instance_segmentation_ex)
      add_example(dnn_dcgan_train_ex)
      add_example(dnn_yolo_train_ex)
      add_example(dnn_self_supervised_learning_ex)
      add_example(face_detection_ex)
      add_example(face_landmark_detection_ex)
      add_example(fhog_ex)
      add_example(fhog_object_detector_ex)
      add_example(gui_api_ex)
      add_example(hough_transform_ex)
      add_example(image_ex)
      add_example(mpc_ex)
      add_example(object_detector_advanced_ex)
      add_example(object_detector_ex)
      add_example(one_class_classifiers_ex)
      add_example(random_cropper_ex)
      add_example(surf_ex)
      add_example(train_object_detector)
      add_example(video_tracking_ex)
      add_example(3d_point_cloud_ex)
   else()
      if (OpenCV_FOUND)
         add_example(webcam_face_pose_ex)
         target_include_directories(webcam_face_pose_ex PRIVATE ${OpenCV_INCLUDE_DIRS})
         target_link_libraries(webcam_face_pose_ex PRIVATE ${OpenCV_LIBS})
      endif()
   endif()

   if (DLIB_LINK_WITH_SQLITE3)
      add_example(sqlite_ex)
   endif()
endif()

if (DLIB_BUILD_TOOLS)
   add_executable(htmlify ../tools/htmlify/htmlify.cpp ../tools/htmlify/to_xml.cpp)
   target_compile_features(htmlify PUBLIC cxx_std_14)
   target_link_libraries(htmlify PRIVATE dlib::dlib)

   add_executable(dtoc ../tools/convert_dlib_nets_to_caffe/main.cpp)
   target_compile_features(dtoc PUBLIC cxx_std_14)
   target_link_libraries(dtoc PRIVATE dlib::dlib)

   add_executable(train_face_5point_model ../tools/archive/train_face_5point_model.cpp)
   target_compile_features(train_face_5point_model PUBLIC cxx_std_14)
   target_link_libraries(train_face_5point_model dlib::dlib)

   if (NOT DLIB_NO_GUI_SUPPORT)
      add_executable(imglab
         ../tools/imglab/src/main.cpp
         ../tools/imglab/src/metadata_editor.cpp
         ../tools/imglab/src/convert_pascal_xml.cpp
         ../tools/imglab/src/convert_pascal_v1.cpp
         ../tools/imglab/src/convert_idl.cpp
         ../tools/imglab/src/common.cpp
         ../tools/imglab/src/cluster.cpp
         ../tools/imglab/src/flip_dataset.cpp)
      target_compile_features(imglab PUBLIC cxx_std_14)
      target_link_libraries(imglab PRIVATE dlib::dlib)
   endif()
endif()
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

   if (DLIB_BUILD_TOOLS)
      install(TARGETS imglab dtoc train_face_5point_model RUNTIME DESTINATION bin)
      install(PROGRAMS ../tools/imglab/convert_imglab_paths_to_relative ../tools/imglab/copy_imglab_dataset DESTINATION bin)
   endif()

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