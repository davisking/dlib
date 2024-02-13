cmake_minimum_required(VERSION 3.17.0)

set(CPACK_PACKAGE_NAME "dlib")
set(CPACK_PACKAGE_VERSION_MAJOR "19")
set(CPACK_PACKAGE_VERSION_MINOR "24")
set(CPACK_PACKAGE_VERSION_PATCH "99")
set(VERSION ${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH})
project(dlib VERSION ${VERSION} LANGUAGES C CXX)

# Generator expression helpers
set(IS_CLANG      "$<CXX_COMPILER_ID:Clang>")
set(IS_CLANG_OLD  "$<AND:${IS_CLANG},$<VERSION_LESS:$<CXX_COMPILER_VERSION>,7.0.0>>")
set(IS_WIN32      "$<BOOL:${WIN32}>")
set(IS_NOT_WIN32  "$<NOT:${IS_WIN32}>")

# Dependencies
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(APPEND CMAKE_PREFIX_PATH /opt/intel/oneapi/mkl/latest)
# set(Matlab_ROOT_DIR /usr/share/matlab)
include(CMakeDependentOption)
find_package(PkgConfig)
find_package(Matlab)
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
find_package(OpenCV QUIET)

# Options
option(DLIB_BUILD_TESTS          "Build tests"                    OFF)
option(DLIB_BUILD_PYTHON         "Build python bindings"          OFF)
option(DLIB_BUILD_EXAMPLES       "Build examples"                 OFF)
option(DLIB_BUILD_TOOLS          "Build tools"                    OFF)
option(DLIB_ISO_CPP_ONLY         "Build using standard c++ only"  OFF)
option(DLIB_ENABLE_ASSERTS       "Enable Debug assert statements" OFF)
option(DLIB_ENABLE_STACK_TRACE   "Enabled Debug stack trace"      OFF)
option(DLIB_JPEG_SUPPORT         "Enable JPEG support"            ON)
option(DLIB_PNG_SUPPORT          "Enable PNG support"             ON)
cmake_dependent_option(DLIB_BUILD_MATLAB        "Build matlab bindings"    ON  "Matlab_FOUND" OFF)
cmake_dependent_option(DLIB_USE_BLAS            "Enable BLAS support"      ON  "MKL_FOUND OR BLAS_FOUND" OFF)
cmake_dependent_option(DLIB_USE_LAPACK          "Enable LAPACK support"    ON  "MKL_FOUND OR LAPACK_FOUND" OFF)
cmake_dependent_option(DLIB_USE_AVX             "Enable AVX SIMD"          ON  "AVX_FOUND"         OFF)
cmake_dependent_option(DLIB_USE_SSE             "Enable SSE SIMD"          ON  "SSE_FOUND"         OFF)
cmake_dependent_option(DLIB_USE_NEON            "Enable NEON SIMD"         ON  "NEON_FOUND"        OFF)
cmake_dependent_option(DLIB_GIF_SUPPORT         "Enable GIF support"       ON  "GIF_FOUND"         OFF)
cmake_dependent_option(DLIB_WEBP_SUPPORT        "Enable WEBP support"      ON  "WEBP_FOUND"        OFF)
cmake_dependent_option(DLIB_NO_GUI_SUPPORT      "Disable GUI support"      OFF "X11_FOUND OR WIN32" ON)
cmake_dependent_option(DLIB_LINK_WITH_SQLITE3   "Enable Sqlite3 support"   ON  "SQLite3_FOUND"     OFF)
cmake_dependent_option(DLIB_USE_FFMPEG          "Enable FFMPEG support"    ON  "FFMPEG_FOUND"      OFF)
cmake_dependent_option(DLIB_USE_CUDA            "Enable CUDA support"      ON  "CUDAToolkit_FOUND" OFF)
cmake_dependent_option(DLIB_USE_MKL_SEQUENTIAL  "Enable MKL sequential"    ON  "MKL_FOUND"         OFF)
cmake_dependent_option(DLIB_USE_MKL_WITH_TBB    "Enable MKL tbb"           ON  "MKL_FOUND"         OFF)
cmake_dependent_option(DLIB_USE_MKL_FFT         "Enable MKL FFT"           ON  "MKL_FOUND"         OFF)

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
target_compile_options(dlib PUBLIC $<${IS_WIN32}:/bigobj>)
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
      target_compile_definitions(dlib PUBLIC DLIB_ENABLE_ASSERTS)
      target_compile_definitions(dlib PUBLIC ENABLE_ASSERTS)
   else()
      target_compile_definitions(dlib PUBLIC DLIB_DISABLE_ASSERTS)
   endif()

   if (DLIB_ENABLE_STACK_TRACE)
      target_compile_definitions(dlib PUBLIC DLIB_ENABLE_STACK_TRACE)
   endif()

   if (DLIB_USE_AVX)
      target_compile_options(dlib PUBLIC ${AVX_CFLAGS})
      target_compile_definitions(dlib PUBLIC $<${HAVE_AVX2}:DLIB_HAVE_AVX2>)
      target_compile_definitions(dlib PUBLIC $<${HAVE_AVX}:DLIB_HAVE_AVX>)
   endif()

   if (DLIB_USE_SSE)
      target_compile_options(dlib PUBLIC ${SSE_CFLAGS})
      target_compile_definitions(dlib PUBLIC $<${HAVE_SSE42}:DLIB_HAVE_SSE42>)
      target_compile_definitions(dlib PUBLIC $<${HAVE_SSE41}:DLIB_HAVE_SSE41>)
      target_compile_definitions(dlib PUBLIC $<${HAVE_SSE3}:DLIB_HAVE_SSE3>)
   endif()

   if (DLIB_USE_NEON)
      target_compile_features(dlib PUBLIC ${NEON_CFLAGS})
      target_compile_definitions(dlib PUBLIC DLIB_HAVE_NEON)
   endif()

   if (DLIB_JPEG_SUPPORT)
      target_sources(dlib PRIVATE
         image_loader/jpeg_loader.cpp
         image_saver/save_jpeg.cpp)
      target_compile_definitions(dlib PUBLIC DLIB_JPEG_SUPPORT)

      if (JPEG_FOUND)
         target_link_libraries(dlib PUBLIC JPEG::JPEG)
      else()
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
      target_sources(dlib PRIVATE
         image_loader/png_loader.cpp
         image_saver/save_png.cpp)
      target_compile_definitions(dlib PUBLIC DLIB_PNG_SUPPORT)

      if (PNG_FOUND)
         target_link_libraries(dlib PUBLIC PNG::PNG)

      else()
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
      target_compile_definitions(dlib PUBLIC DLIB_GIF_SUPPORT)
      target_link_libraries(dlib PUBLIC GIF::GIF)
   endif()

   if (DLIB_WEBP_SUPPORT)
      target_sources(dlib PRIVATE
         image_loader/webp_loader.cpp
         image_saver/save_webp.cpp)
      target_compile_definitions(dlib PUBLIC DLIB_WEBP_SUPPORT)
      target_link_libraries(dlib PUBLIC WEBP::WEBP)
   endif()

   if (DLIB_LINK_WITH_SQLITE3)
      target_link_libraries(dlib PUBLIC SQLite::SQLite3)
      target_compile_definitions(dlib PUBLIC DLIB_LINK_WITH_SQLITE3)
   endif()

   if (DLIB_USE_FFMPEG)
      target_link_libraries(dlib PUBLIC PkgConfig::FFMPEG)
      target_compile_definitions(dlib PUBLIC DLIB_USE_FFMPEG)
   endif()

   if (DLIB_USE_MKL_SEQUENTIAL)
      target_compile_definitions(dlib PUBLIC DLIB_USE_MKL_SEQUENTIAL)
      target_link_libraries(dlib PUBLIC mkl::sequential)
   elseif(DLIB_USE_MKL_WITH_TBB)
      target_compile_definitions(dlib PUBLIC DLIB_USE_MKL_WITH_TBB)
      target_link_libraries(dlib PUBLIC mkl::tbb)
   endif()

   if (DLIB_USE_MKL_FFT)
      # Libs are already set above
      target_compile_definitions(dlib PUBLIC DLIB_USE_MKL_FFT)
   endif()

   if (DLIB_USE_BLAS)
      target_compile_definitions(dlib PUBLIC DLIB_USE_BLAS)
      if(BLAS_FOUND)
         target_link_libraries(dlib PUBLIC BLAS::BLAS)
      endif()
   endif()

   if (DLIB_USE_LAPACK)
      target_compile_definitions(dlib PUBLIC DLIB_USE_LAPACK)
      if (LAPACK_FOUND)
         target_link_libraries(dlib PUBLIC LAPACK::LAPACK)
      endif()
   endif()

   if (DLIB_USE_CUDA)
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

# dtest
if (DLIB_BUILD_TESTS)
   add_executable(dtest
      test/main.cpp 
      test/tester.cpp
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
      test/constexpr_if.cpp
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
      test/math.cpp
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
      test/elastic_net.cpp
      test/te.cpp
      test/ffmpeg.cpp
      test/optional.cpp
      test/scope.cpp
   )

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
   
   target_link_libraries(dtest dlib::dlib)
   get_filename_component(DLIB_FFMPEG_DATA ${CMAKE_SOURCE_DIR}/test/ffmpeg_data/details.cfg REALPATH)
   target_compile_definitions(dtest PRIVATE DLIB_FFMPEG_DATA="${DLIB_FFMPEG_DATA}")
   target_link_options(dtest PRIVATE $<$<AND:$<CONFIG:RELEASE>,${IS_NOT_WIN32}>:-s>)
endif()

# Python bindings
if (DLIB_BUILD_PYTHON)
   add_subdirectory(external/pybind11)
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
   
   if(NOT DLIB_NO_GUI_SUPPORT)
      target_sources(_dlib_pybind11 PRIVATE ../tools/python/src/gui.cpp)
   endif()
   target_link_libraries(_dlib_pybind11 PRIVATE dlib::dlib)
   target_compile_definitions(_dlib_pybind11 PRIVATE DLIB_NO_ABORT_ON_2ND_FATAL_ERROR)
   set_target_properties(_dlib_pybind11 PROPERTIES
      INTERPROCEDURAL_OPTIMIZATION ON
      CXX_VISIBILITY_PRESET "hidden"
      CUDA_VISIBILITY_PRESET "hidden")
endif()

# Matlab bindings
if (DLIB_BUILD_MATLAB)
   macro(add_mex_function name)
      matlab_add_mex(NAME ${name} MODULE SRC matlab/${name}.cpp LINK_TO dlib)
      target_compile_definitions(${name} PRIVATE MEX_FILENAME=${name})
   endmacro()

   add_mex_function(example_mex_function)
   add_mex_function(example_mex_callback)
   add_mex_function(example_mex_struct)
   add_mex_function(example_mex_class)
endif()

# Examples
if (DLIB_BUILD_EXAMPLES)
   macro(add_example name)
      add_executable(${name} ../examples/${name}.cpp)
      target_link_libraries(${name} dlib::dlib)
      target_link_options(${name}  PRIVATE $<$<AND:$<CONFIG:RELEASE>,${IS_NOT_WIN32}>:-s>)
   endmacro()

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
   add_example(bayes_net_ex)
   add_example(bayes_net_from_disk_ex)
   add_example(bridge_ex)
   add_example(bsp_ex)
   add_example(compress_stream_ex)
   add_example(config_reader_ex)
   add_example(custom_trainer_ex)
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
   if (DLIB_LINK_WITH_SQLITE3)
      add_example(sqlite_ex)
   endif()
   if (NOT DLIB_NO_GUI_SUPPORT)
      add_example(bayes_net_gui_ex)
      add_example(dnn_face_recognition_ex)
      add_example(dnn_dcgan_train_ex)
      add_example(dnn_mmod_ex)
      add_example(dnn_mmod_face_detection_ex)
      add_example(dnn_mmod_find_cars_ex)
      add_example(dnn_mmod_find_cars2_ex)
      add_example(dnn_mmod_dog_hipsterizer)
      add_example(dnn_imagenet_ex)
      add_example(dnn_self_supervised_learning_ex)
      add_example(dnn_semantic_segmentation_ex)
      add_example(dnn_instance_segmentation_ex)
      add_example(dnn_yolo_train_ex)
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
   endif()
   if (DLIB_USE_FFMPEG AND NOT DLIB_NO_GUI_SUPPORT)
      add_example(ffmpeg_webcam_face_pose_ex)
      add_example(ffmpeg_video_demuxing_ex)
      add_example(ffmpeg_video_demuxing2_ex)
      add_example(ffmpeg_video_decoding_ex)
      add_example(ffmpeg_video_decoding2_ex)
      add_example(ffmpeg_info_ex)
      add_example(ffmpeg_screen_grab_ex)
      add_example(ffmpeg_video_encoding_ex)
      add_example(ffmpeg_video_muxing_ex)
      add_example(ffmpeg_rtsp_ex)
      add_example(ffmpeg_microphone_to_file_ex)
      add_example(ffmpeg_file_to_speaker_ex)
   endif()

   if (OpenCV_FOUND AND NOT DLIB_NO_GUI_SUPPORT)
      add_example(webcam_face_pose_ex)
      target_include_directories(webcam_face_pose_ex PRIVATE ${OpenCV_INCLUDE_DIRS})
      target_link_libraries(webcam_face_pose_ex ${OpenCV_LIBS} )
   endif()
endif()

# Tools
if (DLIB_BUILD_TOOLS)
   if (NOT DLIB_NO_GUI_SUPPORT)
      add_executable(imglab
         ../tools/imglab/src/main.cpp
         ../tools/imglab/src/metadata_editor.h
         ../tools/imglab/src/metadata_editor.cpp
         ../tools/imglab/src/convert_pascal_xml.h
         ../tools/imglab/src/convert_pascal_xml.cpp
         ../tools/imglab/src/convert_pascal_v1.h
         ../tools/imglab/src/convert_pascal_v1.cpp
         ../tools/imglab/src/convert_idl.h
         ../tools/imglab/src/convert_idl.cpp
         ../tools/imglab/src/common.h
         ../tools/imglab/src/common.cpp
         ../tools/imglab/src/cluster.cpp
         ../tools/imglab/src/flip_dataset.cpp)
      target_link_libraries(imglab PRIVATE dlib::dlib)
   endif()

   add_executable(htmlify 
      ../tools/htmlify/htmlify.cpp
      ../tools/htmlify/to_xml.cpp)
   target_link_libraries(htmlify PRIVATE dlib::dlib)

   add_executable(dtoc
      ../tools/convert_dlib_nets_to_caffe/main.cpp)
   target_link_libraries(dtoc PRIVATE dlib::dlib)
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

if (DLIB_BUILD_PYTHON)
   if (CMAKE_LIBRARY_OUTPUT_DIRECTORY) 
      configure_file(../tools/python/dlib/__init__.py.in ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/dlib/__init__.py)
   endif()
endif()

if (DLIB_BUILD_TOOLS)
   if (NOT DLIB_NO_GUI_SUPPORT)
      install(TARGETS imglab RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}  COMPONENT Tools)
      install(PROGRAMS 
         ../tools/imglab/convert_imglab_paths_to_relative 
         ../tools/imglab/copy_imglab_dataset 
         DESTINATION ${CMAKE_INSTALL_BINDIR}
         COMPONENT Tools)
   endif()
   install(TARGETS htmlify RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Tools)
   INSTALL(TARGETS dtoc    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Tools)
endif()

if (DLIB_BUILD_MATLAB)
   set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ${CMAKE_INSTALL_PREFIX})
   include(InstallRequiredSystemLibraries)
   install(TARGETS example_mex_function DESTINATION . COMPONENT Matlab)
   install(TARGETS example_mex_callback DESTINATION . COMPONENT Matlab)
   install(TARGETS example_mex_struct   DESTINATION . COMPONENT Matlab)
   install(TARGETS example_mex_class    DESTINATION . COMPONENT Matlab)
endif()
