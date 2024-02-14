
cmake_minimum_required(VERSION 3.17.0)

project(dlib_python_bindings)

add_subdirectory(../../dlib dlib_build)
add_subdirectory(../../dlib/external/pybind11 pybind11_build)

pybind11_add_module(_dlib_pybind11 
   src/dlib.cpp
   src/matrix.cpp
   src/vector.cpp
   src/svm_c_trainer.cpp
   src/svm_rank_trainer.cpp
   src/decision_functions.cpp
   src/other.cpp
   src/basic.cpp
   src/cca.cpp
   src/sequence_segmenter.cpp
   src/svm_struct.cpp
   src/image.cpp
   src/image2.cpp
   src/image3.cpp
   src/image4.cpp
   src/rectangles.cpp
   src/object_detection.cpp
   src/shape_predictor.cpp
   src/correlation_tracker.cpp
   src/face_recognition.cpp
   src/cnn_face_detector.cpp
   src/global_optimization.cpp
   src/image_dataset_metadata.cpp
   src/numpy_returns.cpp
   src/line.cpp)
   
if(NOT DLIB_NO_GUI_SUPPORT)
   target_sources(_dlib_pybind11 PRIVATE src/gui.cpp)
endif()

target_link_libraries(_dlib_pybind11 PRIVATE dlib::dlib)
target_compile_definitions(_dlib_pybind11 PRIVATE DLIB_NO_ABORT_ON_2ND_FATAL_ERROR)
set_target_properties(_dlib_pybind11 PROPERTIES
   INTERPROCEDURAL_OPTIMIZATION ON
   CXX_VISIBILITY_PRESET "hidden"
   CUDA_VISIBILITY_PRESET "hidden")

if (CMAKE_LIBRARY_OUTPUT_DIRECTORY) 
   configure_file(dlib/__init__.py.in ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/dlib/__init__.py)
endif()