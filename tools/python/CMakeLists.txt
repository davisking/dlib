
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

set(USE_SSE4_INSTRUCTIONS ON CACHE BOOL "Use SSE4 instructions")

# Set this to disable link time optimization. The only reason for
# doing this to make the compile faster which is nice when developing
# new modules.
# set(PYBIND11_LTO_CXX_FLAGS "")


# Avoid cmake warnings about changes in behavior of some Mac OS X path 
# variable we don't care about.
if (POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()


# To avoid dll hell, always link everything statically when compiling in
# visual studio.  This way, the resulting library won't depend on a bunch
# of other dll files and can be safely copied to someone elese's computer
# and expected to run.
if (MSVC)
    include(${CMAKE_CURRENT_LIST_DIR}/../../dlib/cmake_utils/tell_visual_studio_to_use_static_runtime.cmake)
endif()

add_subdirectory(../../dlib/external/pybind11 ./pybind11_build)
add_subdirectory(../../dlib ./dlib_build)

if (USING_OLD_VISUAL_STUDIO_COMPILER)
   message(FATAL_ERROR "You have to use a version of Visual Studio that supports C++11.  As of December 2017, the only versions that have good enough C++11 support to compile the dlib Python API is a fully updated Visual Studio 2015 or a fully updated Visual Studio 2017.  Older versions of either of these compilers have bad C++11 support and will fail to compile the Python extension. ***SO UPDATE YOUR VISUAL STUDIO TO MAKE THIS ERROR GO AWAY***")
endif()


# Test for numpy
find_package(PythonInterp)
if(PYTHONINTERP_FOUND)
   execute_process( COMMAND ${PYTHON_EXECUTABLE} -c "import numpy" OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE NUMPYRC)
   if(NUMPYRC EQUAL 1)
      message(WARNING "Numpy not found. Functions that return numpy arrays will not work without Numpy installed!")
   else()
      message(STATUS "Found Python with installed numpy package")
   endif()
else()
   message(WARNING "Numpy not found. Functions that return numpy arrays will not work without Numpy installed!")
endif()

add_definitions(-DDLIB_VERSION=${DLIB_VERSION})

# Tell cmake to compile all these cpp files into a dlib python module.
set(python_srcs
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
   src/rectangles.cpp
   src/object_detection.cpp
   src/shape_predictor.cpp
   src/correlation_tracker.cpp
   src/face_recognition.cpp
   src/cnn_face_detector.cpp
   src/global_optimization.cpp
   src/image_dataset_metadata.cpp
   src/numpy_returns.cpp
)

# Only add the GUI module if requested
if(NOT ${DLIB_NO_GUI_SUPPORT})
   list(APPEND python_srcs src/gui.cpp)
endif()

pybind11_add_module(dlib_python ${python_srcs})
target_link_libraries(dlib_python PRIVATE dlib::dlib)
# Set the output library name to dlib because that's what setup.py and distutils expects.
set_target_properties(dlib_python PROPERTIES OUTPUT_NAME dlib)

