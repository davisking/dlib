
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)


set(USE_SSE4_INSTRUCTIONS ON CACHE BOOL "Use SSE4 instructions")
# Make DLIB_ASSERT statements not abort the python interpreter, but just return an error.
add_definitions(-DDLIB_NO_ABORT_ON_2ND_FATAL_ERROR)

include(../../dlib/cmake_utils/add_python_module)

# Test for numpy
find_package(PythonInterp)
if(PYTHONINTERP_FOUND)
   execute_process( COMMAND ${PYTHON_EXECUTABLE} -c "import numpy" OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE NUMPYRC)
   if(NUMPYRC EQUAL 1)
      message(WARNING "Numpy not found. Functions that return numpy arrays will throw exceptions!")
   else()
      message(STATUS "Found Python with installed numpy package")
      execute_process( COMMAND ${PYTHON_EXECUTABLE} -c "import sys; from numpy import get_include; sys.stdout.write(get_include())" OUTPUT_VARIABLE NUMPY_INCLUDE_PATH)
      message(STATUS "Numpy include path '${NUMPY_INCLUDE_PATH}'")
      include_directories(${NUMPY_INCLUDE_PATH})
   endif()
else()
   message(WARNING "Numpy not found. Functions that return numpy arrays will throw exceptions!")
   set(NUMPYRC 1)
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
)

# Only add the Numpy returning functions if Numpy is present
if(NUMPYRC EQUAL 1)
   list(APPEND python_srcs src/numpy_returns_stub.cpp)
else()
   list(APPEND python_srcs src/numpy_returns.cpp)
endif()

# Only add the GUI module if requested
if(NOT ${DLIB_NO_GUI_SUPPORT})
   list(APPEND python_srcs src/gui.cpp)
endif()

add_python_module(dlib ${python_srcs})

# When you run "make install" we will copy the compiled dlib.so (or dlib.pyd)
# library file to the python_examples folder.
install_dlib_to(../../python_examples)
