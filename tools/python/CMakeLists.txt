
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)


set(USE_SSE4_INSTRUCTIONS ON CACHE BOOL "Use SSE4 instructions")
# Make DLIB_ASSERT statements not abort the python interpreter, but just return an error.
add_definitions(-DDLIB_NO_ABORT_ON_2ND_FATAL_ERROR)

include(../../dlib/cmake_utils/add_python_module)

# Test for numpy
FIND_PACKAGE(PythonInterp)
IF(PYTHONINTERP_FOUND)
    EXECUTE_PROCESS( COMMAND ${PYTHON_EXECUTABLE} -c "import numpy" OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE NUMPYRC)
    IF(NUMPYRC EQUAL 1)
        MESSAGE(WARNING "Numpy not found. Functions that return numpy arrays will throw exceptions!")
    ELSE(NUMPYRC EQUAL 1)
        MESSAGE(STATUS "Found Python with installed numpy package")
        EXECUTE_PROCESS( COMMAND ${PYTHON_EXECUTABLE} -c "import sys; from numpy import get_include; sys.stdout.write(get_include())" OUTPUT_VARIABLE NUMPY_INCLUDE_PATH)
        MESSAGE(STATUS "Numpy include path '${NUMPY_INCLUDE_PATH}'")
        include_directories(${NUMPY_INCLUDE_PATH})
    ENDIF(NUMPYRC EQUAL 1)
ELSE(PYTHONINTERP_FOUND)
    MESSAGE(FATAL_ERROR "Could not find Python interpreter")
ENDIF(PYTHONINTERP_FOUND)

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
IF(NUMPYRC EQUAL 1)
    list(APPEND python_srcs src/numpy_returns_stub.cpp)
ELSE(NUMPYRC EQUAL 1)
    list(APPEND python_srcs src/numpy_returns.cpp)
ENDIF(NUMPYRC EQUAL 1)

if(NOT ${DLIB_NO_GUI_SUPPORT})
   list(APPEND python_srcs src/gui.cpp)
endif(NOT ${DLIB_NO_GUI_SUPPORT})

# Only add the GUI module if requested
if(NOT ${DLIB_NO_GUI_SUPPORT})
   list(APPEND python_srcs src/gui.cpp)
endif(NOT ${DLIB_NO_GUI_SUPPORT})

add_python_module(dlib ${python_srcs})

# When you run "make install" we will copy the compiled dlib.so (or dlib.pyd)
# library file to the python_examples folder.
install_dlib_to(../../python_examples)
