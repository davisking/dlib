CMAKE_MINIMUM_REQUIRED(VERSION 3.10.0)

project(dlib_python_bindings)

# Detect if building for Android
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    message(STATUS "Building dlib Python bindings for Android")

    # Paths to Python include and library (from environment)
    if(NOT DEFINED ENV{PYTHON_INCLUDE_DIR})
        message(FATAL_ERROR "PYTHON_INCLUDE_DIR environment variable not set")
    endif()
    if(NOT DEFINED ENV{PYTHON_LIBRARY})
        message(FATAL_ERROR "PYTHON_LIBRARY environment variable not set")
    endif()

    set(PYTHON_INCLUDE_DIR $ENV{PYTHON_INCLUDE_DIR})
    set(PYTHON_LIBRARY $ENV{PYTHON_LIBRARY})

    # Paths to pybind11 include
    if(NOT DEFINED ENV{PYBIND11_INCLUDE_DIR})
        message(FATAL_ERROR "PYBIND11_INCLUDE_DIR environment variable not set")
    endif()
    set(PYBIND11_INCLUDE_DIR $ENV{PYBIND11_INCLUDE_DIR})

    include_directories(${PYTHON_INCLUDE_DIR} ${PYBIND11_INCLUDE_DIR})
else()
    # Desktop build: include pybind11 normally
    add_subdirectory(../../dlib/external/pybind11 pybind11_build)
    #include_directories(${PYBIND11_INCLUDE_DIR})
endif()

# Add dlib core
add_subdirectory(../../dlib dlib_build)
add_definitions(-DDLIB_VERSION=${DLIB_VERSION})

# Source files
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
   src/line.cpp
)

if(NOT ${DLIB_NO_GUI_SUPPORT})
   list(APPEND python_srcs src/gui.cpp)
endif()

# --- Module target ---
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    # Android: build shared library manually
    add_library(_dlib_pybind11 SHARED ${python_srcs})
    target_link_libraries(_dlib_pybind11 PRIVATE dlib::dlib ${PYTHON_LIBRARY})
else()
    # Desktop: use pybind11 helper
    pybind11_add_module(_dlib_pybind11 ${python_srcs})
    target_link_libraries(_dlib_pybind11 PRIVATE dlib::dlib)
endif()

# Copy __init__.py for Python package
if (CMAKE_LIBRARY_OUTPUT_DIRECTORY) 
   configure_file(${PROJECT_SOURCE_DIR}/dlib/__init__.py.in ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/dlib/__init__.py)
endif()
