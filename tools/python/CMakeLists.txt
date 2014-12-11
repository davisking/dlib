
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.4)


# Uncomment this to make a Python 3 compatible library instead of a Python 2
# library.
#set(PYTHON3 true)

include(../../dlib/add_python_module)

# Tell cmake to compile all these cpp files into a dlib python module.
add_python_module(dlib 
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
   src/object_detection.cpp
   src/shape_predictor.cpp
   src/gui.cpp
   )

# When you run "make install" we will copy the compiled dlib.so (or dlib.pyd) 
# library file to the python_examples folder.
install_dlib_to(../../python_examples)
