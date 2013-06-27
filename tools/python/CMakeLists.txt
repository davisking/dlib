
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

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
   )

# When you run "make install" we will copy the compiled dlib.so (or dlib.pyd) 
# library file to the python_examples folder.
install_dlib_to(../../python_examples)
