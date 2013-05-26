
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

include(../../dlib/add_python_module)

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
   )

install_dlib_to(../../python_examples)
