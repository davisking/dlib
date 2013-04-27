
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

include(../../dlib/add_python_module)

add_python_module(dlib 
   src/dlib.cpp 
   src/matrix.cpp 
   src/vector.cpp
   src/svm_c_trainer.cpp
   src/decision_functions.cpp
   src/other.cpp
   src/basic.cpp
   )
