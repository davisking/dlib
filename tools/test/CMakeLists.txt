cmake_minimum_required(VERSION 3.17.0)

project(Tests)

find_package(dlib REQUIRED)

add_executable(assignment_learning_ex ../../examples/assignment_learning_ex.cpp)
target_link_libraries(assignment_learning_ex dlib::dlib)