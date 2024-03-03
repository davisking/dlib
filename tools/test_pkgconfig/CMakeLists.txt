cmake_minimum_required(VERSION 3.17.0)

project(Tests)

find_package(PkgConfig REQUIRED)
pkg_check_modules(DLIB IMPORTED_TARGET dlib)

add_executable(assignment_learning_ex ../../examples/assignment_learning_ex.cpp)
target_link_libraries(assignment_learning_ex PkgConfig::DLIB)