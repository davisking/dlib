#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#


cmake_minimum_required(VERSION 3.10.0)
PROJECT(archive)


add_subdirectory(../../dlib dlib_build)

add_executable(train_face_5point_model train_face_5point_model.cpp)
target_link_libraries(train_face_5point_model dlib::dlib)

