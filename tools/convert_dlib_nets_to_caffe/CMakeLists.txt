cmake_minimum_required(VERSION 3.17.0)

project(dtoc)
add_subdirectory(../../dlib dlib_build)
add_executable(dtoc main.cpp)
target_link_libraries(dtoc dlib::dlib)
include(GNUInstallDirs)
install(TARGETS dtoc RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})