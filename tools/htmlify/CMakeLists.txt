cmake_minimum_required(VERSION 3.17.0)

project(htmlify)
add_subdirectory(../../dlib dlib_build)
add_executable(htmlify htmlify.cpp to_xml.cpp)   
target_link_libraries(htmlify dlib::dlib )
include(GNUInstallDirs)
install(TARGETS htmlify RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})