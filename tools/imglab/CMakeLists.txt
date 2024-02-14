cmake_minimum_required(VERSION 3.17.0)

project(imglab)
add_subdirectory(../../dlib dlib_build)

add_executable(imglab 
   src/main.cpp
   src/metadata_editor.h
   src/metadata_editor.cpp
   src/convert_pascal_xml.h
   src/convert_pascal_xml.cpp
   src/convert_pascal_v1.h
   src/convert_pascal_v1.cpp
   src/convert_idl.h
   src/convert_idl.cpp
   src/common.h
   src/common.cpp
   src/cluster.cpp
   src/flip_dataset.cpp
)

target_link_libraries(imglab dlib::dlib)
include(GNUInstallDirs)
install(TARGETS imglab RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(PROGRAMS convert_imglab_paths_to_relative copy_imglab_dataset DESTINATION ${CMAKE_INSTALL_BINDIR} )