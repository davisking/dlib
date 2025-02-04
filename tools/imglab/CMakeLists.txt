#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#

cmake_minimum_required(VERSION 3.10.0)

# create a variable called target_name and set it to the string "imglab"
set (target_name imglab)

PROJECT(${target_name})
add_subdirectory(../../dlib dlib_build)

# add all the cpp files we want to compile to this list.  This tells
# cmake that they are part of our target (which is the executable named imglab)
add_executable(${target_name} 
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


# Tell cmake to link our target executable to dlib.  
target_link_libraries(${target_name} dlib::dlib )


install(TARGETS ${target_name}
	RUNTIME DESTINATION bin
	)
install(PROGRAMS convert_imglab_paths_to_relative copy_imglab_dataset DESTINATION bin )

