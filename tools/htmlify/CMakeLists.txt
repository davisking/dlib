#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#

cmake_minimum_required(VERSION 3.10.0)

# create a variable called target_name and set it to the string "htmlify"
set (target_name htmlify)

project(${target_name})

add_subdirectory(../../dlib dlib_build)

# add all the cpp files we want to compile to this list.  This tells
# cmake that they are part of our target (which is the executable named htmlify)
add_executable(${target_name} 
   htmlify.cpp
   to_xml.cpp
   )

# Tell cmake to link our target executable to dlib.
target_link_libraries(${target_name} dlib::dlib )



install(TARGETS ${target_name}
	RUNTIME DESTINATION bin
	)


