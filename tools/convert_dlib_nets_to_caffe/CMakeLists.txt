#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#

cmake_minimum_required(VERSION 3.10.0)

set (target_name dtoc)

PROJECT(${target_name})

add_subdirectory(../../dlib dlib_build)

add_executable(${target_name} 
   main.cpp
   )

target_link_libraries(${target_name} dlib::dlib )


INSTALL(TARGETS ${target_name}
	RUNTIME DESTINATION bin
	)


