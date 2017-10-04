#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#

cmake_minimum_required(VERSION 2.8.12)

set (target_name dtoc)

PROJECT(${target_name})

include(../../dlib/cmake)

ADD_EXECUTABLE(${target_name} 
   main.cpp
   )

TARGET_LINK_LIBRARIES(${target_name} dlib::dlib )


INSTALL(TARGETS ${target_name}
	RUNTIME DESTINATION bin
	)


