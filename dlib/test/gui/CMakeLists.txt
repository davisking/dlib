#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#

# create a variable called target_name and set it to the string "test"
set (target_name gui)

project(${target_name})

add_subdirectory(../.. dlib_build)

# add all the cpp files we want to compile to this list.  This tells
# cmake that they are part of our target (which is the executable named test)
add_executable(${target_name} main.cpp )


# Tell cmake to link our target executable to dlib.
target_link_libraries(${target_name} dlib::dlib )

