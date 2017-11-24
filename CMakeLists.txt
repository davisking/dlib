cmake_minimum_required(VERSION 2.8.12)



get_directory_property(has_parent PARENT_DIRECTORY)
if(NOT has_parent)
   # Set this so that the dlib subfolder will build both a static and shared
   # library, since the only reason to build this CMakeLists.txt by itself is
   # if you want to install dlib.  It should be noted however that installing
   # dlib is not necessary.  A simpler approach is to use it the way shown in
   # examples/CMakeLists.txt 
   set(DLIB_IN_PROJECT_BUILD false)
endif()

add_subdirectory(dlib)
