cmake_minimum_required(VERSION 2.8.12)




#############################################################################
#                                                                           #
#  READ examples/CMakeLists.txt TO SEE HOW TO USE DLIB FROM C++ WITH CMAKE  #
#                                                                           #
#############################################################################





get_directory_property(has_parent PARENT_DIRECTORY)
if(NOT has_parent)
   # When you call add_subdirectory(dlib) from a parent CMake project dlib's
   # CMake scripts will assume you want to statically compile dlib into
   # whatever you are building rather than create a standalone copy of dlib.
   # This means CMake will build dlib as a static library, disable dlib's
   # install targets so they don't clutter your project, and adjust a few other
   # minor things that are convenient when statically building dlib as part of
   # your own projects.
   #
   # On the other hand, if there is no parent CMake project or if
   # DLIB_IN_PROJECT_BUILD is set to false, CMake will compile dlib as a normal
   # standalone library (either shared or static, based on the state of CMake's
   # BUILD_SHARED_LIBS flag), and include the usual install targets so you can
   # install dlib on your computer via `make install`.  Since the only reason
   # to build this CMakeLists.txt (the one you are reading right now) by itself
   # is if you want to install dlib, we indicate as such by setting
   # DLIB_IN_PROJECT_BUILD to false.
   set(DLIB_IN_PROJECT_BUILD false)
endif()
add_subdirectory(dlib)
