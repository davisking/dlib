
cmake_minimum_required (VERSION 2.8.12)
project (myproject)
set(java_package_name net.dlib)
set(source_files
   )

include(../cmake_utils/use_cpp_11.cmake)
include(../cmake_utils/release_build_by_default)

include_directories(
   .
   )

# Additional dependencies
#include(../../dlib/cmake)
#set(additional_link_libraries dlib)

# tell swig to put the output files into the local folder.
set(install_target_output_folder .)


include(cmake_swig_jni)


