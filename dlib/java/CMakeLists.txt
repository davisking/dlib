
cmake_minimum_required(VERSION 3.8.0)
project (myproject)
set(java_package_name net.dlib)
set(source_files
   )

include(../cmake_utils/release_build_by_default)

include_directories(
   .
   )

# Additional dependencies
#add_subdirectory(../../dlib dlib_build)
#set(additional_link_libraries dlib::dlib)

# Tell swig to put the output files (the shared library and .jar) into the local folder.
set(install_target_output_folder .)

# Alternatively, instead of using install_target_output_folder, you can tell
# cmake to output the shared library, java source files, and the jar to
# separate output folders.  These commands would put them into folders thelib,
# thesrc, and thejar, respectively.
#set(install_shared_library_output_folder thelib)
#set(install_java_source_output_folder    thesrc)
#set(install_jar_output_folder            thejar)


include(cmake_swig_jni)


