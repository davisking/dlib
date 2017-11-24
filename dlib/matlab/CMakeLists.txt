
cmake_minimum_required(VERSION 2.8.12)

PROJECT(mex_functions)

include(cmake_mex_wrapper)

add_subdirectory(.. dlib_build)


# You can tell cmake where to put the mex files when you run 'make install' by
# setting this variable.  The path is relative to this CMakeLists.txt file.
set(install_target_output_folder .)

# Compile the example_mex_function.cpp file and link it to dlib.  Note
# that you can give a list of things to link to here.  E.g. 
#    add_mex_function(some_other_mex_function pthread dlib fftw) 
add_mex_function(example_mex_function dlib::dlib)
add_mex_function(example_mex_callback dlib::dlib)
add_mex_function(example_mex_struct dlib::dlib)
add_mex_function(example_mex_class dlib::dlib)

