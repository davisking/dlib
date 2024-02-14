cmake_minimum_required(VERSION 3.17.0)

project(mex_functions)

# Dependendices
find_package(Matlab REQUIRED)
set(DLIB_USE_CUDA OFF CACHE BOOL "" FORCE)      # Using CUDA in matlab is a bit flaky
set(DLIB_NO_GUI_SUPPORT ON CACHE BOOL "" FORCE) # Don't build GUI stuff
add_subdirectory(.. dlib_build)

# Helper macro
macro(add_mex_function name)
    matlab_add_mex(NAME ${name} MODULE SRC ${name}.cpp LINK_TO dlib)
    target_compile_definitions(${name} PRIVATE MEX_FILENAME=${name})
endmacro()

add_mex_function(example_mex_function)
add_mex_function(example_mex_callback)
add_mex_function(example_mex_struct)
add_mex_function(example_mex_class)

set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION ${CMAKE_INSTALL_PREFIX})
include(InstallRequiredSystemLibraries)
install(TARGETS example_mex_function DESTINATION . COMPONENT Matlab)
install(TARGETS example_mex_callback DESTINATION . COMPONENT Matlab)
install(TARGETS example_mex_struct   DESTINATION . COMPONENT Matlab)
install(TARGETS example_mex_class    DESTINATION . COMPONENT Matlab)