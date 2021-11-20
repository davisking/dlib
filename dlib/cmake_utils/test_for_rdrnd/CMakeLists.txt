
cmake_minimum_required(VERSION 2.8.12)
project(rdrnd_test)

set(USE_RDRND_INSTRUCTIONS ON CACHE BOOL "Use RDRND instructions")

# Pull this in since it sets the RDRND compile options by putting that kind of stuff into the active_compile_opts list.
include(../set_compiler_specific_options.cmake)


try_run(run_result compile_result ${PROJECT_BINARY_DIR}/rdrnd_test_try_run_build ${CMAKE_CURRENT_LIST_DIR}/rdrnd_test.cpp
   COMPILE_DEFINITIONS ${active_compile_opts})

message(STATUS "run_result = ${run_result}")
message(STATUS "compile_result = ${compile_result}")

if ("${run_result}" EQUAL 0 AND compile_result)
   message(STATUS "Ran RDRND test program successfully, you have RDRND available.")
else()
   message(STATUS "Unable to run RDRND test program, you don't seem to have RDRND instructions available.")
   # make this build fail so that calling try_compile statements will error in this case.
   add_library(make_this_build_fail ${CMAKE_CURRENT_LIST_DIR}/this_file_doesnt_compile.cpp)
endif()
