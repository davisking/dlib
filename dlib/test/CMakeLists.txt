#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#

cmake_minimum_required(VERSION 3.8.0)

# create a variable called target_name and set it to the string "dtest"
set (target_name dtest)
PROJECT(${target_name})

# compile the dlib/all/source.cpp file into its own object just to make sure it compiles
set(DLIB_TEST_COMPILE_ALL_SOURCE_CPP ON)

add_subdirectory(.. dlib_build)

add_executable(${target_name} 
)

get_filename_component(DLIB_FFMPEG_DATA ${CMAKE_SOURCE_DIR}/ffmpeg_data/details.cfg REALPATH)
target_compile_definitions(${target_name} PRIVATE DLIB_FFMPEG_DATA="${DLIB_FFMPEG_DATA}")

if (CMAKE_COMPILER_IS_GNUCXX)
   # Turn on all warnings, and treat them as errors.
   add_compile_options(-W -Wall -Wextra -Wpedantic -Werror)
   add_compile_options(-fdiagnostics-color=always)
   # I don't care about unused testing functions though.  I like to keep them
   # around.  Don't warn about it.
   add_compile_options(-Wno-unused-function)
   add_compile_options(-Wno-strict-overflow)
   add_compile_options(-Wno-maybe-uninitialized)

   if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11.3.0 AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS_EQUAL 11.4.0)
      # These versions of gcc wrongly yield this warning:
      #  dlib::row_major_layout::layout<double, 0, 2, dlib::memory_manager_stateless_kernel_1<char> >::data’ with nonzero offset 8 [-Werror=free-nonheap-object]
      #   61 |                 delete [] item;
      # Which by inspection of the dlib::row_major_layout::layout class you can see is clearly incorrect, as `data` is most
      # definitely heap allocated.
      add_compile_options(-Wno-free-nonheap-object)
   endif()


elseif (MSVC)
   # Treat warnings as errors.
   add_compile_options(/WX)
else() # basically Clang
   # Treat warnings as errors, but do not turn on all warnings.
   add_compile_options(-W -Werror)
   # This is for the comment in face_detection_ex.cpp that says "faces/*.jpg"
   add_compile_options(-Wno-comment)

   if(${CMAKE_VERSION} VERSION_GREATER "3.8.0") 
      # strip debug symbols to make the binary smaller
      target_link_options(${target_name} PRIVATE $<$<CONFIG:RELEASE>:-s>)
   endif()
endif()

if (NOT DLIB_NO_GUI_SUPPORT)
   add_subdirectory(gui)
   add_subdirectory(tools)

   if (CMAKE_COMPILER_IS_GNUCXX)
      # The tutorials set some variables that are not used.
      add_definitions("-Wno-unused-but-set-variable")
   endif()

   add_subdirectory(examples)
endif()
