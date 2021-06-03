cmake_minimum_required(VERSION 2.8.12)
# Every project needs a name.  We call this the "examples" project.
project(SealNet)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Tell cmake we will need dlib.  This command will pull in dlib and compile it
# into your project.  Note that you don't need to compile or install dlib.  All
# cmake needs is the dlib source code folder and it will take care of everything.
# add_subdirectory(../dlib dlib_build)


# cmake --build . --config Release

if (NOT DEFINED DLIB_PATH)
   set(DLIB_PATH "./")
endif()

include(${DLIB_PATH}/dlib/cmake)

# If you have cmake 3.14 or newer you can even use FetchContent instead of
# add_subdirectory() to pull in dlib as a dependency.  So instead of using the
# above add_subdirectory() command, you could use the following three commands
# to make dlib available:
#  include(FetchContent)
#  FetchContent_Declare(dlib
#      GIT_REPOSITORY https://github.com/davisking/dlib.git
#      GIT_TAG        v19.18
#  )
#  FetchContent_MakeAvailable(dlib)


# The next thing we need to do is tell CMake about the code you want to
# compile.  We do this with the add_executable() statement which takes the name
# of the output executable and then a list of .cpp files to compile.  Here we
# are going to compile one of the dlib example programs which has only one .cpp
# file, assignment_learning_ex.cpp.  If your program consisted of multiple .cpp
# files you would simply list them here in the add_executable() statement.  


add_executable(seal dnn_mmod_dog_hipsterizer.cpp)


# Finally, you need to tell CMake that this program, assignment_learning_ex,
# depends on dlib.  You do that with this statement: 


target_link_libraries(seal dlib::dlib)