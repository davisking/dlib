# dlib C++ library [![Travis Status](https://travis-ci.org/davisking/dlib.svg?branch=master)](https://travis-ci.org/davisking/dlib)

Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating complex software in C++ to solve real world problems. See [http://dlib.net](http://dlib.net) for the main project documentation and API reference.

Some common questions are answered in [FAQ](http://dlib.net/faq.html). The best way to get support is using [StackOverflow](http://stackoverflow.com/questions/tagged/dlib).

## Installing dlib as a shared library into your system

Dlib is designed to be compiled with [CMake](https://cmake.org/). After cloning dlib from git or downloading it's sources, go into dlib folder and type:

```bash
mkdir build
cd build
cmake .. 
cmake --build . --target install
```

On some Linux systems last command should be run as `sudo` to get access to /usr/local folder:
```
sudo cmake --build . --target install
```

## Using dlib in your project
After dlib is installed in your system, you can now use it with CMake like this:
```CMake
cmake_minimum_required(VERSION 2.8.4)
find_package(dlib REQUIRED)

include_directories(${dlib_INCLUDE_DIRS})
link_directories(${DLIB_LIBRARY_DIRS})

add_executable(dlibsample dlibsample.cpp)
target_link_libraries(dlibsample ${dlib_LIBRARIES})
```

Some additional ways to compile and use dlib are described in [Documentation](http://dlib.net/compile.html)

## Compiling dlib C++ example programs

Go into the examples folder and type:

```bash
mkdir build; cd build; cmake .. ; cmake --build .
```

That will build all the examples.
If you have a CPU that supports AVX instructions then turn them on like this:

```bash
mkdir build; cd build; cmake .. -DUSE_AVX_INSTRUCTIONS=1; cmake --build .
```

Doing so will make some things run faster.



## Compiling dlib Python API

Before you can run the Python example programs you must compile dlib. Type:

```bash
python setup.py install
```

or type

```bash
python setup.py install --yes USE_AVX_INSTRUCTIONS
```

if you have a CPU that supports AVX instructions, since this makes some things run faster.  Note that you need to have boost-python installed to compile the Python API.



## Running the unit test suite

Type the following to compile and run the dlib unit test suite:

```bash
cd dlib/test
mkdir build
cd build
cmake ..
cmake --build . --config Release
./dtest --runall
```

Note that on windows your compiler might put the test executable in a subfolder called `Release`. If that's the case then you have to go to that folder before running the test.

This library is licensed under the Boost Software License, which can be found in [dlib/LICENSE.txt](https://github.com/davisking/dlib/blob/master/dlib/LICENSE.txt).  The long and short of the license is that you can use dlib however you like, even in closed source commercial software.

## dlib sponsors

This research is based in part upon work supported by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA) under contract number 2014-14071600010. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of ODNI, IARPA, or the U.S. Government.

