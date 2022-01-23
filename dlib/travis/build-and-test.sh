#!/usr/bin/env bash
# Exit if anything fails.
set -eux

# execute the contents of MATRIX_EVAL if it's set
if [ -n "${MATRIX_EVAL+set}" ]; then
    eval "${MATRIX_EVAL}"
fi

CXX_FLAGS="-std=c++11"
if [ ! -z ${CXXFLAGS+set} ]; then
    CXX_FLAGS="${CXXFLAGS}"
fi

# build dlib and tests
if [ "$VARIANT" = "test" ]; then
  mkdir build
  cd build
  cmake ../dlib/test -DCMAKE_CXX_FLAGS="${CXX_FLAGS}"
  cmake --build . --target dtest -- -j 2
  ./dtest --runall $DISABLED_TESTS
fi

# build dlib and tests
if [ "$VARIANT" = "test-debug" ]; then
  mkdir build
  cd build
  cmake ../dlib/test -DDLIB_ENABLE_ASSERTS=1 -DCMAKE_CXX_FLAGS="${CXX_FLAGS}"
  cmake --build . --target dtest -- -j 2
  ./dtest --runall $DISABLED_TESTS
fi

# The point of this test is just to make sure the cmake scripts work with the
# oldest version of cmake we are supposed to support.
if [ "$VARIANT" = "old-cmake" ]; then
  mkdir build
  cd build
  CMAKEDIR=../cmake

  $CMAKEDIR/2.8/bin/cmake ../dlib/test/tools
  $CMAKEDIR/2.8/bin/cmake --build .  -- -j 2

  rm -rf *
  $CMAKEDIR/3.1/bin/cmake ../dlib/test/tools
  $CMAKEDIR/3.1/bin/cmake --build .  -- -j 2

  rm -rf *
  $CMAKEDIR/3.5/bin/cmake ../dlib/test/tools
  $CMAKEDIR/3.5/bin/cmake --build .  -- -j 2


  # just to make sure there isn't anything funny about building standalone dlib
  rm -rf *
  $CMAKEDIR/2.8/bin/cmake ../dlib
  $CMAKEDIR/2.8/bin/cmake --build .  -- -j 2
fi

if [ "$VARIANT" = "python-api" ]; then
  python setup.py test --clean
  pip uninstall numpy -y
  python setup.py test --clean
fi
