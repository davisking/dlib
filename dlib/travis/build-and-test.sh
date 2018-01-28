#!/usr/bin/env bash
# Exit if anything fails.
set -eux

# execute the contents of MATRIX_EVAL if it's set
if [[ -v MATRIX_EVAL ]]; then
    eval "${MATRIX_EVAL}"
fi

# build dlib and tests
if [ "$VARIANT" = "test" ]; then
  mkdir build
  cd build
  cmake ../dlib/test 
  cmake --build . --target dtest -- -j 2
  ./dtest --runall
fi

if [ "$VARIANT" = "dlib_all_source_cpp" ]; then
  mkdir build
  cd build
  cmake ../dlib/test 
  cmake --build . --target dlib_all_source_cpp -- -j 2
fi

if [ "$VARIANT" = "tools" ]; then
  mkdir build
  cd build
  cmake ../dlib/test/tools 
  cmake --build .  -- -j 2
fi

if [ "$VARIANT" = "examples" ]; then
  mkdir build
  cd build
  cmake ../examples 
  cmake --build . -- -j 1
fi

if [ "$VARIANT" = "python-api" ]; then
  python setup.py test --clean
  pip uninstall numpy -y
  python setup.py test --clean
fi

