#!/usr/bin/env bash
# Exit if anything fails.
set -eux

# build dlib and tests
if [ "$VARIANT" = "test" ]; then
  mkdir build
  cd build
  cmake ../dlib/test -DCMAKE_BUILD_TYPE=Release
  cmake --build . --target dtest -- -j 2
  ./dtest --runall
fi

if [ "$VARIANT" = "examples" ]; then
  mkdir build
  cd build
  cmake ../examples -DCMAKE_BUILD_TYPE=Release
  cmake --build . -- -j 1
fi

if [ "$VARIANT" = "python-api" ]; then
  python setup.py test --clean
  pip uninstall numpy
  python setup.py test --clean
fi

