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
  pip uninstall numpy -y
  python setup.py test --clean
fi

