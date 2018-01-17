#!/usr/bin/env bash
# Exit if anything fails.
set -eux

# build dlib and tests
if [ "$VARIANT" = "test" ]; then
  mkdir build
  cd build
  ../cmake/bin/cmake ../dlib/test -DCMAKE_BUILD_TYPE=Release
  ../cmake/bin/cmake --build . --target dtest -- -j 2
  ./dtest --runall
fi

if [ "$VARIANT" = "examples" ]; then
  mkdir build
  cd build
  ../cmake/bin/cmake ../examples -DCMAKE_BUILD_TYPE=Release
  ../cmake/bin/cmake --build . -- -j 1
fi

if [ "$VARIANT" = "python-api" ]; then
  python setup.py test --clean
  pip install --user numpy
  python setup.py test --clean
fi

if [ "$VARIANT" = "python3-api" ]; then
  pip3 install -U setuptools
  python3 setup.py test --clean
  pip install --user numpy
  python3 setup.py test --clean
fi
