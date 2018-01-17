#!/usr/bin/env bash
# Exit if anything fails.
set -eux

# build dlib and tests
mkdir build
cd build
if [ "$VARIANT" = "test" ]; then
  ../cmake/bin/cmake ../dlib/test -DCMAKE_BUILD_TYPE=Release
  ../cmake/bin/cmake --build . --target dtest -- -j 2
  ./dtest --runall
fi
if [ "$VARIANT" = "examples" ]; then
  ../cmake/bin/cmake ../examples -DCMAKE_BUILD_TYPE=Release
  ../cmake/bin/cmake --build . -- -j 1
fi

if [ "$VARIANT" = "python-api" ]; then
  python setup.py test --clean
  pip install --user numpy
  python setup.py test --clean
fi

if [ "$VARIANT" = "python3-api" ]; then
  python3 setup.py test --clean
  pip install --user numpy
  python3 setup.py test --clean
fi
