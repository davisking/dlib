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
  ../cmake/bin/cmake --build . -- -j 2
fi

cd ..