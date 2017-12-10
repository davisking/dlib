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
  ../cmake/bin/cmake ../tools/python -DCMAKE_BUILD_TYPE=Release
  ../cmake/bin/cmake --build . --target install -- -j 2
  ../python_examples/svm_rank.py

  rm -rf *
  pip install --user numpy
  ../cmake/bin/cmake ../tools/python -DCMAKE_BUILD_TYPE=Release
  ../cmake/bin/cmake --build . --target install -- -j 2
  ../python_examples/svm_rank.py
fi

if [ "$VARIANT" = "python3-api" ]; then
  ../cmake/bin/cmake ../tools/python -DPYTHON3=ON -DCMAKE_BUILD_TYPE=Release
  ../cmake/bin/cmake --build . --target install -- -j 2
  python3 ../python_examples/svm_rank.py

  pip3 install --user numpy
  rm -rf *
  ../cmake/bin/cmake ../tools/python -DPYTHON3=ON -DCMAKE_BUILD_TYPE=Release
  ../cmake/bin/cmake --build . --target install -- -j 2
  python3 ../python_examples/svm_rank.py
fi
