#!/bin/sh
#
# Copyright (c) 2019 Paul Dreik
#

set -e
me=$(basename $0)
fuzzdir=$(readlink -f "$(dirname "$0")")
root=$(readlink -f "$(dirname "$0")/..")


echo $me: root=$root

here=$(pwd)

CXXFLAGSALL="-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION= -g -std=c++1z -O3"
CMAKEFLAGSALL="$root -GNinja -DCMAKE_BUILD_TYPE=Debug"

#builds fuzzers for local fuzzing with libfuzzer with asan+usan
builddir=$here/build-fuzzers-libfuzzer
mkdir -p $builddir
cd $builddir
CXX="clang++-7" \
CXXFLAGS="$CXXFLAGSALL -fsanitize=fuzzer-no-link,address,undefined" \
cmake $CMAKEFLAGSALL -DCMAKE_BUILD_TYPE=Debug $fuzzdir

echo $me: building
cmake --build $builddir



#builds fuzzers for local fuzzing with libfuzzer without sanitizers
builddir=$here/build-fuzzers-fast
mkdir -p $builddir
cd $builddir
CXX="clang++-7" \
CXXFLAGS="$CXXFLAGSALL -fsanitize=fuzzer-no-link" \
cmake $CMAKEFLAGSALL -DCMAKE_BUILD_TYPE=Release $fuzzdir

echo $me: building
cmake --build $builddir


echo $me: all good

