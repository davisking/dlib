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

CXXFLAGSALL="-DFUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION= -g"
CMAKEFLAGSALL="$root -GNinja -DCMAKE_BUILD_TYPE=Debug"

#builds fuzzers for local fuzzing with libfuzzer with asan+usan
builddir=$here/build-fuzzers-libfuzzer
mkdir -p $builddir
cd $builddir
CXX="clang++" \
CXXFLAGS="$CXXFLAGSALL -fsanitize=fuzzer-no-link,address,undefined" \
cmake $CMAKEFLAGSALL -DCMAKE_BUILD_TYPE=Debug $fuzzdir

echo $me: building
cmake --build $builddir


echo $me: all good

