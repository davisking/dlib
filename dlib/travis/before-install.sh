#!/usr/bin/env bash

# Exit if anything fails.
set -eux

## download CMAKE 3.5 to get colored output
if [[ ! -x cmake/bin/cmake && -d cmake ]]; then
    rm -rf cmake
fi
if [[ ! -d cmake ]]; then
  CMAKE_URL="http://www.cmake.org/files/v3.5/cmake-3.5.2-Linux-x86_64.tar.gz"
  mkdir -v cmake 
  wget --no-check-certificate -O - ${CMAKE_URL} | tar --strip-components=1 -xz -C cmake
fi



