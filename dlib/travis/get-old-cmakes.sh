#!/usr/bin/env bash

# Exit if anything fails.
set -eux

echo "Checking if cmake already downloaded"
if [ ! -x cmake/2.8/bin/cmake ] || [ ! -x cmake/3.1/bin/cmake ] || [ ! -x cmake/3.5/bin/cmake ]; then
    echo "Didn't find it, clearing old cmake folder"
    rm -rf cmake
fi

if [[ ! -d cmake ]]; then
  echo "Downloading cmake..."

  CMAKE_URL="http://www.cmake.org/files/v2.8/cmake-2.8.12-Linux-i386.tar.gz"
  mkdir -p cmake/2.8
  wget --no-check-certificate -O - ${CMAKE_URL} | tar --strip-components=1 -xz -C cmake/2.8

  CMAKE_URL="http://www.cmake.org/files/v3.1/cmake-3.1.2-Linux-x86_64.tar.gz"
  mkdir -p cmake/3.1
  wget --no-check-certificate -O - ${CMAKE_URL} | tar --strip-components=1 -xz -C cmake/3.1

  CMAKE_URL="http://www.cmake.org/files/v3.5/cmake-3.5.2-Linux-x86_64.tar.gz"
  mkdir -p cmake/3.5
  wget --no-check-certificate -O - ${CMAKE_URL} | tar --strip-components=1 -xz -C cmake/3.5

fi
