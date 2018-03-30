#!/usr/bin/env bash




# Make sure the binaries are there, if not then delete the directory and redownload
./cmake/2.8/bin/cmake --version || rm -rf cmake
./cmake/3.1/bin/cmake --version || rm -rf cmake
./cmake/3.5/bin/cmake --version || rm -rf cmake


# Exit if anything fails.
set -eux

if [[ ! -d cmake ]]; then
  echo "Downloading cmake..."

  # Travis requires 64bit binaries but they aren't available for this version of cmake, so we build from source
  CMAKE_URL="https://cmake.org/files/v2.8/cmake-2.8.12.1.tar.gz"
  mkdir -p cmake/2.8
  wget --no-check-certificate -O - ${CMAKE_URL} | tar --strip-components=1 -xz -C cmake/2.8
  pushd cmake/2.8
  ./configure
  make -j2
  popd

  CMAKE_URL="http://www.cmake.org/files/v3.1/cmake-3.1.2-Linux-x86_64.tar.gz"
  mkdir -p cmake/3.1
  wget --no-check-certificate -O - ${CMAKE_URL} | tar --strip-components=1 -xz -C cmake/3.1

  CMAKE_URL="http://www.cmake.org/files/v3.5/cmake-3.5.2-Linux-x86_64.tar.gz"
  mkdir -p cmake/3.5
  wget --no-check-certificate -O - ${CMAKE_URL} | tar --strip-components=1 -xz -C cmake/3.5

fi


#make sure the binaries are really there
./cmake/2.8/bin/cmake --version
./cmake/3.1/bin/cmake --version 
./cmake/3.5/bin/cmake --version 

