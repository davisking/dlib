#!/bin/bash

build_img()
{
  cd ${IMG_BUILD}
  cmake ..
  cmake --build . --config Debug
}
build_dnn()
{
  cd ${DNN_BUILD}
  cmake ..
  cmake --build . --config Debug
}

export MAKEFLAGS=-j$(($(grep -c ^processor /proc/cpuinfo) - 1))

PJ_ROOT=$(exec pwd)

DLIB_PATH=$PJ_ROOT/include/
DNN_ROOT=$PJ_ROOT/include/dnn
DNN_BUILD=$DNN_ROOT/build

IMG_ROOT=$PJ_ROOT/imglab
IMG_BUILD=$IMG_ROOT/build


echo $DLIB_PATH
echo $DNN_ROOT
echo $PJ_ROOT



#cd ${IMG_ROOT}
if [ ! -d ${IMG_BUILD} ]
  then
    echo "nodir"
    mkdir "${IMG_BUILD}"
  else
    echo "dir exists"
    #rm -rf "${IMG_ROOT}/build"
    #mkdir "${IMG_ROOT}/build"
fi

if [ ! -d ${DNN_BUILD} ]
  then
    echo "nodir"
    mkdir "${DNN_BUILD}"
    build_dnn
  else
    echo "dir exists"
    #rm -rf "${IMG_ROOT}/build"
    #mkdir "${IMG_ROOT}/build"
fi

build_img



