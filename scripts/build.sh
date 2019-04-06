#!/bin/sh

if [ ! -d build ]; then
  mkdir build
fi

cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE $SRC_DIR
cmake --build .
