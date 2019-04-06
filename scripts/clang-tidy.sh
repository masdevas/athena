#!/bin/sh

if [ ! -d clang-tidy-build ]; then
  mkdir clang-tidy-build
fi

cd clang-tidy-build
CC=clang-8 CXX=clang++-8 cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON $SRC_DIR
run-clang-tidy-8.py -export-fixes=tidy-fixes.yaml -header-filter='include/*.h' -checks='-*,modernize-*, clang-analyzer-*,performance-*' src/* include/*
