#!/bin/sh

if [ ! -d infer-tidy-build ]; then
  mkdir infer-tidy-build
fi

cd infer-tidy-build
infer compile -- cmake -DDISABLE_TESTS=ON $SRC_DIR
infer run -- make -j 4
