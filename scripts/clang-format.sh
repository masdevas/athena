#!/bin/sh

python3 /opt/run-clang-format.py --clang-format-executable=clang-format-8 -r src include tests > format-fixes.diff
