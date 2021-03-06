#!/bin/bash

set -e

if [[ $1 ]]
then
    git clang-format-8 $1 --diff --extensions h,cpp > format-fixes.diff
else
    find include/ -name "*.h" | xargs clang-format-8 -i
    find src/ -iname "*.cpp" -o -iname "*.h" | xargs clang-format-8 -i
    find tests/ -iname "*.cpp" -o -iname "*.h" | xargs clang-format-8 -i
    git -c core.fileMode=false diff -U0 --no-color > format-fixes.diff
fi

git reset --hard HEAD

touch format-fixes.diff
