#!/bin/bash

UVER=`echo $1 | sed 's/\./_/g'`

if [ ! -f /tmp/boost_$UVER.tar.gz ]; then
wget --directory-prefix=/tmp/ https://dl.bintray.com/boostorg/release/$1/source/boost_$UVER.tar.gz
fi

if [ ! -d $2/boost ]; then
tar -xzf /tmp/boost_$UVER.tar.gz -C $2
mv $2/boost_$UVER $2/boost
fi