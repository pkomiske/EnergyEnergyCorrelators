#!/bin/bash

yum install -y zlib-devel

# install boost
curl -LO https://dl.bintray.com/boostorg/release/1.75.0/source/boost_1_75_0.tar.gz
tar xf boost_1_75_0.tar.gz
rm -f boost_1_75_0.tar.gz
cd boost_1_75_0
./bootstrap.sh --prefix=/usr/local --with-libraries=iostreams,serialization
./b2 install
cd ..
rm -rf boost_1_75_0
