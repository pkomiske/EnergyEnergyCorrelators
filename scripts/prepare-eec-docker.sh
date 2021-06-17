#!/bin/bash

yum install -y zlib-devel

# install boost
curl -LO https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz
tar xf boost_1_76_0.tar.gz
rm -f boost_1_76_0.tar.gz
cd boost_1_76_0
./bootstrap.sh --prefix=/usr/local --with-libraries=iostreams,serialization
./b2 install
cd ..
rm -rf boost_1_76_0
