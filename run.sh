#!/bin/bash
export LD_LIBRARY_PATH=$HOME/local/lib
export PKG_CONFIG_PATH=$HOME/CSC485C/pkgconfig

git pull
g++ -std=c++11 project/sobel.cpp -o project/sobel `pkg-config --cflags --libs opencv` -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread
./project/sobel project/images/rgb1.jpg