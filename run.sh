#!/bin/bash
export LD_LIBRARY_PATH=$HOME/local/lib
# export PKG_CONFIG_PATH=$HOME/CSC485C/pkgconfig

git pull
cd project
g++ -std=c++11 sobel.cpp -o sobel `pkg-config --cflags --libs opencv4` -isystem ../benchmark/include -L../benchmark/build/src -lbenchmark -lpthread
./sobel --benchmark_out=benchmarking.csv --benchmark_out_format=csv