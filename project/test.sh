#!/bin/bash

g++ -std=c++17 benchmark.cpp -o test_run -mavx -march=native -fopenmp `pkg-config --clfags --libs opencv` -lpthread

./test_run $1 $2