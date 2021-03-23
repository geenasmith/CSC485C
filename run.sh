#!/bin/bash

git pull
cd project
mkdir -p outputs
mkdir -p outputs/simd
mkdir -p outputs/basic

g++ -std=c++17 sobel_nonsimd.cpp -o ./bin/sobel_nonsimd `pkg-config --cflags --libs opencv`
g++ -std=c++17 sobel_simd.cpp -o ./bin/sobel_simd `pkg-config --cflags --libs opencv` -mavx -march=native

perf stat -o ./outputs/basic/non_simd -e task-clock,cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./bin/sobel_nonsimd
perf stat -o ./outputs/basic/simd -e task-clock,cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./bin/sobel_simd
perf stat -o ./outputs/simd/simd -e fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single ./bin/sobel_simd

python3 perfToCsv.py