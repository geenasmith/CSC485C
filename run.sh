#!/bin/bash

git pull
cd project
mkdir -p outputs
g++ -std=c++17 sobel_simd.cpp -o sobel `pkg-config --cflags --libs opencv` -mavx -march=native

perf stat -o ./outputs/simd_20000_basic.out -e task-clock,cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./sobel
perf stat -o ./outputs/simd_20000_simd.out -e fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single ./sobel