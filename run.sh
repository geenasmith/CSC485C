#!/bin/bash
export LD_LIBRARY_PATH=$HOME/local/lib
export PKG_CONFIG_PATH=$HOME/CSC485C/pkgconfig

method=baseline
thread=1
cd project
mkdir -p outputs
mkdir -p outputs/basic
mkdir -p outputs/simd
g++ -std=c++17 benchmark.cpp -o ./$method$thread `pkg-config --cflags --libs opencv` -mavx -march=native -fopenmp -Wall
perf stat -o ./outputs/basic/$method$thread -e task-clock,cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./$method$thread
perf stat -o ./outputs/simd/$method$thread -e fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single ./$method$thread