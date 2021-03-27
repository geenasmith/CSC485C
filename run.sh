#!/bin/bash
export LD_LIBRARY_PATH=$HOME/local/lib
export PKG_CONFIG_PATH=$HOME/CSC485C/pkgconfig

declare -a images=("frac0.png" "frac1.png" "frac2.png" "frac3.png" "frac4.png" "frac5.png" "frac6.png" "frac7.png" "frac8.png" "frac9.png")
declare -a test_cases=("0" "1" "2" "3" "4" "5" "6")
declare -a thread_count=("1")

cd project
mkdir -p outputs
mkdir -p outputs/basic
mkdir -p outputs/simd
mkdir -p outputs/time
g++ -std=c++17 benchmark.cpp -o ./sobel `pkg-config --cflags --libs opencv` -mavx -march=native -fopenmp

for tc in "${test_cases[@]}"; do for threads in "${thread_count[@]}"; do for img in "${images[@]}"; do 
    perf stat -o ./outputs/basic/${img}_${tc}_${threads} -e task-clock,cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses ./sobel "sampleset/${img}" "$tc" "$threads" > ./outputs/time/${img}_${tc}_${threads}
    perf stat -o ./outputs/simd/${img}_${tc}_${threads} -e fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.256b_packed_single ./sobel "sampleset/${img}" "$tc" "$threads"
done; done; done

python3 perfToCsv.py
